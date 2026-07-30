#!/usr/bin/env python3
"""
Run from the repository root with exactly two GPUs, for example:

Memory:
  uv run torchrun --standalone --nproc_per_node=2 \
      scripts/benchmark_optimizer_sharding.py --task memory --optimizer baseline

  uv run torchrun --standalone --nproc_per_node=2 \
      scripts/benchmark_optimizer_sharding.py --task memory --optimizer sharded

Timing:
  uv run torchrun --standalone --nproc_per_node=2 \
      scripts/benchmark_optimizer_sharding.py --task timing --optimizer baseline

  uv run torchrun --standalone --nproc_per_node=2 \
      scripts/benchmark_optimizer_sharding.py --task timing --optimizer sharded

The script writes per-rank and aggregate JSON/CSV files under
results/optimizer_state_sharding/.

By default, it uses the assignment's DDP adapter. Pass --ddp torch to use
PyTorch DistributedDataParallel instead.
"""

from __future__ import annotations

import argparse, csv, gc, json, math, os, statistics, time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as TorchDDP
from torch.optim import AdamW, Optimizer

from cs336_basics.model import BasicsTransformerLM
from tests.adapters import get_sharded_optimizer


MODEL_CONFIGS = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    },
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
    },
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20,
    },
    "xl": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
    },
    "10b": {
        "d_model": 4608,
        "d_ff": 12288,
        "num_layers": 50,
        "num_heads": 36,
    },
}


# Assignment-standard xl configuration.
CONFIG = {
    "vocab_size": 10_000,
    "context_length": 512,
    "d_model": 2560,
    "d_ff": 10_240,
    "num_layers": 32,
    "num_heads": 32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("memory", "timing"), required=True,
                        help="Run the memory experiment for part (a), or timing experiment for part (b).")
    parser.add_argument("--optimizer", choices=("baseline", "sharded"), required=True,
                        help="baseline = ordinary AdamW; sharded = your ShardedOptimizer adapter.")
    parser.add_argument("--ddp", choices=("assignment", "torch"), default="assignment",
                        help="Gradient-synchronization implementation used in both comparisons.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--foreach", action="store_true",
                        help="Enable AdamW foreach kernels. Keep this identical in both runs.")
    parser.add_argument("--autocast-bf16", action="store_true",
                        help=("Use BF16 autocast for forward/backward while keeping FP32 model parameters. "
                              "The assignment-standard run is FP32, so leave this off unless FP32 is OOM."))
    parser.add_argument("--output-dir", type=Path, default=Path("results/optimizer_state_sharding"))
    parser.add_argument(
            "--model-size",
            choices=("xl", "medium", "large", "small"),
            default="xl",
        )
    args = parser.parse_args()

    CONFIG.update(**MODEL_CONFIGS[args.model_size])

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.context_length <= 0:
        parser.error("--context-length must be positive")
    if args.context_length > CONFIG["context_length"]:
        parser.error(f"--context-length cannot exceed the model maximum ({CONFIG['context_length']}) in this script")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps cannot be negative")
    if args.measurement_steps <= 0:
        parser.error("--measurement-steps must be positive")

    return args


def setup_distributed() -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError("Launch this script with torchrun. Missing environment variables: " + ", ".join(missing))
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError(f"The assignment's standard configuration requires 2 GPUs, but WORLD_SIZE={world_size}")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl",device_id=device,)
    return rank, local_rank, world_size, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)
    dist.barrier()


def mib(num_bytes: int | float) -> float:
    return float(num_bytes) / (1024**2)


def unique_cuda_tensor_nbytes(tensors: Iterable[torch.Tensor], device: torch.device) -> int:
    total = 0
    seen: set[int] = set()
    for tensor in tensors:
        if tensor is None or not isinstance(tensor, torch.Tensor):
            continue
        if tensor.device != device:
            continue
        tensor_id = id(tensor)
        if tensor_id in seen:
            continue
        seen.add(tensor_id)
        total += tensor.numel() * tensor.element_size()
    return total


def nested_cuda_tensor_nbytes(value: Any, device: torch.device) -> int:
    """Recursively count CUDA tensors inside optimizer state values."""
    seen: set[int] = set()

    def visit(item: Any) -> int:
        if isinstance(item, torch.Tensor):
            if item.device != device:
                return 0
            item_id = id(item)
            if item_id in seen:
                return 0
            seen.add(item_id)
            return item.numel() * item.element_size()
        if isinstance(item, dict):
            return sum(visit(v) for v in item.values())
        if isinstance(item, (list, tuple)):
            return sum(visit(v) for v in item)
        return 0

    return visit(value)


def optimizer_state_nbytes(optimizer: Optimizer, device: torch.device) -> int:
    return nested_cuda_tensor_nbytes(optimizer.state, device)


def build_model(args: argparse.Namespace, device: torch.device) -> BasicsTransformerLM:
    config = dict(CONFIG)
    config["context_length"] = args.context_length
    with torch.device(device):
        return BasicsTransformerLM(**config)


def build_ddp(
    args: argparse.Namespace,
    model: torch.nn.Module,
    local_rank: int,
) -> tuple[torch.nn.Module, Any]:
    """
    Returns:
        training_model
        finish_backward callback, called after loss.backward()
    """
    if args.ddp == "torch":
        ddp_model = TorchDDP(model, device_ids=[local_rank], output_device=local_rank,
                             broadcast_buffers=False, gradient_as_bucket_view=True)

        def finish_backward(_: Optimizer) -> None:
            return None

        return ddp_model, finish_backward

    # Import lazily so --ddp torch works even if the assignment DDP adapter is
    # temporarily unfinished.
    from tests.adapters import ddp_on_after_backward, get_ddp

    ddp_model = get_ddp(model)

    def finish_backward(optimizer: Optimizer) -> None:
        ddp_on_after_backward(ddp_model, optimizer)

    return ddp_model, finish_backward


def build_optimizer(args: argparse.Namespace, parameters: Iterable[torch.nn.Parameter]) -> Optimizer:
    kwargs = {"lr": args.lr, "weight_decay": args.weight_decay, "betas": (0.9, 0.999), "eps": 1e-8,
              "foreach": args.foreach}
    if args.optimizer == "baseline":
        return AdamW(parameters, **kwargs)
    return get_sharded_optimizer(parameters, AdamW, **kwargs)


def make_batch(args: argparse.Namespace, rank: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 10_000 + rank)
    shape = (args.batch_size, args.context_length)
    input_ids = torch.randint(0, CONFIG["vocab_size"], shape, device=device, dtype=torch.long, generator=generator)
    targets = torch.randint(0, CONFIG["vocab_size"], shape, device=device, dtype=torch.long, generator=generator)
    return input_ids, targets


def autocast_context(args: argparse.Namespace):
    if args.autocast_bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def forward_backward(args: argparse.Namespace, training_model: torch.nn.Module, optimizer: Optimizer,
                      finish_backward: Any, input_ids: torch.Tensor, targets: torch.Tensor) -> None:
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(args):
        logits = training_model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    loss.backward()
    finish_backward(optimizer)
    del logits, loss


def train_iteration(args: argparse.Namespace, training_model: torch.nn.Module, optimizer: Optimizer,
                     finish_backward: Any, input_ids: torch.Tensor, targets: torch.Tensor) -> None:
    forward_backward(args, training_model, optimizer, finish_backward, input_ids, targets)
    optimizer.step()


def memory_record(*, checkpoint: str, rank: int, device: torch.device, model: torch.nn.Module,
                   optimizer: Optimizer, input_ids: torch.Tensor | None = None,
                   targets: torch.Tensor | None = None) -> dict[str, Any]:
    parameter_bytes = unique_cuda_tensor_nbytes(model.parameters(), device)
    gradient_bytes = unique_cuda_tensor_nbytes(
        (parameter.grad for parameter in model.parameters()),
        device,
    )
    buffer_bytes = unique_cuda_tensor_nbytes(model.buffers(), device)
    state_bytes = optimizer_state_nbytes(optimizer, device)

    data_tensors = [
        tensor for tensor in (input_ids, targets) if tensor is not None
    ]
    data_bytes = unique_cuda_tensor_nbytes(data_tensors, device)

    current_allocated = torch.cuda.memory_allocated(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    current_reserved = torch.cuda.memory_reserved(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    accounted = parameter_bytes + gradient_bytes + buffer_bytes + state_bytes + data_bytes

    return {
        "rank": rank,
        "checkpoint": checkpoint,
        "current_allocated_mib": mib(current_allocated),
        "peak_allocated_mib": mib(peak_allocated),
        "current_reserved_mib": mib(current_reserved),
        "peak_reserved_mib": mib(peak_reserved),
        "parameter_mib": mib(parameter_bytes),
        "gradient_mib": mib(gradient_bytes),
        "optimizer_state_mib": mib(state_bytes),
        "buffer_mib": mib(buffer_bytes),
        "input_and_target_mib": mib(data_bytes),
        "unattributed_current_mib": mib(max(current_allocated - accounted, 0)),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def common_metadata(args: argparse.Namespace, rank: int, world_size: int, device: torch.device) -> dict[str, Any]:
    return {"task": args.task, "optimizer": args.optimizer, "ddp": args.ddp, "rank": rank,
            "world_size": world_size, "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device), "batch_size_per_rank": args.batch_size,
            "context_length": args.context_length, "vocab_size": CONFIG["vocab_size"],
            "d_model": CONFIG["d_model"], "d_ff": CONFIG["d_ff"], "num_layers": CONFIG["num_layers"],
            "num_heads": CONFIG["num_heads"], "autocast_bf16": args.autocast_bf16,
            "adamw_foreach": args.foreach, "lr": args.lr, "weight_decay": args.weight_decay}


def rank_result_path(args: argparse.Namespace, rank: int) -> Path:
    return args.output_dir / f"{args.task}_{args.optimizer}_{args.ddp}_rank{rank}.json"


def summary_result_path(args: argparse.Namespace) -> Path:
    return args.output_dir / f"{args.task}_{args.optimizer}_{args.ddp}_summary.json"


def load_all_rank_results(args: argparse.Namespace, world_size: int) -> list[dict[str, Any]]:
    return [json.loads(rank_result_path(args, rank).read_text(encoding="utf-8")) for rank in range(world_size)]


def print_memory_summary(summary: dict[str, Any]) -> None:
    print("\nMemory results (maximum across ranks, MiB)")
    print(f"{'checkpoint':<24}{'current':>12}{'peak':>12}{'params':>12}{'grads':>12}{'opt state':>12}{'other':>12}")
    print("-" * 96)
    for row in summary["checkpoint_maxima"]:
        print(f"{row['checkpoint']:<24}{row['current_allocated_mib']:>12.1f}{row['peak_allocated_mib']:>12.1f}"
              f"{row['parameter_mib']:>12.1f}{row['gradient_mib']:>12.1f}{row['optimizer_state_mib']:>12.1f}"
              f"{row['unattributed_current_mib']:>12.1f}")


def summarize_memory(args: argparse.Namespace, rank_results: list[dict[str, Any]]) -> dict[str, Any]:
    checkpoints = ("after_model_initialization", "directly_before_optimizer_step", "directly_after_optimizer_step")
    numeric_fields = ("current_allocated_mib", "peak_allocated_mib", "current_reserved_mib", "peak_reserved_mib",
                      "parameter_mib", "gradient_mib", "optimizer_state_mib", "buffer_mib", "input_and_target_mib",
                      "unattributed_current_mib")

    maxima = []
    for checkpoint in checkpoints:
        records = [r for result in rank_results for r in result["records"] if r["checkpoint"] == checkpoint]
        row: dict[str, Any] = {"checkpoint": checkpoint}
        for field in numeric_fields:
            row[field] = max(float(record[field]) for record in records)
        maxima.append(row)

    summary = {"metadata": rank_results[0]["metadata"],
               "interpretation": "peak_allocated_mib cumulative from before model construction through the named "
                                 "checkpoint. current_allocated_mib is the active tensor memory at the checkpoint.",
               "checkpoint_maxima": maxima, "per_rank": rank_results}
    write_json(summary_result_path(args), summary)
    write_csv(args.output_dir / f"{args.task}_{args.optimizer}_{args.ddp}_summary.csv", maxima)
    return summary


def run_memory(args: argparse.Namespace, rank: int, local_rank: int, world_size: int, device: torch.device) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model = build_model(args, device)
    training_model, finish_backward = build_ddp(args, model, local_rank)
    optimizer = build_optimizer(args, training_model.parameters())

    synchronize(device)
    records = [memory_record(checkpoint="after_model_initialization", rank=rank, device=device,
                              model=model, optimizer=optimizer)]

    input_ids, targets = make_batch(args, rank, device)
    forward_backward(args, training_model, optimizer, finish_backward, input_ids, targets)
    synchronize(device)
    records.append(memory_record(checkpoint="directly_before_optimizer_step", rank=rank, device=device,
                                  model=model, optimizer=optimizer, input_ids=input_ids, targets=targets))

    optimizer.step()
    synchronize(device)
    records.append(memory_record(checkpoint="directly_after_optimizer_step", rank=rank, device=device,
                                  model=model, optimizer=optimizer, input_ids=input_ids, targets=targets))

    result = {"metadata": common_metadata(args, rank, world_size, device), "records": records}
    write_json(rank_result_path(args, rank), result)

    dist.barrier()
    if rank == 0:
        summary = summarize_memory(args, load_all_rank_results(args, world_size))
        print_memory_summary(summary)
        print(f"\nSaved: {summary_result_path(args)}")
    dist.barrier()


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_timing(args: argparse.Namespace, rank_results: list[dict[str, Any]]) -> dict[str, Any]:
    per_rank_times = [result["step_times_ms"] for result in rank_results]
    step_count = len(per_rank_times[0])
    if any(len(times) != step_count for times in per_rank_times):
        raise RuntimeError("Ranks produced different numbers of timing samples")

    global_times = [max(per_rank_times[r][s] for r in range(len(per_rank_times))) for s in range(step_count)]

    per_rank_summaries = []
    for result in rank_results:
        times = [float(v) for v in result["step_times_ms"]]
        per_rank_summaries.append({"rank": result["metadata"]["rank"],
                                   "mean_ms": statistics.mean(times),
                                   "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
                                   "min_ms": min(times), "max_ms": max(times)})

    global_mean_ms = statistics.mean(global_times)
    tokens_per_iteration = rank_results[0]["metadata"]["world_size"] * args.batch_size * args.context_length

    summary = {"metadata": rank_results[0]["metadata"], "warmup_steps": args.warmup_steps,
               "measurement_steps": args.measurement_steps, "global_step_times_ms": global_times,
               "global": {"mean_ms": global_mean_ms,
                          "std_ms": statistics.stdev(global_times) if len(global_times) > 1 else 0.0,
                          "median_ms": statistics.median(global_times), "p95_ms": percentile(global_times, 0.95),
                          "min_ms": min(global_times), "max_ms": max(global_times),
                          "tokens_per_second": tokens_per_iteration / (global_mean_ms / 1000.0)},
               "per_rank": per_rank_summaries, "raw_per_rank": rank_results}
    write_json(summary_result_path(args), summary)
    write_csv(args.output_dir / f"{args.task}_{args.optimizer}_{args.ddp}_summary.csv",
              per_rank_summaries + [{"rank": "global", **summary["global"]}])
    return summary


def print_timing_summary(summary: dict[str, Any]) -> None:
    g = summary["global"]
    print("\nTiming results (slowest-rank wall time)")
    print(f"mean   : {g['mean_ms']:.3f} ms/iteration\n"
          f"std    : {g['std_ms']:.3f} ms\n"
          f"median : {g['median_ms']:.3f} ms\n"
          f"p95    : {g['p95_ms']:.3f} ms\n"
          f"tokens : {g['tokens_per_second']:.1f} tokens/s")


def run_timing(args: argparse.Namespace, rank: int, local_rank: int, world_size: int, device: torch.device) -> None:
    gc.collect()
    torch.cuda.empty_cache()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model = build_model(args, device)
    training_model, finish_backward = build_ddp(args, model, local_rank)
    optimizer = build_optimizer(args, training_model.parameters())
    input_ids, targets = make_batch(args, rank, device)

    for _ in range(args.warmup_steps):
        train_iteration(args, training_model, optimizer, finish_backward, input_ids, targets)
        torch.cuda.synchronize(device)

    dist.barrier()
    torch.cuda.synchronize(device)

    step_times_ms: list[float] = []
    for _ in range(args.measurement_steps):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        train_iteration(args, training_model, optimizer, finish_backward, input_ids, targets)
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        step_times_ms.append(elapsed_ms)

    result = {"metadata": common_metadata(args, rank, world_size, device), "step_times_ms": step_times_ms}
    write_json(rank_result_path(args, rank), result)

    dist.barrier()
    if rank == 0:
        summary = summarize_timing(args, load_all_rank_results(args, world_size))
        print_timing_summary(summary)
        print(f"\nSaved: {summary_result_path(args)}")
    dist.barrier()


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, device = setup_distributed()

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            print(f"task={args.task}, optimizer={args.optimizer}, ddp={args.ddp}, "
                  f"world_size={world_size}, gpu={torch.cuda.get_device_name(device)}")

        if args.task == "memory":
            run_memory(args, rank, local_rank, world_size, device)
        else:
            run_timing(args, rank, local_rank, world_size, device)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()