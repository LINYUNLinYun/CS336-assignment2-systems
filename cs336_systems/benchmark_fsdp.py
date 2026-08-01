#!/usr/bin/env python3
"""
Run with exactly two GPUs. The script supports two comparable modes:

  baseline:
      A complete model replica on each GPU, with no FSDP communication.

  fsdp:
      The educational FSDP implementation, including weight all-gathers.

Both modes use the same model, local batch size, inputs, autocast dtype,
warm-up count, and timing method. Run each mode in a separate torchrun process
to avoid CUDA allocator/cache effects contaminating the comparison.

Examples:
  uv run torchrun --standalone --nproc_per_node=2 \
      cs336_systems/benchmark_fsdp_compare.py \
      --mode baseline --compute-dtype fp16 --skip-profile-step

  uv run torchrun --standalone --nproc_per_node=2 \
      cs336_systems/benchmark_fsdp_compare.py \
      --mode fsdp --compute-dtype fp16 --skip-profile-step
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import types
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.cuda.nvtx as nvtx

from cs336_basics.model import BasicsTransformerLM
from tests.adapters import get_fsdp



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


CONFIG = {
    "vocab_size": 10_000,
    "context_length": 512,
    "d_model": 2_560,
    "d_ff": 10_240,
    "num_layers": 32,
    "num_heads": 32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile XL FSDP forward all-gathers on two GPUs."
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "fsdp"),
        default="fsdp",
        help=(
            "baseline keeps a complete model replica on each rank; "
            "fsdp wraps the model with the assignment FSDP implementation."
        ),
    )
    parser.add_argument(
        "--compute-dtype",
        choices=("fp32", "fp16", "bf16"),
        default="fp16",
        help=(
            "Weight communication/compute dtype. fp32 passes compute_dtype=None; "
            "fp16/bf16 cast shards before all-gather."
        ),
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=4,
        help="Assignment-standard global batch size; divided evenly over ranks.",
    )
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/fsdp_accounting"),
    )
    parser.add_argument(
        "--skip-profile-step",
        action="store_true",
        help="Skip the final PROFILE_STEP NVTX range (useful for timing-only runs).",
    )

    # Defaults are the user's Section 6 measurements. Override them if the
    # final Section 6 run changes.
    parser.add_argument(
        "--section6-baseline-peak-mib",
        type=float,
        default=52_430.3,
        help="Peak MiB with ordinary replicated AdamW/DDP.",
    )
    parser.add_argument(
        "--section6-sharded-optimizer-peak-mib",
        type=float,
        default=41_118.7,
        help="Peak MiB with optimizer-state sharding from Section 6.",
    )
    parser.add_argument(
        "--section6-parameter-mib",
        type=float,
        default=12_995.9,
        help="Full replicated parameter memory per rank in Section 6.",
    )
    parser.add_argument(
        "--section6-gradient-mib",
        type=float,
        default=12_995.9,
        help="Full replicated gradient memory per rank in Section 6.",
    )
    parser.add_argument(
                "--model-size",
                choices=("xl", "medium", "large", "small"),
                default="xl",
            )
    # args = parser.parse_args()
    args = parser.parse_args()

    CONFIG.update(**MODEL_CONFIGS[args.model_size])
    
    if args.global_batch_size <= 0:
        parser.error("--global-batch-size must be positive")
    if not 1 <= args.context_length <= CONFIG["context_length"]:
        parser.error("--context-length must be in [1, 512]")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps cannot be negative")
    if args.measurement_steps <= 0:
        parser.error("--measurement-steps must be positive")
    return args


def setup_distributed() -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    missing = [
        name
        for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
        if name not in os.environ
    ]
    if missing:
        raise RuntimeError(
            "Launch with torchrun; missing environment variables: "
            + ", ".join(missing)
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError(
            f"This assignment benchmark requires WORLD_SIZE=2, got {world_size}"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size, device


def dtype_from_name(name: str) -> torch.dtype | None:
    if name == "fp32":
        return None
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def mib(num_bytes: int | float) -> float:
    return float(num_bytes) / (1024**2)


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)
    dist.barrier()


def fsdp_module_labels(fsdp_model: nn.Module) -> dict[int, str]:
    """Build module-index labels such as '17:layers.2.attn.q_proj'."""
    ordered = getattr(fsdp_model, "_ordered_sharded_modules", None)
    module_infos = getattr(fsdp_model, "_module_infos", None)
    wrapped = getattr(fsdp_model, "module", None)
    if ordered is None or module_infos is None or wrapped is None:
        raise RuntimeError(
            "The benchmark expects the educational FSDP implementation to expose "
            "module, _ordered_sharded_modules, and _module_infos."
        )

    name_by_id = {id(module): name for name, module in wrapped.named_modules()}
    labels: dict[int, str] = {}
    for index, module in enumerate(ordered):
        name = name_by_id.get(id(module), f"unnamed_{index}")
        labels[index] = f"{index}:{name}"
    return labels


def instrument_fsdp_with_nvtx(fsdp_model: nn.Module) -> dict[int, str]:
    """
    Add three useful classes of ranges without modifying the FSDP source file:

      FSDP_PREFETCH_LAUNCH[i:name]
          CPU-side launch of the asynchronous all-gather.

      FSDP_ACQUIRE[i:name]
          The point at which the layer consumes the prefetched weight. If the
          all-gather is late, the compute stream will visibly stall here before
          the layer's first GEMM.

      FSDP_LAYER_TOTAL[i:name]
          The complete sharded module call, including acquire and compute.
    """
    labels = fsdp_module_labels(fsdp_model)

    original_start = fsdp_model._start_weight_prefetch
    original_acquire = fsdp_model._acquire_full_weight

    def start_with_nvtx(self: nn.Module, module_index: int) -> Any:
        label = labels.get(module_index, f"{module_index}:out_of_range")
        with nvtx.range(f"FSDP_PREFETCH_LAUNCH[{label}]"):
            return original_start(module_index)

    def acquire_with_nvtx(
        self: nn.Module,
        info: Any,
        module_index: int,
    ) -> torch.Tensor:
        label = labels.get(module_index, f"{module_index}:unknown")
        with nvtx.range(f"FSDP_ACQUIRE[{label}]"):
            return original_acquire(info, module_index)

    fsdp_model._start_weight_prefetch = types.MethodType(
        start_with_nvtx, fsdp_model
    )
    fsdp_model._acquire_full_weight = types.MethodType(
        acquire_with_nvtx, fsdp_model
    )

    # Wrap each already-patched Linear/Embedding forward with a layer range.
    for index, module in enumerate(fsdp_model._ordered_sharded_modules):
        original_forward = module.forward
        label = labels[index]

        def layer_forward_with_nvtx(
            self: nn.Module,
            *args: Any,
            _original_forward=original_forward,
            _label=label,
            **kwargs: Any,
        ) -> Any:
            with nvtx.range(f"FSDP_LAYER_TOTAL[{_label}]"):
                return _original_forward(*args, **kwargs)

        module.forward = types.MethodType(layer_forward_with_nvtx, module)

    return labels


def write_layer_metadata(
    fsdp_model: nn.Module,
    labels: dict[int, str],
    compute_dtype: torch.dtype | None,
    output_dir: Path,
    rank: int,
    world_size: int,
    model_size: str,
) -> None:
    """Write per-layer weight and communication sizes for the write-up."""
    if rank != 0:
        return

    dtype = compute_dtype or torch.float32
    element_size = torch.empty((), dtype=dtype).element_size()
    rows: list[dict[str, Any]] = []

    for index, module in enumerate(fsdp_model._ordered_sharded_modules):
        info, _, kind = fsdp_model._module_infos[id(module)]
        local_shard_bytes = info.shard_numel * element_size
        gathered_bytes = info.padded_numel * element_size
        received_bytes = (world_size - 1) * local_shard_bytes
        rows.append(
            {
                "module_index": index,
                "module": labels[index].split(":", 1)[1],
                "kind": kind,
                "full_shape": list(info.full_shape),
                "full_numel": info.full_numel,
                "communication_dtype": str(dtype).replace("torch.", ""),
                "local_shard_mib": mib(local_shard_bytes),
                "received_per_rank_mib": mib(received_bytes),
                "gathered_buffer_mib": mib(gathered_bytes),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"fsdp_{model_size}_layer_communication.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / f"fsdp_{model_size}_layer_communication.json"
    json_path.write_text(json.dumps(rows, indent=2))
    print(f"Layer communication metadata: {csv_path}")


def build_model(
    args: argparse.Namespace,
    device: torch.device,
    rank: int,
) -> tuple[nn.Module, torch.Tensor, dict[int, str], torch.dtype | None]:
    """Build either the full-replica baseline or the FSDP model."""
    # Identical seeds make model initialization identical across ranks/modes.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    config = dict(CONFIG)
    config["context_length"] = args.context_length
    base_model = BasicsTransformerLM(**config).to(device)

    compute_dtype = dtype_from_name(args.compute_dtype)
    labels: dict[int, str] = {}

    if args.mode == "fsdp":
        model = get_fsdp(base_model, compute_dtype=compute_dtype)
        labels = instrument_fsdp_with_nvtx(model)

        # The wrapper owns the model now; remove the extra Python reference.
        del base_model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        model = base_model

    local_batch_size = args.global_batch_size // dist.get_world_size()
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 10_000 + rank)
    input_ids = torch.randint(
        0,
        config["vocab_size"],
        (local_batch_size, args.context_length),
        device=device,
        generator=generator,
    )

    return model, input_ids, labels, compute_dtype

def forward_once(
    model: nn.Module,
    input_ids: torch.Tensor,
    compute_dtype: torch.dtype | None,
    mode: str,
) -> torch.Tensor:
    """Run one training-mode forward under identical autocast settings."""
    range_name = "FSDP_FORWARD" if mode == "fsdp" else "BASELINE_FORWARD"
    with nvtx.range(range_name):
        if compute_dtype is None:
            return model(input_ids)
        with torch.autocast(device_type="cuda", dtype=compute_dtype):
            return model(input_ids)


def warm_up(
    model: nn.Module,
    input_ids: torch.Tensor,
    steps: int,
    device: torch.device,
    compute_dtype: torch.dtype | None,
    mode: str,
) -> None:
    for step in range(steps):
        with nvtx.range(f"WARMUP_STEP[{step}]"):
            logits = forward_once(model, input_ids, compute_dtype, mode)
        del logits
        torch.cuda.synchronize(device)
    dist.barrier()


def time_forward_passes(
    model: nn.Module,
    input_ids: torch.Tensor,
    steps: int,
    device: torch.device,
    compute_dtype: torch.dtype | None,
    mode: str,
) -> list[float]:
    times_ms: list[float] = []
    for step in range(steps):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with nvtx.range(f"TIMED_FORWARD[{step}]"):
            logits = forward_once(model, input_ids, compute_dtype, mode)
        end.record()
        end.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
        del logits
    dist.barrier()
    return times_ms

def gather_timing_results(
    local_times_ms: list[float],
    device: torch.device,
    rank: int,
) -> dict[str, Any] | None:
    local = torch.tensor(local_times_ms, dtype=torch.float64, device=device)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)

    if rank != 0:
        return None

    per_rank = [tensor.cpu().tolist() for tensor in gathered]
    max_rank_per_step = [
        max(per_rank[rank_index][step] for rank_index in range(len(per_rank)))
        for step in range(len(local_times_ms))
    ]
    mean_ms = statistics.mean(max_rank_per_step)
    std_ms = (
        statistics.stdev(max_rank_per_step)
        if len(max_rank_per_step) > 1
        else 0.0
    )
    return {
        "per_rank_ms": per_rank,
        "max_rank_per_step_ms": max_rank_per_step,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "median_ms": statistics.median(max_rank_per_step),
        "min_ms": min(max_rank_per_step),
        "max_ms": max(max_rank_per_step),
    }


def run_profile_step(
    model: nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
    compute_dtype: torch.dtype | None,
    mode: str,
) -> tuple[float, float]:
    """Run one NVTX-captured forward step on both ranks."""
    dist.barrier()
    torch.cuda.reset_peak_memory_stats(device)

    nvtx.range_push("PROFILE_STEP")
    try:
        dist.barrier()
        logits = forward_once(model, input_ids, compute_dtype, mode)
        torch.cuda.synchronize(device)
        del logits
        dist.barrier()
    finally:
        nvtx.range_pop()

    dist.barrier()
    peak_allocated_mib = mib(torch.cuda.max_memory_allocated(device))
    peak_reserved_mib = mib(torch.cuda.max_memory_reserved(device))
    return peak_allocated_mib, peak_reserved_mib

def print_memory_accounting(args: argparse.Namespace) -> dict[str, float]:
    """
    Estimate the FSDP peak from the Section 6 measurements.

    Starting from the optimizer-sharded peak, two-rank FSDP additionally halves
    replicated parameters and gradients. Activation memory is unchanged, and
    all-gather buffers are intentionally ignored as requested by the problem.
    """
    extra_saving_mib = (
        args.section6_parameter_mib + args.section6_gradient_mib
    ) / 2.0
    predicted_fsdp_peak_mib = (
        args.section6_sharded_optimizer_peak_mib - extra_saving_mib
    )
    total_saving_vs_baseline_mib = (
        args.section6_baseline_peak_mib - predicted_fsdp_peak_mib
    )
    total_saving_percent = (
        100.0
        * total_saving_vs_baseline_mib
        / args.section6_baseline_peak_mib
    )

    result = {
        "additional_saving_vs_sharded_optimizer_mib": extra_saving_mib,
        "predicted_fsdp_peak_mib": predicted_fsdp_peak_mib,
        "total_saving_vs_baseline_mib": total_saving_vs_baseline_mib,
        "total_saving_vs_baseline_percent": total_saving_percent,
    }
    print("\nSection 6 -> FSDP accounting (ignoring all-gather buffers):")
    print(
        f"  additional parameter+gradient saving: {extra_saving_mib:.1f} MiB "
        f"({extra_saving_mib / 1024:.2f} GiB)"
    )
    print(
        f"  predicted FSDP peak: {predicted_fsdp_peak_mib:.1f} MiB "
        f"({predicted_fsdp_peak_mib / 1024:.2f} GiB)"
    )
    print(
        f"  saving vs ordinary baseline: {total_saving_vs_baseline_mib:.1f} MiB "
        f"({total_saving_vs_baseline_mib / 1024:.2f} GiB, "
        f"{total_saving_percent:.2f}%)"
    )
    return result


def main() -> None:
    args = parse_args()
    rank = local_rank = world_size = -1
    try:
        rank, local_rank, world_size, device = setup_distributed()
        if args.global_batch_size % world_size != 0:
            raise ValueError(
                "--global-batch-size must be divisible by WORLD_SIZE"
            )

        if rank == 0:
            print(
                f"{args.mode.upper()} {args.model_size.upper()} benchmark: "
                f"GPU={torch.cuda.get_device_name(local_rank)}, "
                f"world_size={world_size}, global_batch={args.global_batch_size}, "
                f"local_batch={args.global_batch_size // world_size}, "
                f"context={args.context_length}, compute_dtype={args.compute_dtype}"
            )

        model, input_ids, labels, compute_dtype = build_model(
            args, device, rank
        )
        synchronize(device)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.mode == "fsdp":
            write_layer_metadata(
                model,
                labels,
                compute_dtype,
                args.output_dir,
                rank,
                world_size,
                args.model_size,
            )

        warm_up(
            model,
            input_ids,
            args.warmup_steps,
            device,
            compute_dtype,
            args.mode,
        )
        local_times = time_forward_passes(
            model,
            input_ids,
            args.measurement_steps,
            device,
            compute_dtype,
            args.mode,
        )
        timing = gather_timing_results(local_times, device, rank)

        profile_peak_allocated = profile_peak_reserved = None
        if not args.skip_profile_step:
            profile_peak_allocated, profile_peak_reserved = run_profile_step(
                model,
                input_ids,
                device,
                compute_dtype,
                args.mode,
            )

        if rank == 0:
            assert timing is not None
            accounting = (
                print_memory_accounting(args)
                if args.mode == "fsdp"
                else None
            )
            print("\nForward timing (max of the two ranks for each step):")
            print(
                f"  mean={timing['mean_ms']:.3f} ms, "
                f"std={timing['std_ms']:.3f} ms, "
                f"median={timing['median_ms']:.3f} ms, "
                f"min={timing['min_ms']:.3f} ms, "
                f"max={timing['max_ms']:.3f} ms"
            )
            if profile_peak_allocated is not None:
                print(
                    "  rank-0 PROFILE_STEP peak: "
                    f"allocated={profile_peak_allocated:.1f} MiB, "
                    f"reserved={profile_peak_reserved:.1f} MiB"
                )

            summary = {
                "mode": args.mode,
                "model": args.model_size,
                "model_config": {
                    **CONFIG,
                    "context_length": args.context_length,
                },
                "world_size": world_size,
                "global_batch_size": args.global_batch_size,
                "local_batch_size": args.global_batch_size // world_size,
                "compute_dtype": args.compute_dtype,
                "gpu": torch.cuda.get_device_name(local_rank),
                "timing": timing,
                "section6_accounting": accounting,
                "profile_step_rank0_peak_allocated_mib": profile_peak_allocated,
                "profile_step_rank0_peak_reserved_mib": profile_peak_reserved,
            }
            output_path = (
                args.output_dir
                / f"{args.mode}_{args.model_size}_{args.compute_dtype}_summary.json"
            )
            output_path.write_text(json.dumps(summary, indent=2))
            print(f"Summary: {output_path}")

        synchronize(device)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()