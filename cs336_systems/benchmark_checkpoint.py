from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import BasicsTransformerLM


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


def split_layers(
    layers: Sequence[nn.Module],
    num_checkpoints: int,
) -> list[tuple[nn.Module, ...]]:
    """Split layers into `num_checkpoints` contiguous, nearly equal groups."""
    num_layers = len(layers)
    if not 1 <= num_checkpoints <= num_layers:
        raise ValueError(
            f"num_checkpoints must be in [1, {num_layers}], got {num_checkpoints}"
        )

    base_size, remainder = divmod(num_layers, num_checkpoints)
    group_sizes = [
        base_size + (1 if group_idx < remainder else 0)
        for group_idx in range(num_checkpoints)
    ]

    groups: list[tuple[nn.Module, ...]] = []
    start = 0
    for group_size in group_sizes:
        groups.append(tuple(layers[start : start + group_size]))
        start += group_size

    return groups


def run_layer_group(hidden: Tensor, layers: tuple[nn.Module, ...]) -> Tensor:
    """Run one contiguous group of Transformer blocks."""
    for layer in layers:
        hidden = layer(hidden)
    return hidden


def checkpointed_model_forward(
    model: BasicsTransformerLM,
    token_ids: Tensor,
    layer_groups: list[tuple[nn.Module, ...]],
) -> Tensor:
    """Run BasicsTransformerLM with one non-nested checkpoint per layer group."""
    hidden = model.token_embeddings(token_ids)

    for group in layer_groups:
        # Bind the current group as a default argument. Without `group=group`,
        # Python's late-bound closure could use the final group during backward.
        def run_group(x: Tensor, group: tuple[nn.Module, ...] = group) -> Tensor:
            return run_layer_group(x, group)

        hidden = checkpoint(
            run_group,
            hidden,
            use_reentrant=False,
        )

    hidden = model.ln_final(hidden)
    return model.lm_head(hidden)


def make_optimizer(
    name: str,
    model: nn.Module,
    learning_rate: float,
):
    if name == "pytorch":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if name == "my":
        from cs336_basics.optimizer import AdamW

        return AdamW(model.parameters(), lr=learning_rate)

    raise ValueError(f"Unknown optimizer: {name}")


def run_training_step(
    model: BasicsTransformerLM,
    layer_groups: list[tuple[nn.Module, ...]],
    inputs: Tensor,
    targets: Tensor,
    optimizer,
    vocab_size: int,
    mixed_precision: str,
) -> float:
    """Run forward, loss, backward, and optimizer; return the scalar loss."""
    if mixed_precision == "bf16":
        amp_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        amp_context = torch.autocast(device_type="cuda", enabled=False)

    with amp_context:
        logits = checkpointed_model_forward(model, inputs, layer_groups)

    # Keep the loss/reduction in FP32, matching benchmark_nvtx.py.
    loss = F.cross_entropy(
        logits.float().reshape(-1, vocab_size),
        targets.reshape(-1),
    )
    loss.backward()
    optimizer.step()

    return loss.detach().item()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark non-nested activation checkpointing with a selectable "
            "number of checkpoint groups."
        )
    )
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="xl")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--num-checkpoints",
        type=int,
        default=32,
        help=(
            "Number of contiguous checkpoint groups. For the 32-layer XL model, "
            "1 means one group of 32 blocks and 32 means one checkpoint per block."
        ),
    )
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measurement-steps", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--optimizer", choices=["pytorch", "my"], default="pytorch")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--mixed-precision",
        choices=["none", "bf16"],
        default="none",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if args.measurement_steps < 1:
        raise ValueError("measurement_steps must be at least 1")

    device = torch.device(args.device)
    config = MODEL_CONFIGS[args.model_size]
    num_layers = config["num_layers"]
    if not 1 <= args.num_checkpoints <= num_layers:
        raise ValueError(
            f"For model {args.model_size}, --num-checkpoints must be between "
            f"1 and {num_layers}"
        )

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        **config,
    ).to(device)
    model.train()

    layer_groups = split_layers(model.layers, args.num_checkpoints)
    group_sizes = [len(group) for group in layer_groups]

    inputs = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.context_length),
        device=device,
    )
    targets = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.context_length),
        device=device,
    )
    optimizer = make_optimizer(args.optimizer, model, args.learning_rate)

    print("Checkpoint benchmark configuration")
    print(f"  GPU:              {torch.cuda.get_device_name(device)}")
    print(f"  Model:            {args.model_size} ({num_layers} blocks)")
    print(f"  Context length:   {args.context_length}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Precision:        {args.mixed_precision}")
    print(f"  Optimizer:        {args.optimizer}")
    print(f"  Checkpoints:      {args.num_checkpoints}")
    print(f"  Group sizes:      {group_sizes}")
    print(f"  Warm-up steps:    {args.warmup_steps}")
    print(f"  Measured steps:   {args.measurement_steps}")

    # Warm-up also initializes AdamW state before peak-memory measurements.
    for warmup_idx in range(args.warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        run_training_step(
            model,
            layer_groups,
            inputs,
            targets,
            optimizer,
            args.vocab_size,
            args.mixed_precision,
        )
        torch.cuda.synchronize(device)
        print(f"Warm-up {warmup_idx + 1}/{args.warmup_steps} complete")

    optimizer.zero_grad(set_to_none=True)

    elapsed_ms_values: list[float] = []
    peak_gib_values: list[float] = []
    loss_values: list[float] = []

    for step_idx in range(args.measurement_steps):
        # Gradients from the previous step are not part of the next step's
        # starting state. Cached memory does not affect max_memory_allocated.
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        start = time.perf_counter()
        loss = run_training_step(
            model,
            layer_groups,
            inputs,
            targets,
            optimizer,
            args.vocab_size,
            args.mixed_precision,
        )
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)

        elapsed_ms_values.append(elapsed_ms)
        peak_gib_values.append(peak_gib)
        loss_values.append(loss)

        print(
            f"Step {step_idx + 1}/{args.measurement_steps}: "
            f"time={elapsed_ms:.3f} ms, peak={peak_gib:.3f} GiB, loss={loss:.6f}"
        )

    mean_time_ms = statistics.mean(elapsed_ms_values)
    std_time_ms = statistics.pstdev(elapsed_ms_values)
    max_peak_gib = max(peak_gib_values)
    mean_peak_gib = statistics.mean(peak_gib_values)

    print("\nSummary")
    print(f"  Total step time:  {mean_time_ms:.3f} +/- {std_time_ms:.3f} ms")
    print(f"  Maximum peak:     {max_peak_gib:.3f} GiB")
    print(f"  Mean peak:        {mean_peak_gib:.3f} GiB")

    output_path = args.output
    if output_path is None:
        output_path = Path("results") / (
            f"checkpoint_{args.model_size}_ctx{args.context_length}_"
            f"n{args.num_checkpoints}_{args.mixed_precision}.csv"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "model": args.model_size,
        "num_layers": num_layers,
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "precision": args.mixed_precision,
        "optimizer": args.optimizer,
        "num_checkpoints": args.num_checkpoints,
        "group_sizes": "-".join(str(size) for size in group_sizes),
        "warmup_steps": args.warmup_steps,
        "measurement_steps": args.measurement_steps,
        "mean_total_step_ms": mean_time_ms,
        "std_total_step_ms": std_time_ms,
        "max_peak_memory_gib": max_peak_gib,
        "mean_peak_memory_gib": mean_peak_gib,
        "final_loss": loss_values[-1],
        "gpu": torch.cuda.get_device_name(device),
    }

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)

    print(f"  CSV written to:   {output_path}")


if __name__ == "__main__":
    main()