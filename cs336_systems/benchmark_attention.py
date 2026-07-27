#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch

from cs336_basics.model import scaled_dot_product_attention


BATCH_SIZE = 8
HEAD_DIMS = [16, 32, 64, 128]
SEQUENCE_LENGTHS = [256, 1024, 4096, 8192, 16384]


def synchronize(device: torch.device) -> None:
    """Wait until all queued work on the selected CUDA device has finished."""
    torch.cuda.synchronize(device)


def clear_cuda_memory(device: torch.device) -> None:
    """Release Python references and clear unused cached CUDA blocks."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def make_inputs(
    *,
    batch_size: int,
    sequence_length: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create Q, K and V with shape [batch, sequence_length, head_dim]."""
    shape = (batch_size, sequence_length, head_dim)

    q = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)

    return q, k, v


def eager_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    return scaled_dot_product_attention(q, k, v, mask=None)


def make_attention_fn(implementation: str) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    if implementation == "eager":
        return eager_attention
    if implementation == "compiled":
        return torch.compile(eager_attention)
    raise ValueError(f"Unsupported implementation: {implementation}")

def mean_and_std(values_ms: list[float]) -> tuple[float, float]:
    if not values_ms:
        raise ValueError("Cannot summarize an empty timing list.")

    mean_ms = statistics.mean(values_ms)
    std_ms = statistics.pstdev(values_ms)
    return mean_ms, std_ms


def warm_up_forward(
    *,
    attention_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    warmup_steps: int,
    device: torch.device,
) -> None:
    """
    Warm up the no-grad forward path.

    torch.compile may specialize this path separately from grad-enabled
    execution, so it must be warmed up before benchmark_forward().
    """
    with torch.no_grad():
        for _ in range(warmup_steps):
            output = attention_fn(q, k, v)
            synchronize(device)
            del output


def warm_up_backward(
    *,
    attention_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    warmup_steps: int,
    device: torch.device,
) -> None:
    """Warm up the grad-enabled forward and backward paths."""
    for _ in range(warmup_steps):
        q.grad = None
        k.grad = None
        v.grad = None

        output = attention_fn(q, k, v)
        grad_output = torch.randn_like(output)
        output.backward(grad_output)

        synchronize(device)
        del output, grad_output


def benchmark_forward(
    *,
    attention_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    measurement_steps: int,
    device: torch.device,
) -> list[float]:
    """
    Time forward-only inference.

    torch.no_grad() is used so this measures forward compute without retaining
    an autograd graph for backward.
    """
    times_ms: list[float] = []

    with torch.no_grad():
        for _ in range(measurement_steps):
            synchronize(device)
            start = time.perf_counter()

            output = attention_fn(q, k, v)

            synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            times_ms.append(elapsed_ms)

            del output

    return times_ms


def measure_memory_before_backward(
    *,
    attention_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float, float, float]:
    """
    Run one grad-enabled forward pass and measure memory before backward.

    Returns:
        output
        grad_output
        total allocated memory after forward (MiB)
        extra allocated memory caused by forward/saved tensors (MiB)
        peak allocated memory during forward (MiB)
    """
    q.grad = None
    k.grad = None
    v.grad = None

    synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    baseline_bytes = torch.cuda.memory_allocated(device)

    output = attention_fn(q, k, v)
    grad_output = torch.randn_like(output)

    synchronize(device)

    after_forward_bytes = torch.cuda.memory_allocated(device)
    peak_bytes = torch.cuda.max_memory_allocated(device)

    total_mib = after_forward_bytes / 1024**2
    forward_delta_mib = (after_forward_bytes - baseline_bytes) / 1024**2
    peak_mib = peak_bytes / 1024**2

    return output, grad_output, total_mib, forward_delta_mib, peak_mib


def benchmark_backward(
    *,
    attention_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    measurement_steps: int,
    device: torch.device,
    first_output: torch.Tensor | None = None,
    first_grad_output: torch.Tensor | None = None,
) -> list[float]:
    """
    Time backward only.

    Each iteration builds a fresh graph before timing. The forward pass is
    deliberately outside the timed interval, so the recorded value represents
    backward execution rather than forward + backward.
    """
    times_ms: list[float] = []

    for step in range(measurement_steps):
        q.grad = None
        k.grad = None
        v.grad = None
        # 如果指定了 first_output 和 first_grad_output，则在第一次迭代中使用它们，而不是重新计算 forward。
        if step == 0 and first_output is not None and first_grad_output is not None:
            output = first_output
            grad_output = first_grad_output
        else:
            output = attention_fn(q, k, v)
            grad_output = torch.randn_like(output)

        synchronize(device)
        start = time.perf_counter()

        output.backward(grad_output)

        synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)

        del output, grad_output

    return times_ms


def benchmark_one_configuration(
    *,
    attention_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    implementation: str,
    sequence_length: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup_steps: int,
    measurement_steps: int,
) -> dict[str, Any]:
    """Benchmark one (sequence_length, head_dim) configuration."""
    clear_cuda_memory(device)

    q, k, v = make_inputs(
        batch_size=BATCH_SIZE,
        sequence_length=sequence_length,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )

    input_memory_mib = (
        q.numel() * q.element_size()
        + k.numel() * k.element_size()
        + v.numel() * v.element_size()
    ) / 1024**2

    # Warm up the two execution modes separately. A compiled no-grad graph
    # can differ from the grad-enabled graph used for backward.
    warm_up_forward(
        attention_fn=attention_fn,
        q=q,
        k=k,
        v=v,
        warmup_steps=warmup_steps,
        device=device,
    )
    warm_up_backward(
        attention_fn=attention_fn,
        q=q,
        k=k,
        v=v,
        warmup_steps=warmup_steps,
        device=device,
    )

    q.grad = None
    k.grad = None
    v.grad = None
    clear_cuda_memory(device)

    forward_times_ms = benchmark_forward(
        attention_fn=attention_fn,
        q=q,
        k=k,
        v=v,
        measurement_steps=measurement_steps,
        device=device,
    )
    forward_mean_ms, forward_std_ms = mean_and_std(forward_times_ms)

    (
        saved_output,
        saved_grad_output,
        allocated_before_backward_mib,
        forward_saved_delta_mib,
        forward_peak_mib,
    ) = measure_memory_before_backward(
        attention_fn=attention_fn,
        q=q,
        k=k,
        v=v,
        device=device,
    )

    backward_times_ms = benchmark_backward(
        attention_fn=attention_fn,
        q=q,
        k=k,
        v=v,
        measurement_steps=measurement_steps,
        device=device,
        first_output=saved_output,
        first_grad_output=saved_grad_output,
    )
    backward_mean_ms, backward_std_ms = mean_and_std(backward_times_ms)

    final_peak_mib = torch.cuda.max_memory_allocated(device) / 1024**2

    result = {
        "implementation": implementation,
        "batch_size": BATCH_SIZE,
        "sequence_length": sequence_length,
        "head_dim": head_dim,
        "dtype": str(dtype).replace("torch.", ""),
        "input_qkv_memory_mib": input_memory_mib,
        "allocated_before_backward_mib": allocated_before_backward_mib,
        "forward_saved_delta_mib": forward_saved_delta_mib,
        "forward_peak_memory_mib": forward_peak_mib,
        "overall_peak_memory_mib": final_peak_mib,
        "forward_mean_ms": forward_mean_ms,
        "forward_std_ms": forward_std_ms,
        "backward_mean_ms": backward_mean_ms,
        "backward_std_ms": backward_std_ms,
        "warmup_steps": warmup_steps,
        "measurement_steps": measurement_steps,
        "gpu": torch.cuda.get_device_name(device),
        "status": "success",
        "error": "",
    }

    del q, k, v
    clear_cuda_memory(device)

    return result


def oom_result(
    *,
    implementation: str,
    sequence_length: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup_steps: int,
    measurement_steps: int,
    error: BaseException,
) -> dict[str, Any]:
    return {
        "implementation": implementation,
        "batch_size": BATCH_SIZE,
        "sequence_length": sequence_length,
        "head_dim": head_dim,
        "dtype": str(dtype).replace("torch.", ""),
        "input_qkv_memory_mib": None,
        "allocated_before_backward_mib": None,
        "forward_saved_delta_mib": None,
        "forward_peak_memory_mib": None,
        "overall_peak_memory_mib": None,
        "forward_mean_ms": None,
        "forward_std_ms": None,
        "backward_mean_ms": None,
        "backward_std_ms": None,
        "warmup_steps": warmup_steps,
        "measurement_steps": measurement_steps,
        "gpu": torch.cuda.get_device_name(device),
        "status": "OOM",
        "error": str(error).replace("\n", " "),
    }


def save_results(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the CS336 PyTorch attention implementation."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/attention/pytorch_attention_results.csv"),)
    parser.add_argument(
        "--implementation",
        choices=("eager", "compiled"),
        default="eager",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA-capable GPU.")

    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative.")
    if args.measurement_steps <= 0:
        raise ValueError("--measurement-steps must be positive.")

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    attention_fn = make_attention_fn(args.implementation)

    total_configs = len(HEAD_DIMS) * len(SEQUENCE_LENGTHS)

    print("PyTorch attention benchmark")
    print(f"Implementation:    {args.implementation}")
    print(f"GPU:               {torch.cuda.get_device_name(device)}")
    print(f"Device:            {device}")
    print(f"Dtype:             {dtype}")
    print(f"Batch size:        {BATCH_SIZE}")
    print(f"Warm-up steps:     {args.warmup_steps}")
    print(f"Measurement steps: {args.measurement_steps}")
    print(f"Configurations:    {total_configs}")
    if(args.implementation == "compiled"):
        output_file = args.output / "pytorch_attention_fp32_compiled.csv"
    else:
        output_file = args.output / "pytorch_attention_fp32.csv"
    print(f"Output:            {output_file}")

    records: list[dict[str, Any]] = []
    config_index = 0

    for head_dim in HEAD_DIMS:
        for sequence_length in SEQUENCE_LENGTHS:
            config_index += 1
            print("\n" + "=" * 72)
            print(
                f"[{config_index}/{total_configs}] "
                f"sequence_length={sequence_length}, head_dim={head_dim}"
            )

            try:
                result = benchmark_one_configuration(
                    attention_fn=attention_fn,
                    implementation=args.implementation,
                    sequence_length=sequence_length,
                    head_dim=head_dim,
                    dtype=dtype,
                    device=device,
                    warmup_steps=args.warmup_steps,
                    measurement_steps=args.measurement_steps,
                )

                print(
                    f"forward:  {result['forward_mean_ms']:.3f} "
                    f"± {result['forward_std_ms']:.3f} ms"
                )
                print(
                    f"backward: {result['backward_mean_ms']:.3f} "
                    f"± {result['backward_std_ms']:.3f} ms"
                )
                print(
                    "memory before backward: "
                    f"{result['allocated_before_backward_mib']:.2f} MiB"
                )
                print(
                    "forward allocation delta: "
                    f"{result['forward_saved_delta_mib']:.2f} MiB"
                )

            except torch.OutOfMemoryError as error:
                print("CUDA OOM")
                result = oom_result(
                    implementation=args.implementation,
                    sequence_length=sequence_length,
                    head_dim=head_dim,
                    dtype=dtype,
                    device=device,
                    warmup_steps=args.warmup_steps,
                    measurement_steps=args.measurement_steps,
                    error=error,
                )
                clear_cuda_memory(device)

            except RuntimeError as error:
                if "out of memory" not in str(error).lower():
                    raise

                print("CUDA OOM")
                result = oom_result(
                    implementation=args.implementation,
                    sequence_length=sequence_length,
                    head_dim=head_dim,
                    dtype=dtype,
                    device=device,
                    warmup_steps=args.warmup_steps,
                    measurement_steps=args.measurement_steps,
                    error=error,
                )
                clear_cuda_memory(device)

            records.append(result)

            # Save after every configuration so partial results survive interruption.
            save_results(records, output_file)

    dataframe = pd.DataFrame(records)

    print("\n" + "=" * 72)
    print("Final results")
    display_columns = [
        "implementation",
        "sequence_length",
        "head_dim",
        "dtype",
        "forward_mean_ms",
        "backward_mean_ms",
        "allocated_before_backward_mib",
        "forward_saved_delta_mib",
        "status",
    ]
    print(dataframe[display_columns].to_string(index=False))
    print(f"\nSaved CSV to: {output_file}")


if __name__ == "__main__":
    main()