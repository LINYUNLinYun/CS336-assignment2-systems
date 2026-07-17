import argparse
import math
import timeit
import statistics
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
from einops import einsum
from jaxtyping import Bool, Float
from torch import Tensor

import cs336_basics.model
from cs336_basics.model import BasicsTransformerLM, TransformerBlock
from cs336_basics.nn_utils import softmax


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


def make_autocast_context(mixed_precision: str):
    """Return the context manager used for the forward pass."""
    if mixed_precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if mixed_precision == "none":
        return nullcontext()
    raise ValueError(f"Unsupported mixed precision mode: {mixed_precision}")


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention with NVTX ranges for profiling."""
    d_k = K.shape[-1]

    # QK^T matrix multiplication + scaling.
    with nvtx.range("my_attn_scores"):
        attention_scores = (
            einsum(Q, K, "... query d_k, ... key d_k -> ... query key")
            / math.sqrt(d_k)
        )

    # Masking is separate from the score matmul so that the matmul timing is clean.
    if mask is not None:
        with nvtx.range("my_attn_mask"):
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("my_attn_softmax"):
        attention_weights = softmax(attention_scores, dim=-1)

    with nvtx.range("my_attn_output"):
        output = einsum(
            attention_weights,
            V,
            "... query key, ... key d_v -> ... query d_v",
        )

    return output

# Save the original implementation before monkey-patching.
_original_transformer_block_forward = TransformerBlock.forward


def annotated_transformer_block_forward(
    self: TransformerBlock,
    x: Tensor,
):
    """Wrap one complete TransformerBlock forward pass in an NVTX range."""
    block_idx = getattr(self, "_nvtx_block_idx", -1)

    with nvtx.range(f"transformer_block_{block_idx}_forward"):
        return _original_transformer_block_forward(self, x)
    
def transformer_block_backward_pre_hook(
    module: TransformerBlock,
    grad_output,
):
    block_idx = getattr(module, "_nvtx_block_idx", -1)
    nvtx.range_push(f"transformer_block_{block_idx}_backward")


def transformer_block_backward_hook(
    module: TransformerBlock,
    grad_input,
    grad_output,
):
    nvtx.range_pop()


def run_step(mode: str, measure: bool = False, annotate: bool = False):
    timings = {}

    step_ctx = nvtx.range("my_train_step") if annotate else nullcontext()

    with step_ctx:
        # ---------------- Zero grad ----------------
        zero_grad_ctx = nvtx.range("my_zero_grad") if annotate else nullcontext()
        with zero_grad_ctx:
            optimizer.zero_grad(set_to_none=True)
        
        # ---------------- Forward ----------------
        if measure:
            torch.cuda.synchronize()
            start = timeit.default_timer()

        forward_ctx = nvtx.range("my_forward") if annotate else nullcontext()
        amp_ctx = make_autocast_context(args.mixed_precision)
        grad_ctx = torch.no_grad() if mode == "forward" else nullcontext()

        with forward_ctx:
            with grad_ctx:
                with amp_ctx:
                    logits = model(inputs)

            if measure:
                torch.cuda.synchronize()

        if measure:
            timings["forward"] = timeit.default_timer() - start

        if mode == "forward":
            return timings

        # ---------------- Loss ----------------
        loss_ctx = nvtx.range("my_cross_loss") if annotate else nullcontext()
        with loss_ctx:
            # Keep loss/reduction in FP32 for numerical stability.
            loss = F.cross_entropy(
                logits.float().reshape(-1, args.vocab_size),
                targets.reshape(-1),
            )

            if measure:
                torch.cuda.synchronize()

        # ---------------- Backward ----------------
        if measure:
            torch.cuda.synchronize()
            start = timeit.default_timer()

        backward_ctx = nvtx.range("my_backward") if annotate else nullcontext()
        with backward_ctx:
            loss.backward()

            if measure:
                torch.cuda.synchronize()

        if measure:
            timings["backward"] = timeit.default_timer() - start

        if mode == "forward_backward":
            return timings

        # ---------------- Optimizer ----------------
        if measure:
            torch.cuda.synchronize()
            start = timeit.default_timer()

        optimizer_ctx = nvtx.range("my_optimizer_step") if annotate else nullcontext()
        with optimizer_ctx:
            optimizer.step()

            if measure:
                torch.cuda.synchronize()

        if measure:
            timings["optimizer"] = timeit.default_timer() - start

    return timings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-size",
        choices=MODEL_CONFIGS,
        default="small",
    )
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--mode",
        choices=["forward", "forward_backward", "full_step"],
        default="full_step",
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--optimizer", default="pytorch", choices=["pytorch", "my"])
    parser.add_argument(
        "--mixed-precision",
        default="none",
        choices=["none", "bf16"],
        help="Mixed precision mode: none or bf16",
    )
    # 内存记录
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Record a CUDA memory snapshot",
    )

    parser.add_argument(
        "--memory-snapshot",
        default="memory_snapshot.pickle",
    )
    # 标记tranformer block 
    parser.add_argument("--annotate-blocks", action="store_true", help="Annotate each TransformerBlock with NVTX ranges")

    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model_size]

    # Monkey-patch the attention implementation before model construction.
    cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    if args.annotate_blocks:
        TransformerBlock.forward = annotated_transformer_block_forward

    device = torch.device(args.device)
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        **config,
    ).to(device)
    # backward hook
    if args.annotate_blocks:
        backward_hook_handles = []
        for block_idx, block in enumerate(model.layers):
            # 为block编号
            block._nvtx_block_idx = block_idx

            pre_handle = block.register_full_backward_pre_hook(
                transformer_block_backward_pre_hook
            )
            post_handle = block.register_full_backward_hook(
                transformer_block_backward_hook
            )

            backward_hook_handles.extend([pre_handle, post_handle])

    model.train()
    print(model)

    inputs = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        device=device,
    )

    targets = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        device=device,
    )

    if args.optimizer == "pytorch":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
        )
    elif args.optimizer == "my":
        print("using my optimizer --------------")
        from cs336_basics.optimizer import AdamW

        optimizer = AdamW(
            model.parameters(),
            lr=1e-3,
        )
    else:
        raise RuntimeError("invalid optimizer")

    print(f"Precision mode: {args.mixed_precision}")
    print(f"Warming up for {args.warmup_steps} steps...")

    # Warm-up steps.
    for _ in range(args.warmup_steps):
        run_step(args.mode, measure=False, annotate=False)
        torch.cuda.synchronize()

    results = {
        "forward": [],
        "backward": [],
        "optimizer": [],
    }
    # 清除最后一次 warm-up 留下的梯度
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    # Start recording memory history. 热身后
    if args.profile_memory:
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)
        torch.cuda.reset_peak_memory_stats()

        run_step(
            args.mode,
            measure=False,
            annotate=True,
        )

        torch.cuda.synchronize()
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak allocated memory: {peak_gib:.2f} GiB")

        torch.cuda.memory._dump_snapshot(
            args.memory_snapshot
        )

        torch.cuda.memory._record_memory_history(
            enabled=None
        )

        print(f"Memory snapshot saved to: {args.memory_snapshot}")
        raise SystemExit(0)

    print(f"Measuring {args.measurement_steps} steps...")

    # Measurement steps.
    with nvtx.range("my_measurement_region"):
        for _ in range(args.measurement_steps):
            timings = run_step(
                args.mode,
                measure=True,
                annotate=True,
            )

            for name, elapsed in timings.items():
                results[name].append(elapsed * 1000)  # seconds -> ms

    print("\nBenchmark results:")

    records = []

    for name, values in results.items():
        if not values:
            continue

        mean_ms = statistics.mean(values)
        std_ms = statistics.pstdev(values)

        print(f"{name:>10}: {mean_ms:.3f} ± {std_ms:.3f} ms")

        records.append(
            {
                "model": args.model_size,
                "batch_size": args.batch_size,
                "context_length": args.context_length,
                "mode": args.mode,
                "phase": name,
                "warmup_steps": args.warmup_steps,
                "measurement_steps": args.measurement_steps,
                "mean_ms": mean_ms,
                "std_ms": std_ms,
                "gpu": torch.cuda.get_device_name(device),
                "precision": args.mixed_precision,
            }
        )

    df = pd.DataFrame(records)

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / (
        f"{args.model_size}_{args.mode}_"
        f"{args.mixed_precision}_warmup{args.warmup_steps}.csv"
    )

    df.to_csv(output_file, index=False)

    print(f"\nResults saved to: {output_file}")

    print("\nLaTeX table:")
    print(
        df.to_latex(
            index=False,
            float_format="%.3f",
        )
    )
