import argparse
import math
import torch
import timeit
import statistics
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
import cs336_basics.model
from contextlib import nullcontext
import torch.cuda.nvtx as nvtx
from einops import einsum, rearrange
import einx
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor
from cs336_basics.nn_utils import softmax
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

def run_step(
    mode: str,
    measure: bool = False,
    annotate: bool = False,     # 是否开启 nvtx 标记
):
    timings = {}

    step_ctx = nvtx.range("my_train_step") if annotate else nullcontext()

    with step_ctx:
        # ---------------- Zero grad ----------------
        zero_grad_ctx = (
            nvtx.range("my_zero_grad")
            if annotate
            else nullcontext()
        )

        with zero_grad_ctx:
            optimizer.zero_grad(set_to_none=True)

        # ---------------- Forward ----------------
        if measure:
            torch.cuda.synchronize()
            start = timeit.default_timer()

        forward_ctx = (
            nvtx.range("my_forward")
            if annotate
            else nullcontext()
        )

        with forward_ctx:
            logits = model(inputs)

            # 让 my_forward 包含 GPU forward kernels 完成时间
            if measure:
                torch.cuda.synchronize()

        if measure:
            timings["forward"] = timeit.default_timer() - start

        if mode == "forward":
            return timings

        # ---------------- Loss ----------------
        loss_ctx = (
            nvtx.range("my_cross_loss")
            if annotate
            else nullcontext()
        )

        with loss_ctx:
            loss = F.cross_entropy(
                logits.reshape(-1, args.vocab_size),
                targets.reshape(-1),
            )

            # 如果你想让 my_cross_loss 的 range 更准确，可以保留
            if measure:
                torch.cuda.synchronize()

        # ---------------- Backward ----------------
        if measure:
            torch.cuda.synchronize()
            start = timeit.default_timer()

        backward_ctx = (
            nvtx.range("my_backward")
            if annotate
            else nullcontext()
        )

        with backward_ctx:
            loss.backward()

            # 让 my_backward 包含 GPU backward kernels 完成时间
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

        optimizer_ctx = (
            nvtx.range("my_optimizer_step")
            if annotate
            else nullcontext()
        )

        with optimizer_ctx:
            optimizer.step()

            # 让 my_optimizer_step 包含 GPU optimizer kernels 完成时间
            if measure:
                torch.cuda.synchronize()

        if measure:
            timings["optimizer"] = timeit.default_timer() - start

    return timings


    

@nvtx.range("scaled dot product attention") 
def annotated_scaled_dot_product_attention(
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys    d_k"],
        V: Float[Tensor, " ... keys    d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None,
    ) -> Float[Tensor, " ... queries d_v"]:

    d_k = K.shape[-1]
    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")

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
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model_size]
    # 替换实现
    cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    device = torch.device(args.device)
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        **config,
    ).to(args.device)

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    print(f"Warming up for {args.warmup_steps} steps...")

    # 空跑几轮 预热
    for _ in range(args.warmup_steps):
        run_step(args.mode, measure=False, annotate=False)
        torch.cuda.synchronize()

    results = {
        "forward": [],
        "backward": [],
        "optimizer": [],
    }

    print(f"Measuring {args.measurement_steps} steps...")

    # 正式开始测量
    with nvtx.range("my_measurement_region"):
        for _ in range(args.measurement_steps):
            timings = run_step(
                args.mode,
                measure=True,
                annotate=True,
            )

            for name, elapsed in timings.items():
                results[name].append(elapsed * 1000)

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
            }
        )

        df = pd.DataFrame(records)

        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / (
            f"{args.model_size}_{args.mode}_"
            f"warmup{args.warmup_steps}.csv"
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