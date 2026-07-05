import argparse
import torch
import timeit
import statistics
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
from contextlib import nullcontext
import torch.cuda.nvtx as nvtx

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

def run_step(mode: str, measure: bool = False):
    timings = {}

    optimizer.zero_grad(set_to_none=True)

    # Forward
    if measure:
        torch.cuda.synchronize()
        start = timeit.default_timer()

    logits = model(inputs)

    if measure:
        torch.cuda.synchronize()
        timings["forward"] = timeit.default_timer() - start

    if mode == "forward":
        return timings

    loss = F.cross_entropy(
        logits.reshape(-1, args.vocab_size),
        targets.reshape(-1),
    )

    # Backward
    if measure:
        torch.cuda.synchronize()
        start = timeit.default_timer()

    loss.backward()

    if measure:
        torch.cuda.synchronize()
        timings["backward"] = timeit.default_timer() - start

    if mode == "forward_backward":
        return timings

    # Optimizer
    if measure:
        torch.cuda.synchronize()
        start = timeit.default_timer()

    optimizer.step()

    if measure:
        torch.cuda.synchronize()
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
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model_size]

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

    # 空跑几轮
    for _ in range(args.warmup_steps):
        run_step(args.mode, measure=False)
        torch.cuda.synchronize()

    results = {
        "forward": [],
        "backward": [],
        "optimizer": [],
    }

    print(f"Measuring {args.measurement_steps} steps...")

    for _ in range(args.measurement_steps):
        timings = run_step(args.mode, measure=True)

        for name, elapsed in timings.items():
            results[name].append(elapsed * 1000)  # 秒转毫秒

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