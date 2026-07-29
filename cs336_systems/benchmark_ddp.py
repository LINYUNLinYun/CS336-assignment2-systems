from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim

from ddp import NaiveDDP
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


def benchmark(rank: int, args: argparse.Namespace):
    world_size = args.world_size

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    config = MODEL_CONFIGS[args.model_size]

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        **config,
    ).to(device=device, dtype=dtype)

    model = NaiveDDP(model)

    if args.compile:
        model = torch.compile(model)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    x = torch.randint(
        args.vocab_size,
        (args.batch_size, args.context_length),
        device=device,
    )

    y = torch.randint(
        args.vocab_size,
        (args.batch_size, args.context_length),
        device=device,
    )

    step_times: list[float] = []
    comm_times: list[float] = []

    for it in range(args.warmup_steps + args.measurement_steps):

        optimizer.zero_grad(set_to_none=True)

        step_start = torch.cuda.Event(enable_timing=True)
        step_end = torch.cuda.Event(enable_timing=True)
        comm_start = torch.cuda.Event(enable_timing=True)
        comm_end = torch.cuda.Event(enable_timing=True)

        step_start.record()

        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, args.vocab_size),
            y.view(-1),
        )
        loss.backward()

        # torch.cuda.synchronize()
        comm_start.record()
        if args.flat:
            model.synchronize_gradients_flat
        else:
            model.synchronize_gradients()
        comm_end.record()

        optimizer.step()

        step_end.record()
        torch.cuda.synchronize()

        if it >= args.warmup_steps:
            step_times.append(step_start.elapsed_time(step_end))
            comm_times.append(comm_start.elapsed_time(comm_end))

    if rank == 0:
        avg_step = sum(step_times) / len(step_times)
        avg_comm = sum(comm_times) / len(comm_times)

        print("=" * 60)
        print(f"Average step time : {avg_step:.3f} ms")
        print(f"Average comm time : {avg_comm:.3f} ms")
        print(f"Communication ratio : {100 * avg_comm / avg_step:.2f}%")
        print("=" * 60)

        df = pd.DataFrame({
            "model": [args.model_size],
            "batch_size": [args.batch_size],
            "context_length": [args.context_length],
            "vocab_size": [args.vocab_size],
            "dtype": [args.dtype],
            "compile": [args.compile],
            "step_ms": [avg_step],
            "comm_ms": [avg_comm],
            "comm_ratio": [avg_comm / avg_step],
        })

        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / f"{args.model_size}_ddp.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

    dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the CS336 DDP")

    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--measurement-steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ddp"),
    )
    parser.add_argument(
        "--model-size",
        choices=("xl", "medium", "large", "small"),
        default="xl",
    )
    parser.add_argument(
        "--dtype",
        choices=("fp32", "bf16"),
        default="bf16",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
    )
    parser.add_argument(
            "--flat",
            action="store_true",
        )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    world_size = args.world_size

    mp.spawn(
        benchmark,
        args=(args,),
        nprocs=world_size,
        join=True,
    )
