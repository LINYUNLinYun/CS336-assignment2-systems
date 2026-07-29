import time
import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim

from cs336_systems.ddp import NaiveDDP
from cs336_basics.model import BasicsTransformerLM   # 改成你的模型


############################################################
# Configuration
############################################################


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
config = MODEL_CONFIGS["xl"]



BATCH_SIZE = 8
CONTEXT_LENGTH = 512

VOCAB_SIZE = 10000

NUM_WARMUP = 10
NUM_ITERS = 50

WORLD_SIZE =2


def benchmark(rank, world_size):


    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        **config
        
    ).to(device)

    model = NaiveDDP(model)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )


    x = torch.randint(
        VOCAB_SIZE,
        (BATCH_SIZE, CONTEXT_LENGTH),
        device=device,
    )

    y = torch.randint(
        VOCAB_SIZE,
        (BATCH_SIZE, CONTEXT_LENGTH),
        device=device,
    )

    step_times = []
    comm_times = []

    for it in range(NUM_WARMUP + NUM_ITERS):

        optimizer.zero_grad(set_to_none=True)

        ####################################################
        # Total step timer
        ####################################################

        step_start = torch.cuda.Event(enable_timing=True)
        step_end = torch.cuda.Event(enable_timing=True)

        ####################################################
        # Communication timer
        ####################################################

        comm_start = torch.cuda.Event(enable_timing=True)
        comm_end = torch.cuda.Event(enable_timing=True)

        step_start.record()

        logits = model(x)

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, VOCAB_SIZE),
            y.view(-1),
        )

        loss.backward()

        ####################################################
        # Communication
        ####################################################

        comm_start.record()

        model.synchronize_gradients()

        comm_end.record()

        optimizer.step()

        step_end.record()

        torch.cuda.synchronize()

        if it >= NUM_WARMUP:

            step_times.append(
                step_start.elapsed_time(step_end)
            )

            comm_times.append(
                comm_start.elapsed_time(comm_end)
            )

    if rank == 0:

        avg_step = sum(step_times) / len(step_times)

        avg_comm = sum(comm_times) / len(comm_times)

        print("=" * 60)
        print(f"Average step time : {avg_step:.3f} ms")
        print(f"Average comm time : {avg_comm:.3f} ms")
        print(f"Communication ratio : {100 * avg_comm / avg_step:.2f}%")
        print("=" * 60)

    dist.destroy_process_group()

if __name__ == "__main__":

    mp.spawn(
        benchmark,
        args=(WORLD_SIZE,),
        nprocs=WORLD_SIZE,
        join=True,
    )