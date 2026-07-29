import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import datetime
import itertools

def setup(rank, world_size, backend="nccl"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    torch.cuda.set_device(rank)  # 每个 rank 使用不同的 GPU
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def benchmark_all_reduce(rank, world_size, data_size_mb, num_warmup=5, num_iters=10):
    """
    在单个 rank 上执行 all-reduce 基准测试。
    返回该 rank 测量的平均延迟（毫秒）。
    """
    setup(rank, world_size, backend="nccl")
    
    # 计算张量元素数（float32 每个元素 4 字节）
    num_elements = (data_size_mb * 1024 * 1024) // 4
    tensor = torch.randn(num_elements, dtype=torch.float32, device=f"cuda:{rank}")
    
    # 预热
    for _ in range(num_warmup):
        dist.all_reduce(tensor, async_op=False)
        torch.cuda.synchronize()
    
    # 测量
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        dist.all_reduce(tensor, async_op=False)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    elapsed_ms = (end - start) * 1000.0 / num_iters
    
    cleanup()
    return elapsed_ms

def run_benchmark(world_size, data_size_mb):
    """启动 world_size 个进程，收集所有 rank 的测量结果并返回平均值和标准差。"""
    mp.spawn(
        fn=benchmark_all_reduce,
        args=(world_size, data_size_mb),
        nprocs=world_size,
        join=True
    )
    # 注意：上面的 spawn 会并行执行，但我们无法直接收集返回值。
    # 替代方案：使用队列或文件来收集结果。
    # 我们改为在每个进程中自行记录并最后汇总，但为了简便，我们让每个进程打印结果，主进程收集。
    # 更健壮的方式是使用 dist.all_gather_object，但在 spawn 中难以收集。
    # 这里我们采用一种简单方式：每个进程将结果写入一个共享列表（通过 multiprocessing.Manager）。
    # 下面重构代码使用 Manager。

def benchmark_worker(rank, world_size, data_size_mb, result_list, num_warmup=5, num_iters=10):
    """工作函数，将测量结果添加到 result_list 中。"""
    setup(rank, world_size, backend="nccl")
    num_elements = (data_size_mb * 1024 * 1024) // 4
    tensor = torch.randn(num_elements, dtype=torch.float32, device=f"cuda:{rank}")
    
    for _ in range(num_warmup):
        dist.all_reduce(tensor, async_op=False)
        torch.cuda.synchronize()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        dist.all_reduce(tensor, async_op=False)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    elapsed_ms = (end - start) * 1000.0 / num_iters
    result_list.append(elapsed_ms)
    cleanup()

def run_benchmark_managed(world_size, data_size_mb):
    """使用 Manager 收集结果，返回平均延迟和标准差（毫秒）。"""
    manager = mp.Manager()
    result_list = manager.list()
    mp.spawn(
        fn=benchmark_worker,
        args=(world_size, data_size_mb, result_list),
        nprocs=world_size,
        join=True
    )
    results = list(result_list)
    if len(results) == 0:
        return None, None
    avg = sum(results) / len(results)
    std = (sum((x - avg) ** 2 for x in results) / len(results)) ** 0.5
    return avg, std

def main():
    # 参数组合
    world_sizes = [2, 4, ]
    data_sizes_mb = [1, 10, 100, 1024]  # 1024 MB = 1 GB
    
    print("All-Reduce Benchmark (NCCL, GPU)")
    print("=" * 60)
    print(f"{'World Size':>10} | {'Data Size (MB)':>14} | {'Avg Latency (ms)':>18} | {'Std Dev (ms)':>12}")
    print("-" * 60)
    
    for world_size, data_size in itertools.product(world_sizes, data_sizes_mb):
        try:
            avg, std = run_benchmark_managed(world_size, data_size)
            if avg is not None:
                print(f"{world_size:>10} | {data_size:>14} | {avg:>18.3f} | {std:>12.3f}")
            else:
                print(f"{world_size:>10} | {data_size:>14} | {'ERROR':>18} | {'':>12}")
        except Exception as e:
            print(f"{world_size:>10} | {data_size:>14} | {str(e)[:18]:>18} | {'':>12}")

if __name__ == "__main__":
    main()