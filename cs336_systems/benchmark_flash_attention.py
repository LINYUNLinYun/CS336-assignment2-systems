import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import triton.testing

from cs336_systems.flash_attention import FlashAttention

import os
os.makedirs("results", exist_ok=True)

device = "cuda"
batch_size = 1

# 调试先用这些
seq_lens = [128, 256, 512, 1024, 2048, 4096]
# seq_lens = [128, 256, 512, 1024, 2048, 4096]
# 最终交作业
seq_lens = [2**i for i in range(7, 17)]

dims = [16, 32, 64, 128]
dtypes = [torch.bfloat16, torch.float32]

results = []

for dtype in dtypes:
    for d in dims:

        if d <= 16:
            Q_TILE = 128
            K_TILE = 128
        elif d <= 32:
            Q_TILE = 128
            K_TILE = 64
        elif d <= 64:
            Q_TILE = 64
            K_TILE = 64
        else:
            Q_TILE = 32
            K_TILE = 32

        for N in seq_lens:
            print(f"N={N}, D={d}, dtype={dtype}")

            try:
                Q = torch.randn(batch_size, N, d, device=device, dtype=dtype, requires_grad=True)
                K = torch.randn_like(Q, requires_grad=True)
                V = torch.randn_like(Q, requires_grad=True)
                grad = torch.randn_like(Q)

                ####################################################
                # Warmup
                ####################################################

                _ = FlashAttention.apply(Q, K, V, True, Q_TILE, K_TILE)
                _ = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
                torch.cuda.synchronize()

                ####################################################
                # Forward
                ####################################################

                triton_forward = triton.testing.do_bench(
                    lambda: FlashAttention.apply(Q, K, V, True, Q_TILE, K_TILE),
                    warmup=10, rep=50,
                )

                pytorch_forward = triton.testing.do_bench(
                    lambda: F.scaled_dot_product_attention(Q, K, V, is_causal=True),
                    warmup=10, rep=50,
                )

                ####################################################
                # Backward
                ####################################################

                out = FlashAttention.apply(Q, K, V, True, Q_TILE, K_TILE)
                def triton_backward():
                    out.backward(grad, retain_graph=True)
                    Q.grad = None
                    K.grad = None
                    V.grad = None

                out_ref = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
                def pytorch_backward():
                    out_ref.backward(grad, retain_graph=True)
                    Q.grad = None
                    K.grad = None
                    V.grad = None

                triton_backward_time = triton.testing.do_bench(triton_backward, warmup=10, rep=50)
                pytorch_backward_time = triton.testing.do_bench(pytorch_backward, warmup=10, rep=50)

                ####################################################
                # Total
                ####################################################

                def triton_total():
                    out = FlashAttention.apply(Q, K, V, True, Q_TILE, K_TILE)
                    out.backward(grad, retain_graph=True)
                    Q.grad = None
                    K.grad = None
                    V.grad = None

                def pytorch_total():
                    out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
                    out.backward(grad, retain_graph=True)
                    Q.grad = None
                    K.grad = None
                    V.grad = None

                triton_total_time = triton.testing.do_bench(triton_total, warmup=10, rep=50)
                pytorch_total_time = triton.testing.do_bench(pytorch_total, warmup=10, rep=50)

                results.append({
                    "dtype": str(dtype).split(".")[-1],
                    "N": N,
                    "D": d,
                    "PyTorch Fwd(ms)": pytorch_forward,
                    "Triton Fwd(ms)": triton_forward,
                    "PyTorch Bwd(ms)": pytorch_backward_time,
                    "Triton Bwd(ms)": triton_backward_time,
                    "PyTorch Total(ms)": pytorch_total_time,
                    "Triton Total(ms)": triton_total_time,
                })

            except torch.cuda.OutOfMemoryError:
                print("OOM")
                torch.cuda.empty_cache()

####################################################
# Output
####################################################

df = pd.DataFrame(results)
print(df)
print(df.to_markdown())
df.to_csv("results/flash_attention_benchmark.csv", index=False)

####################################################
# 绘图 — 一次性显示所有子图
####################################################

df["dtype"] = df["dtype"].astype(str)
dtype_list = sorted(df["dtype"].unique())  # ["bfloat16", "float32"]
metrics = ["Fwd", "Bwd", "Total"]

colors = {16: "#1f77b4", 32: "#ff7f0e", 64: "#2ca02c", 128: "#d62728"}

n_rows = len(dtype_list) * 2          # 每个 dtype 两行：绝对时间 + 加速比
n_cols = len(metrics)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(5 * n_cols, 4 * n_rows),
    sharex="col",
)
# 保证 axes 是二维的
if n_rows == 1:
    axes = axes.reshape(1, -1)

for dtype_idx, dtype_str in enumerate(dtype_list):
    subset = df[df["dtype"] == dtype_str]

    for col, metric in enumerate(metrics):
        # ---- 绝对时间 ----
        ax_time = axes[dtype_idx * 2, col]
        # ---- 加速比 ----
        ax_speedup = axes[dtype_idx * 2 + 1, col]

        for d in dims:
            data = subset[subset["D"] == d]
            c = colors[d]

            pytorch_col = f"PyTorch {metric}(ms)"
            triton_col = f"Triton {metric}(ms)"

            # 绝对时间
            ax_time.plot(data["N"], data[pytorch_col], "o-",
                         color=c, markersize=4, label=f"PyTorch D={d}")
            ax_time.plot(data["N"], data[triton_col], "s--",
                         color=c, markersize=4, label=f"Triton  D={d}")

            # 加速比
            speedup = data[pytorch_col] / data[triton_col]
            ax_speedup.plot(data["N"], speedup, "D-",
                            color=c, markersize=4, label=f"D={d}")

        # 格式化时间轴
        ax_time.set_xscale("log", base=2)
        ax_time.set_yscale("log")
        ax_time.set_ylabel(f"{metric} (ms)")
        ax_time.grid(True, alpha=0.3)
        ax_time.legend(fontsize=6, ncol=2)

        # 格式化加速比轴
        ax_speedup.axhline(y=1.0, color="gray", linestyle=":", linewidth=1)
        ax_speedup.set_xscale("log", base=2)
        ax_speedup.set_ylabel("Speedup ×")
        ax_speedup.set_xlabel("Sequence length N")
        ax_speedup.grid(True, alpha=0.3)
        ax_speedup.legend(fontsize=6, ncol=2)

        # 列标题（仅第一行）
        ax_time.set_title(f"{dtype_str} — {metric}", fontsize=11)

fig.tight_layout()
fig.savefig("./results/benchmark_comparison.png", dpi=150, bbox_inches="tight")
print("Figure saved to results/benchmark_comparison.png")
print("Done.")
