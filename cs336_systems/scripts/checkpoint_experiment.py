import torch
from cs336_basics.model import RotaryEmbedding, TransformerBlock

# 该模型的 num_layers 为 32
d_model, d_ff, num_heads, context_length = 2560, 10240, 16, 2048

block = TransformerBlock(
    d_model=d_model,
    d_ff=d_ff,
    num_heads=num_heads,
    positional_encoder=RotaryEmbedding(
        dim=d_model // num_heads,
        context_length=context_length,
    ),
).to("cuda:0")

# 尽可能利用 torch.compile 进行融合
block = torch.compile(block, fullgraph=True)

x = torch.randn(
    (4, context_length, d_model),
    requires_grad=True,
).to("cuda:0")

total_size_bytes = 0

def pack_hook(t):
    if isinstance(t, torch.nn.Parameter):
        # 跳过模型参数，避免重复统计
        return t

    global total_size_bytes
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    total_size_bytes += t.numel() * t.element_size()

    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

def unpack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

# with torch.autograd.graph.saved_tensors_hooks(
#     pack_hook,
#     unpack_hook,
# ):
    # y = block(x)
    # pass

# print(
#     "Total size of saved tensors in single TransformerBlock: "
#     f"{total_size_bytes / (1024**2):.2f} MiB"
# )


from torch.utils.checkpoint import checkpoint

def two_blocks(x):
    x = block(x)
    x = block(x)
    return x

def four_blocks_checkpoint(x):
    # checkpoint 会在 forward 中丢弃内部保存的张量。
    #
    # backward 执行到被 checkpoint 包裹的部分时，
    # 它会重新执行一次 forward，重新生成所需张量，
    # 然后继续正常的 backward。
    x = checkpoint(two_blocks, x, use_reentrant=False)
    x = checkpoint(two_blocks, x, use_reentrant=False)
    return x

with torch.autograd.graph.saved_tensors_hooks(
    pack_hook,
    unpack_hook,
):
    y = four_blocks_checkpoint(x)

print(
    "Total size of saved tensors in four TransformerBlocks "
    "with checkpointing: "
    f"{total_size_bytes / (1024**2):.2f} MiB"
)