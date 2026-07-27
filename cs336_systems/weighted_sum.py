import torch
import triton
import triton.language as tl

from einops import rearrange


def weighted_sum(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    PyTorch reference implementation.

    x:      [..., D]
    weight: [D]

    output: [...]
    """
    return (weight * x).sum(dim=-1)


@triton.jit
def weighted_sum_fwd(
    x_ptr,                          # Input pointers
    weight_ptr,                     
    output_ptr,                     # Output pointer

    x_stride_row,
    x_stride_dim,                   # Strides of x

    weight_stride_dim,              # Usually 1
    output_stride_row,              # Usually 1

    NUM_ROWS,                       # 总行数
    D,                              # 每行的特征维度

    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    """
    Each Triton program instance computes the weighted sum
    for one tile of rows.
    """

    row_tile_idx = tl.program_id(0)             # 告诉我们当前运行的是哪个线程块。

    # x is viewed as [NUM_ROWS, D].             建立一个抽象块指针
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(NUM_ROWS, D),
        strides=(x_stride_row, x_stride_dim),
        offsets=(
            row_tile_idx * ROWS_TILE_SIZE,
            0,
        ),
        block_shape=(
            ROWS_TILE_SIZE,
            D_TILE_SIZE,
        ),
        order=(1, 0),                           # 表示第 1 维是更连续、更内层的内存维度。
    )

    # weight has shape [D].
    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    # output has shape [NUM_ROWS].
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(NUM_ROWS,),
        strides=(output_stride_row,),
        offsets=(
            row_tile_idx * ROWS_TILE_SIZE,
        ),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    # One accumulated scalar for each row.
    output = tl.zeros(
        (ROWS_TILE_SIZE,),
        dtype=tl.float32,
    )

    # Iterate over tiles along the D dimension. 向上整除，遍历D维上的tile
    for _ in range(tl.cdiv(D, D_TILE_SIZE)):
        row = tl.load(
            x_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )

        weight = tl.load(
            weight_block_ptr,
            boundary_check=(0,),
            padding_option="zero",
        )

        # row:              [ROWS_TILE_SIZE, D_TILE_SIZE]
        # weight[None, :]:  [1, D_TILE_SIZE] 沿着第1维做sum
        output += tl.sum(
            row * weight[None, :],
            axis=1,
        )

        # Advance to the next tile along D.
        x_block_ptr = x_block_ptr.advance(
            (0, D_TILE_SIZE)
        )

        weight_block_ptr = weight_block_ptr.advance(
            (D_TILE_SIZE,)
        )

    tl.store(
        output_block_ptr,
        output,
        boundary_check=(0,),
    )


@triton.jit
def weighted_sum_backward(
    x_ptr,
    weight_ptr,                     # Forward inputs

    grad_output_ptr,                # Gradient with respect to output

    grad_x_ptr,
    partial_grad_weight_ptr,        # Gradient outputs

    stride_xr,
    stride_xd,

    stride_wd,

    stride_gr,

    stride_gxr,
    stride_gxd,

    stride_gwb,
    stride_gwd,

    NUM_ROWS,
    D,

    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    """
    Computes:

        grad_x[i, j] =
            grad_output[i] * weight[j]

        grad_weight[j] =
            sum_i x[i, j] * grad_output[i]

    Each program computes grad_x directly and writes one partial
    contribution to grad_weight.
    """

    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)

    grad_output_block_ptr = tl.make_block_ptr(
        base=grad_output_ptr,
        shape=(NUM_ROWS,),
        strides=(stride_gr,),
        offsets=(
            row_tile_idx * ROWS_TILE_SIZE,
        ),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_xr, stride_xd),
        offsets=(
            row_tile_idx * ROWS_TILE_SIZE,
            0,
        ),
        block_shape=(
            ROWS_TILE_SIZE,
            D_TILE_SIZE,
        ),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(D,),
        strides=(stride_wd,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    grad_x_block_ptr = tl.make_block_ptr(
        base=grad_x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_gxr, stride_gxd),
        offsets=(
            row_tile_idx * ROWS_TILE_SIZE,
            0,
        ),
        block_shape=(
            ROWS_TILE_SIZE,
            D_TILE_SIZE,
        ),
        order=(1, 0),
    )

    # Shape:
    # [number_of_row_tiles, D]
    partial_grad_weight_block_ptr = tl.make_block_ptr(
        base=partial_grad_weight_ptr,
        shape=(n_row_tiles, D),
        strides=(stride_gwb, stride_gwd),
        offsets=(
            row_tile_idx,
            0,
        ),
        block_shape=(
            1,
            D_TILE_SIZE,
        ),
        order=(1, 0),
    )

    grad_output = tl.load(
        grad_output_block_ptr,
        boundary_check=(0,),
        padding_option="zero",
    )

    for _ in range(tl.cdiv(D, D_TILE_SIZE)):
        # ---------------- grad_x ----------------

        weight = tl.load(
            weight_block_ptr,
            boundary_check=(0,),
            padding_option="zero",
        )

        # Outer product: 外积，计算xij的梯度
        #
        # grad_output[:, None]:
        #     [ROWS_TILE_SIZE, 1]
        #
        # weight[None, :]:
        #     [1, D_TILE_SIZE]
        #
        # result:
        #     [ROWS_TILE_SIZE, D_TILE_SIZE]
        grad_x_row = (
            grad_output[:, None]
            * weight[None, :]
        )
        # 直接存储梯度到grad_x中
        tl.store(
            grad_x_block_ptr,
            grad_x_row,
            boundary_check=(0, 1),
        )

        # ---------------- partial grad_weight ----------------

        row = tl.load(
            x_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )

        # Reduce across the rows handled by this program.
        grad_weight_row = tl.sum(
            row * grad_output[:, None],
            axis=0,
            keep_dims=True,
        )

        tl.store(
            partial_grad_weight_block_ptr,
            grad_weight_row,
            boundary_check=(1,),
        )

        # Advance along D.
        x_block_ptr = x_block_ptr.advance(
            (0, D_TILE_SIZE)
        )

        weight_block_ptr = weight_block_ptr.advance(
            (D_TILE_SIZE,)
        )

        partial_grad_weight_block_ptr = (
            partial_grad_weight_block_ptr.advance(
                (0, D_TILE_SIZE)
            )
        )

        grad_x_block_ptr = grad_x_block_ptr.advance(
            (0, D_TILE_SIZE)
        )


class WeightedSumFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        x:      [..., D]
        weight: [D]
        output: [...]
        """

        D = x.shape[-1]
        output_dims = x.shape[:-1]
        input_shape = x.shape

        assert weight.ndim == 1         # 必须一维
        assert weight.shape[0] == D
        assert x.is_cuda and weight.is_cuda     #  "Expected CUDA tensors"
        assert x.is_contiguous()                # 运算假设内存一定要连续

        # Flatten every dimension except D. 方便处理
        x_2d = rearrange(
            x,
            "... d -> (...) d",
        )
        # 为backward保存输入张量和权重张量，后续取用
        ctx.save_for_backward(
            x_2d,
            weight,
        )

        # Roughly 16 iterations along D. 这里取D的tile大小为D的2次幂除以16，保证每个tile大约有16次迭代
        # 至于为啥是取2的幂，可能是为了内存规整啥的；而为啥是16就不知道了。
        ctx.D_TILE_SIZE = (
            max(1, triton.next_power_of_2(D) // 16)
        )

        ctx.ROWS_TILE_SIZE = 16         # 为啥16
        ctx.input_shape = input_shape

        y = torch.empty(
            output_dims,
            device=x.device,
            dtype=x.dtype,
        )

        n_rows = y.numel()
        # launch grid决定启动多少个program instances。
        grid = (
            triton.cdiv(
                n_rows,
                ctx.ROWS_TILE_SIZE,
            ),
        )

        weighted_sum_fwd[grid](
            x_2d,
            weight,
            y,

            x_2d.stride(0),
            x_2d.stride(1),

            weight.stride(0),
            y.stride(0),

            NUM_ROWS=n_rows,
            D=D,

            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE,
            D_TILE_SIZE=ctx.D_TILE_SIZE,
        )

        return y.view(input_shape[:-1])     # 恢复 形状（从2d）

    @staticmethod
    def backward(
        ctx,
        grad_out: torch.Tensor,
    ):
        x, weight = ctx.saved_tensors

        ROWS_TILE_SIZE = ctx.ROWS_TILE_SIZE
        D_TILE_SIZE = ctx.D_TILE_SIZE
        input_shape = ctx.input_shape

        n_rows, D = x.shape

        # Each Triton program writes one partial grad_weight row.
        partial_grad_weight = torch.empty(
            (
                triton.cdiv(
                    n_rows,
                    ROWS_TILE_SIZE,
                ),
                D,
            ),
            device=x.device,
            dtype=x.dtype,
        )

        grad_x = torch.empty_like(x)

        grad_out_flat = grad_out.contiguous().view(-1)

        grid = (
            triton.cdiv(
                n_rows,
                ROWS_TILE_SIZE,
            ),
        )

        weighted_sum_backward[grid](
            x,
            weight,
            grad_out_flat,

            grad_x,
            partial_grad_weight,

            x.stride(0),
            x.stride(1),

            weight.stride(0),

            grad_out_flat.stride(0),

            grad_x.stride(0),
            grad_x.stride(1),

            partial_grad_weight.stride(0),
            partial_grad_weight.stride(1),

            NUM_ROWS=n_rows,
            D=D,

            ROWS_TILE_SIZE=ROWS_TILE_SIZE,
            D_TILE_SIZE=D_TILE_SIZE,
        )

        # Sum partial results from every row tile.
        grad_weight = partial_grad_weight.sum(dim=0)

        grad_x = grad_x.view(input_shape)

        return grad_x, grad_weight


f_weighted_sum = WeightedSumFunc.apply

if __name__ == "__main__":
    torch.manual_seed(0)

    device = "cuda:0"

    # -----------------------
    # 创建输入
    # -----------------------
    batch = 128
    dim = 256

    x = torch.randn(
        batch,
        dim,
        device=device,
        requires_grad=True,
    )

    weight = torch.randn(
        dim,
        device=device,
        requires_grad=True,
    )


    # -----------------------
    # Triton forward
    # -----------------------
    y_triton = f_weighted_sum(
        x,
        weight,
    )


    # -----------------------
    # PyTorch forward
    # -----------------------
    y_torch = weighted_sum(
        x,
        weight,
    )


    print("Forward check:")
    print(
        torch.allclose(
            y_triton,
            y_torch,
            atol=1e-5,
            rtol=1e-5,
        )
    )

    print(
        "max forward error:",
        (y_triton - y_torch).abs().max().item()
    )


    # -----------------------
    # backward
    # -----------------------

    grad = torch.randn_like(y_triton)


    y_triton.backward(
        grad,
        retain_graph=True,
    )

    grad_x_triton = x.grad.clone()
    grad_w_triton = weight.grad.clone()


    # 清空梯度
    x.grad = None
    weight.grad = None


    y_torch.backward(
        grad,
    )

    grad_x_torch = x.grad.clone()
    grad_w_torch = weight.grad.clone()


    print("\nBackward check:")

    print(
        "grad x:",
        torch.allclose(
            grad_x_triton,
            grad_x_torch,
            atol=1e-5,
            rtol=1e-5,
        )
    )

    print(
        "max grad x error:",
        (grad_x_triton-grad_x_torch)
        .abs()
        .max()
        .item()
    )


    print(
        "grad weight:",
        torch.allclose(
            grad_w_triton,
            grad_w_torch,
            atol=1e-5,
            rtol=1e-5,
        )
    )

    print(
        "max grad weight error:",
        (grad_w_triton-grad_w_torch)
        .abs()
        .max()
        .item()
    )

    print(grad_x_triton,grad_w_triton)
    print(y_triton)