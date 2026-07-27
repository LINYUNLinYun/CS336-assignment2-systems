from __future__ import annotations

import math
import triton
import triton.language as tl
import torch


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, is_causal=False):
        """
        Q: [batch_size, seq_len, d]
        K: [batch_size, seq_len, d]
        V: [batch_size, seq_len, d]
        """
        assert Q.ndim == 3 and K.ndim == 3 and V.ndim == 3
        assert Q.shape == K.shape and K.shape == V.shape
        assert Q.shape[2] >= 16
        assert Q.shape[1] >=16

        batch_size, seq_len, d = Q.shape
        scale = 1.0 / math.sqrt(d)

        # 先固定 tile size，跑通后再调整
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16

        O = torch.empty_like(Q)

        L = torch.empty(
            (batch_size, seq_len),
            device=Q.device,
            dtype=torch.float32,
        )

        # TODO:
        # 1. 遍历 batch
        # 2. 遍历 Q tile
        # 3. 初始化当前 Q tile 的 m、l、O_tile
        # 4. 遍历 K/V tile
        # 5. 按 Algorithm 1 更新 m、l、O_tile
        # 6. 最后归一化 O_tile，并计算 L
        for batch_idx in range(batch_size):
            # 遍历每个batch
            for q_start in range(0, seq_len, Q_TILE_SIZE):
                # 步长 B_q
                q_end = q_start + Q_TILE_SIZE
                q_tile = Q[batch_idx,q_start:q_end]

                # 未归一化输出
                O_tile = torch.zeros((Q_TILE_SIZE,d),device=Q.device,dtype=torch.float32)

                # running max
                m = torch.full((Q_TILE_SIZE,),fill_value=float("-inf"),device=Q.device,dtype=torch.float32)
                # 分母
                l = torch.zeros((Q_TILE_SIZE,),device=Q.device,dtype=torch.float32)

                for k_start in range(0, seq_len, K_TILE_SIZE):
                    k_end = k_start + K_TILE_SIZE
                    k_tile = K[batch_idx, k_start:k_end]
                    v_tile = V[batch_idx, k_start:k_end]
                    # Q_TILE_SIZE,K_TILE_SIZE
                    S_tile = q_tile@k_tile.transpose(-2,-1)*scale # 注意力分数

                    if is_causal:
                        # 因果 
                        q_idx = torch.arange(
                            q_start,
                            q_end,
                            device=Q.device
                        )[:, None]

                        k_idx = torch.arange(
                            k_start,
                            k_end,
                            device=Q.device
                        )[None, :]

                        mask = q_idx < k_idx

                        S_tile = S_tile.masked_fill(
                            mask,
                            float("-inf")
                        )

                    tile_max = S_tile.max(dim=-1).values       # 各行最大值

                    m_new = torch.maximum(m, tile_max)      # 更新全局最大值
                    correction = torch.exp(m - m_new)       # 修正

                    P_tile = torch.exp(S_tile - m_new[:,None ])         #还记得吗？减去最大值是为了数值稳定不然exp会爆
                    l_new = l*correction + P_tile.sum(dim=-1)
                    O_new = O_tile*correction[:,None] + P_tile@v_tile       # 这里O也可以用同样的手段进行修正,o = p v

                    # 更新
                    m = m_new
                    l = l_new
                    O_tile = O_new

                # 所有行遍历完才做归一化
                O_tile = O_tile / l[:,None]

                # L_i = m_i + log(l_i)
                L_tile = m + torch.log(l)               # 这里加m是为了还原原来的分母而不是减去最大值的，用log存是为了数值稳定

                O[batch_idx,q_start:q_end,:] = O_tile
                L[batch_idx,q_start:q_end] = L_tile
        # 所有batch计算完
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal

        grad_Q,grad_K,grad_V = _compiled_attention_backward(Q, K, V, O, L,grad_output,  is_causal)
        
        return grad_Q,grad_K,grad_V,None,None,None


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal:tl.constexpr,
):
    # 程序索引
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # 使用对应的 batch 索引乘以每个张量的 batch stride，
    # 对各个指针进行偏移
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),                     # 获取全部元素
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

    V_block_ptr = tl.make_block_ptr(
                V_ptr + batch_index * stride_vb,
                shape=(N_KEYS, D),
                strides=(stride_vk, stride_vd),
                offsets=(0, 0),
                block_shape=(K_TILE_SIZE, D),
                order=(1, 0),
            )

    O_block_ptr = tl.make_block_ptr(
                    O_ptr + batch_index * stride_ob,
                    shape=(N_QUERIES, D),
                    strides=(stride_oq, stride_od),
                    offsets=(query_tile_index * Q_TILE_SIZE, 0),
                    block_shape=(Q_TILE_SIZE, D),
                    order=(1, 0),
                )
    L_block_ptr = tl.make_block_ptr(
                        L_ptr + batch_index * stride_lb,
                        shape=(N_QUERIES,),
                        strides=(stride_lq,),
                        offsets=(query_tile_index * Q_TILE_SIZE,),
                        block_shape=(Q_TILE_SIZE,),
                        order=(0,),
                    )

    # 读取当前 program 负责的 Q tile shape: [Q_TILE_SIZE, D]
    Q_tile = tl.load(Q_block_ptr)
    # 当前 query tile 的未归一化输出累计值 O_acc
    O_acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    m = tl.full((Q_TILE_SIZE,),float("-inf"), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,),dtype = tl.float32)

    # 唯一循环体 对于每个程序实例来说 要遍历所有的tile in D dimension
    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_tile = tl.load(K_block_ptr)
        V_tile = tl.load(V_block_ptr)

        S_tile = tl.dot(Q_tile, tl.trans(K_tile))*scale

        if is_causal:
            # 因果 
            q_offsets = (
                query_tile_index * Q_TILE_SIZE
                + tl.arange(0, Q_TILE_SIZE)
            )

            k_offsets = (
                i * K_TILE_SIZE
                + tl.arange(0, K_TILE_SIZE)
            )

            causal_mask = q_offsets[:, None] < k_offsets[None, :]

            S_tile += tl.where(causal_mask,-1e6,0.0)

        tile_max = tl.max(S_tile,axis = 1)

        m_new = tl.maximum(m, tile_max)
        correction = tl.exp(m - m_new)

        P_tile = tl.exp(S_tile - m_new[:, None])
        l_new = l*correction + P_tile.sum(axis = -1)

        O_acc = tl.dot(P_tile,V_tile, acc=O_acc*correction[:,None]) 

        # 更新全局信息
        m = m_new
        l = l_new

        # 更新KV指针
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_acc = O_acc / l[:, None]
    L_tile = m + tl.log(l)

    tl.store(O_block_ptr, O_acc.to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, L_tile.to(O_block_ptr.type.element_ty))

    
def _attention_backward( Q, K, V, O, L,grad_outputs,is_causal):
        B, N_q, h_dim = Q.shape
        _, N_k, h_dim = Q.shape
        scale = 1.0 / math.sqrt(h_dim)
        S = Q @ K.transpose(-2,-1) *scale
        if is_causal:
            # 这里不能用上三角矩阵构造因为不是方阵
            q_idx = torch.arange(N_q, device=Q.device)[:, None]   # (N_q, 1)
            k_idx = torch.arange(N_k, device=Q.device)[None, :]   # (1, N_k)
            mask = q_idx < k_idx                                    # (N_q, N_k)
            S = S + mask * (-1e6)
            S += mask*-1e6

        P = torch.exp(S - L.unsqueeze(-1))
        grad_V = P.transpose(-2,-1) @ grad_outputs
        grad_P = grad_outputs @ V.transpose(-2,-1)

        D = (O* grad_outputs).sum(dim=-1)       # 逐元素相乘
        grad_S = P*(grad_P - D.unsqueeze(-1))
        grad_Q = grad_S @ K *scale
        grad_K = grad_S.transpose(-2,-1) @ Q *scale

        return grad_Q,grad_K,grad_V

_compiled_attention_backward = torch.compile(_attention_backward)

class FlashAttention(torch.autograd.Function):
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                is_causal=False,Q_TILE_SIZE=16,K_TILE_SIZE=16,):
        B, N_q, D = Q.shape
        _, N_k, D = K.shape

        O = torch.empty_like(Q)
        L = torch.empty((B,N_q),device=Q.device, dtype=torch.float32)

        grid = ((N_q + Q_TILE_SIZE-1)//Q_TILE_SIZE,B)

        flash_fwd_kernel[grid](
            Q,K,V,O,L,
            Q.stride(0),Q.stride(1),Q.stride(2),
            K.stride(0),K.stride(1),K.stride(2),
            V.stride(0),V.stride(1),V.stride(2),
            O.stride(0),O.stride(1),O.stride(2),
            L.stride(0),L.stride(1),
            N_q,N_k,tl.constexpr(1.0 / math.sqrt(D)),D,Q_TILE_SIZE,K_TILE_SIZE,is_causal)

        # 所有batch计算完
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal

        return O

    
    @staticmethod
    def backward(ctx, grad_outputs):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal

        grad_Q,grad_K,grad_V = _compiled_attention_backward(Q, K, V, O, L,grad_outputs,  is_causal)
        
        return grad_Q,grad_K,grad_V,None,None,None

        


def reference_attention(Q, K, V, is_causal=False):
    d = Q.shape[-1]

    scores = Q @ K.transpose(-1, -2)
    scores = scores / math.sqrt(d)

    if is_causal:
        n_queries = Q.shape[-2]
        n_keys = K.shape[-2]

        query_indices = torch.arange(
            n_queries,
            device=Q.device,
        )
        key_indices = torch.arange(
            n_keys,
            device=Q.device,
        )

        mask = (
            key_indices[None, :]
            > query_indices[:, None]
        )

        scores = scores.masked_fill(
            mask[None, :, :],
            float("-inf"),
        )

    probabilities = torch.softmax(scores, dim=-1)
    return probabilities @ V

if __name__ == '__main__':
    device = "cuda"

    Q = torch.randn(
        2,
        64,
        32,
        device=device,
        dtype=torch.float32,
    )

    K = torch.randn(
        2,
        64,
        32,
        device=device,
        dtype=torch.float32,
    )

    V = torch.randn(
        2,
        64,
        32,
        device=device,
        dtype=torch.float32,
    )

    expected = reference_attention(
        Q,
        K,
        V,
        is_causal=False,
    )

    actual = FlashAttentionPytorch.apply(
        Q,
        K,
        V,
        False,
    )
    print("actual:", actual - expected)
    print("expected:", expected)

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-4,
    )

    print("forward passed")