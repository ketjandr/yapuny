import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _flash_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    n_heads,
    seq_len,
    scale,
    is_causal: tl.constexpr,
    BLOCK_M: tl.constexpr,  # block size for Q (rows)
    BLOCK_N: tl.constexpr,  # block size for K/V (cols)
    HEAD_DIM: tl.constexpr,  # d_k (must be power of 2)
):
    pid_m = tl.program_id(0)   # which Q block
    pid_bh = tl.program_id(1)  # which (batch, head), flattened B*H
    # decompose flattened (batch, head) index
    batch_idx = pid_bh // n_heads
    head_idx = pid_bh % n_heads

    # offsets for this Q block
    qm_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    dk_offsets = tl.arange(0, HEAD_DIM)

    # base offset for this (batch, head)
    q_base = q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_base = k_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_base = v_ptr + batch_idx * stride_vb + head_idx * stride_vh
    o_base = out_ptr + batch_idx * stride_ob + head_idx * stride_oh

    # load q tensor tile
    q_ptrs = q_base + qm_offsets[:, None] * stride_qm + dk_offsets[None, :] * stride_qk
    mask_q = qm_offsets[:, None] < seq_len
    q_tile = tl.load(q_ptrs, mask=mask_q, other=0.0) # (BLOCK_M, HEAD_DIM)

    # initialize accumulated softmax numerator and running max/sum
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    m = tl.full((BLOCK_M,), value=float("-inf"), dtype=tl.float32) # running row-wise max
    l = tl.zeros((BLOCK_M,), dtype=tl.float32) # running row-wise sum

    # tiled attention matrix computation
    for j in range(tl.cdiv(seq_len, BLOCK_N)):
        kn_offsets = j * BLOCK_N + tl.arange(0, BLOCK_N)

        # load k tensor tile
        k_ptrs = k_base + kn_offsets[:, None] * stride_kn + dk_offsets[None, :] * stride_kk
        mask_k = kn_offsets[:, None] < seq_len
        k_tile = tl.load(k_ptrs, mask=mask_k, other=0.0) # (BLOCK_N, HEAD_DIM)

        s_tile = q_tile @ tl.trans(k_tile) * scale

        # apply causal mask
        if is_causal:
            causal_mask = qm_offsets[:, None] >= kn_offsets[None, :]
            s_tile = tl.where(causal_mask, s_tile, float("-inf"))

        # load v tensor tile
        vn_offsets = kn_offsets
        mask_v = mask_k
        v_ptrs = v_base + vn_offsets[:, None] * stride_vn + dk_offsets[None, :] * stride_vk
        v_tile = tl.load(v_ptrs, mask=mask_v, other=0.0) # (BLOCK_N, HEAD_DIM)

        # online softmax update
        m_new = tl.maximum(m, tl.max(s_tile, axis=1)) # new running max
        correction = tl.exp(m - m_new)
        numerator = tl.exp(s_tile - m_new[:, None]) # (BLOCK_M, BLOCK_N)
        l_new = correction * l + tl.sum(numerator, axis=1) # new running sum
        acc = correction[:, None] * acc + numerator @ v_tile
        m = m_new
        l = l_new

    # normalize to a standard probability distribution
    out = acc / l[:, None]

    om_offsets = qm_offsets
    out_ptrs = o_base + om_offsets[:, None] * stride_om + dk_offsets[None, :] * stride_ok
    mask_out = mask_q

    # store back to HBM
    tl.store(out_ptrs, out, mask=mask_out)


def flash_attention(
    q: torch.Tensor,  # (B, H, T, D)
    k: torch.Tensor,  # (B, H, T, D)
    v: torch.Tensor,  # (B, H, T, D)
    is_causal: bool = True,
) -> torch.Tensor:
    """Flash Attention: memory-efficient exact attention."""
    B, H, T, D = q.shape
    scale = D ** -0.5

    out = torch.empty_like(q)

    # block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    assert D <= 128, "HEAD_DIM must be <= 128"
    HEAD_DIM = triton.next_power_of_2(D)

    # grid: one program per (q_block, batch*head)
    grid = (triton.cdiv(T, BLOCK_M), B * H)

    _flash_attention_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        n_heads=H,
        seq_len=T,
        scale=scale,
        is_causal=is_causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=HEAD_DIM,
    )

    return out


class FlashAttention(nn.Module):
    def __init__(self, is_causal: bool = True):
        super().__init__()
        self.is_causal = is_causal

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return flash_attention(q, k, v, self.is_causal)
