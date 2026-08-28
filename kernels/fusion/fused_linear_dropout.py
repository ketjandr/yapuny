import random

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_linear_dropout_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    seed,
    M,
    N,
    K,
    p_drop,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    is_training: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # linear layer: tiled reduction over K
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        x_ptrs = x_ptr + rm_offsets[:, None] * stride_xm + k_offsets[None, :] * stride_xk
        mask_x = (rm_offsets[:, None] < M) & (k_offsets[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_ptr + rn_offsets[:, None] * stride_wn + k_offsets[None, :] * stride_wk
        mask_w = (rn_offsets[:, None] < N) & (k_offsets[None, :] < K)
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc += tl.dot(x_tile, tl.trans(w_tile))

    # add bias
    mask_bias = rn_offsets < N
    bias = tl.load(bias_ptr + rn_offsets, mask=mask_bias, other=0.0)
    acc += bias[None, :]

    if is_training:
        # generate a 2d tensor (from 0, 1, ..., N*M - 1)
        dropout_offsets = rm_offsets[:, None] * N + rn_offsets[None, :]
        dropout_mask = tl.rand(seed, dropout_offsets)
        acc = tl.where(dropout_mask >= p_drop, acc / (1 - p_drop), 0.0)

    # store
    out_ptrs = out_ptr + rm_offsets[:, None] * stride_om + rn_offsets[None, :] * stride_on
    mask_out = (rm_offsets[:, None] < M) & (rn_offsets[None, :] < N)
    tl.store(out_ptrs, acc, mask=mask_out)


def fused_linear_dropout(
    x: torch.Tensor,  # (B, T, C)
    weight: torch.Tensor,  # (N, K)
    bias: torch.Tensor,  # (N,)
    p_drop: float = 0.1,
    training: bool = True,
) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    M, K = x_2d.shape
    N = weight.shape[0]

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    seed = random.randint(0, 2**31)

    _fused_linear_dropout_kernel[grid](
        x_2d,
        weight,
        bias,
        out,
        seed,
        M,
        N,
        K,
        p_drop,
        x_2d.stride(0),
        x_2d.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        is_training=training,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return out.reshape(*orig_shape[:-1], N)


class FusedLinearDropout(nn.Module):
    def __init__(self, in_features: int, out_features: int, p_drop: float = 0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.p_drop = p_drop
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_linear_dropout(x, self.weight, self.bias, self.p_drop, self.training)
