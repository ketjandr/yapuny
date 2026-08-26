import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _quantized_linear_w4_kernel(
    x_ptr,  # (M, K)
    w_q_ptr,  # (N, K//2), dtype uint8 (packed INT4)
    scale_ptr,  # (N,), dtype float32
    bias_ptr,  # (N,)
    y_ptr,  # (M, N)
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_scale = rn_offsets < N
    scale = tl.load(scale_ptr + rn_offsets, mask=mask_scale, other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # linear layer: tiled reduction over K + dequantize/unpacking
    for k_start in range(0, K, BLOCK_K):
        # load packed quantized w_packed_tile, holding two INT4 per byte
        k_packed_offsets = k_start // 2 + tl.arange(0, BLOCK_K // 2)
        w_q_ptrs = w_q_ptr + rn_offsets[:, None] * stride_wn + k_packed_offsets[None, :] * stride_wk
        mask_w = (rn_offsets[:, None] < N) & (k_packed_offsets[None, :] < K // 2)
        w_packed_tile = tl.load(w_q_ptrs, mask=mask_w, other=0) # (BLOCK_N, BLOCK_K // 2)

        # unpack nibbles and sign extend
        lo = ((w_packed_tile & 0xF) ^ 8) - 8
        hi = (((w_packed_tile >> 4) & 0xF) ^ 8) - 8

        # dequantize both lo and hi
        lo_dq = lo.to(tl.float32) * scale[:, None] # (BLOCK_N, BLOCK_K // 2)
        hi_dq = hi.to(tl.float32) * scale[:, None] # (BLOCK_N, BLOCK_K // 2)

        # load x in two halves
        k_even_offsets = k_start + tl.arange(0, BLOCK_K // 2) * 2
        k_odd_offsets = k_even_offsets + 1

        x_even_ptrs = x_ptr + rm_offsets[:, None] * stride_xm + k_even_offsets[None, :] * stride_xk
        x_odd_ptrs = x_ptr + rm_offsets[:, None] * stride_xm + k_odd_offsets[None, :] * stride_xk
        mask_x_even = (rm_offsets[:, None] < M) & (k_even_offsets[None, :] < K)
        mask_x_odd = (rm_offsets[:, None] < M) & (k_odd_offsets[None, :] < K)

        x_even = tl.load(x_even_ptrs, mask=mask_x_even, other=0.0) # (BLOCK_M, BLOCK_K // 2)
        x_odd = tl.load(x_odd_ptrs, mask=mask_x_odd, other=0.0) # (BLOCK_M, BLOCK_K // 2)

        # accumulate the even/lo and odd/hi activations/weights
        # two half-width dot products = one full dot product over K
        acc += tl.dot(x_even, tl.trans(lo_dq))
        acc += tl.dot(x_odd, tl.trans(hi_dq))

    mask_bias = rn_offsets < N
    bias = tl.load(bias_ptr + rn_offsets, mask=mask_bias, other=0.0)
    acc += bias[None, :]

    y_ptrs = y_ptr + rm_offsets[:, None] * stride_ym + rn_offsets[None, :] * stride_yn
    mask_y = (rm_offsets[:, None] < M) & (rn_offsets[None, :] < N)
    tl.store(y_ptrs, acc, mask=mask_y)


def quantized_linear_w4(
    x: torch.Tensor,  # (M, K) or (B, T, C)
    w_q: torch.Tensor,  # (N, K//2), dtype uint8 (packed)
    scale: torch.Tensor,  # (N,), dtype float32
    bias: torch.Tensor,  # (N,)
) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    M, K = x_2d.shape
    N = w_q.shape[0]

    y = torch.empty((M, N), device=x.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _quantized_linear_w4_kernel[grid](
        x_2d, w_q, scale, bias, y,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        w_q.stride(0), w_q.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return y.reshape(*orig_shape[:-1], N)


class QuantizedLinearW4(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        assert in_features % 2 == 0
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("w_q", torch.empty(out_features, in_features // 2, dtype=torch.uint8))
        self.register_buffer("scale", torch.empty(out_features, dtype=torch.float32))
        self.register_buffer("bias", torch.empty(out_features, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return quantized_linear_w4(x, self.w_q, self.scale, self.bias)
