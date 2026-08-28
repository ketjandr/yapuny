import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_linear_gelu_kernel(
    x_ptr,  # (M, K)
    w_ptr,  # (N, K)
    bias_ptr,  # (N,)
    out_ptr,  # (M, N)
    M,  # rows in x
    N,  # output features
    K,  # input features
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
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

    # gelu: apply elementwise GELU
    inner = 0.7978845608 * (acc + 0.044715 * acc * acc * acc)
    # clamp to prevent exp overflow (tanh(10) == 0.9999...)
    inner = tl.where(inner > 10.0, 10.0, tl.where(inner < -10.0, -10.0, inner))
    e2 = tl.exp(2.0 * inner)
    acc = 0.5 * acc * (1.0 + (e2 - 1.0) / (e2 + 1.0))

    # store
    out_ptrs = out_ptr + rm_offsets[:, None] * stride_om + rn_offsets[None, :] * stride_on
    mask_out = (rm_offsets[:, None] < M) & (rn_offsets[None, :] < N)
    tl.store(out_ptrs, acc, mask=mask_out)


def fused_linear_gelu(
    x: torch.Tensor,  # (M, K) or (B, T, C)
    weight: torch.Tensor,  # (N, K)
    bias: torch.Tensor,  # (N,)
) -> torch.Tensor:
    """Fused linear (matmul + bias) + GELU activation."""
    orig_shape = x.shape
    # flatten to 2D
    x_2d = x.reshape(-1, x.shape[-1])
    M, K = x_2d.shape
    N = weight.shape[0]

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # tile sizes (arbitrary)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _fused_linear_gelu_kernel[grid](
        x_2d,
        weight,
        bias,
        out,
        M,
        N,
        K,
        x_2d.stride(0),
        x_2d.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # reshape back: (*orig_shape[:-1], N)
    return out.reshape(*orig_shape[:-1], N)


class FusedLinearGELU(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def init_weights(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.bias)

    def load_from_nodes(self, nodes: dict):
        up = nodes["mlp_up"]
        self.weight.data.copy_(up.fc.weight)
        self.bias.data.copy_(up.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_linear_gelu(x, self.weight, self.bias)
