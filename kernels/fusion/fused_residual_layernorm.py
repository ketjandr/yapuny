import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_residual_layernorm_kernel(
    x_ptr,
    residual_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # ptr_offsets is specific to this pid
    ptr_offsets = pid * n_cols + col_offsets
    x = tl.load(x_ptr + ptr_offsets, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + ptr_offsets, mask=mask, other=0.0)

    # residual add
    z = x + residual

    # layernorm
    # calculate normalized embedding from mean and variance
    mean = tl.sum(z) / n_cols
    diff = tl.where(mask, z - mean, 0.0)
    variance = tl.sum(diff * diff) / n_cols
    z_norm = (z - mean) * tl.math.rsqrt(variance + eps)

    # apply weight * z_norm + bias
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    out = weight * z_norm + bias

    tl.store(out_ptr + ptr_offsets, out, mask=mask)


def fused_residual_layernorm(
    x: torch.Tensor,  # (B, T, C) or (B*T, C) - original embeddings
    residual: torch.Tensor,  # (B, T, C) or (B*T, C) - sublayer deltas
    weight: torch.Tensor,  # (C,)
    bias: torch.Tensor,  # (C,)
    eps: float = 1e-5,
) -> torch.Tensor:  # (B, T, C)

    # flatten to 2D (B*T, C)
    orig_shape = x.shape  # (B, T, C)
    x_flatten = x.reshape(-1, x.shape[-1])
    residual_flatten = residual.reshape(-1, residual.shape[-1])
    n_rows, n_cols = x_flatten.shape  # (B*T, C)

    out = torch.empty_like(x_flatten)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # launch grid: one program per row
    grid = (n_rows,)
    _fused_residual_layernorm_kernel[grid](
        x_flatten,
        residual_flatten,
        out,
        weight,
        bias,
        n_cols,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out.reshape(orig_shape)


class FusedResidualLayerNorm(nn.Module):
    def __init__(self, n_embd: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd))
        self.eps = eps

    def init_weights(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def load_from_nodes(self, nodes: dict):
        ln = nodes["layernorm"]
        self.weight.data.copy_(ln.norm.weight)
        self.bias.data.copy_(ln.norm.bias)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return fused_residual_layernorm(x, residual, self.weight, self.bias, self.eps)
