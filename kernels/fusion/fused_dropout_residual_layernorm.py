import random

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_dropout_residual_layernorm_kernel(
    x_ptr,
    residual_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    seed,
    n_rows,
    n_cols,
    p_drop,
    eps: tl.constexpr,
    is_training: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    ptr_offsets = pid * n_cols + col_offsets

    x = tl.load(x_ptr + ptr_offsets, mask=mask, other=0.0)

    # dropout on sublayer output
    if is_training:
        dropout_mask = tl.rand(seed, ptr_offsets)
        x = tl.where(dropout_mask >= p_drop, x / (1 - p_drop), 0.0)

    # residual add
    residual = tl.load(residual_ptr + ptr_offsets, mask=mask, other=0.0)
    z = residual + x

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


def fused_dropout_residual_layernorm(
    x: torch.Tensor,  # sublayer output (dropout applied to this)
    residual: torch.Tensor,  # skip connection
    weight: torch.Tensor,  # (C,) layernorm weight
    bias: torch.Tensor,  # (C,) layernorm bias
    p_drop: float = 0.1,
    training: bool = True,
    eps: float = 1e-5,
) -> torch.Tensor:
    orig_shape = x.shape
    x_flat = x.reshape(-1, x.shape[-1])
    residual_flat = residual.reshape(-1, residual.shape[-1])
    n_rows, n_cols = x_flat.shape

    out = torch.empty_like(x_flat)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    seed = random.randint(0, 2**31)

    _fused_dropout_residual_layernorm_kernel[grid](
        x_flat,
        residual_flat,
        out,
        weight,
        bias,
        seed,
        n_rows,
        n_cols,
        p_drop,
        eps=eps,
        is_training=training,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out.reshape(orig_shape)


class FusedDropoutResidualLayerNorm(nn.Module):
    def __init__(self, n_embd: int, p_drop: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd))
        self.p_drop = p_drop
        self.eps = eps

    def init_weights(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def load_from_nodes(self, nodes: dict):
        ln = nodes["layernorm"]
        self.weight.data.copy_(ln.norm.weight)
        self.bias.data.copy_(ln.norm.bias)

    def save_to_nodes(self, nodes: dict):
        ln = nodes["layernorm"]
        ln.norm.weight.data.copy_(self.weight)
        ln.norm.bias.data.copy_(self.bias)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return fused_dropout_residual_layernorm(
            x, residual, self.weight, self.bias, self.p_drop, self.training, self.eps
        )
