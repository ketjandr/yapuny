import random

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_dropout_residual_kernel(
    x_ptr,
    residual_ptr,
    out_ptr,
    seed,
    n_elements,
    p_drop,
    is_training: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    ptr_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = ptr_offsets < n_elements

    residual = tl.load(residual_ptr + ptr_offsets, mask=mask, other=0.0)

    # dropout: perform only during training
    if is_training:
        dropout_mask = tl.rand(seed, ptr_offsets)
        residual = tl.where(dropout_mask >= p_drop, residual / (1 - p_drop), 0.0)

    # residual add
    x = tl.load(x_ptr + ptr_offsets, mask=mask, other=0.0)
    out = x + residual

    tl.store(out_ptr + ptr_offsets, out, mask=mask)


def fused_dropout_residual(
    x: torch.Tensor,  # original input
    residual: torch.Tensor,  # sublayer output (dropout applied to this)
    p_drop: float = 0.1,
    training: bool = True,
) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = x.numel()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    seed = random.randint(0, 2**31)

    _fused_dropout_residual_kernel[grid](
        x,
        residual,
        out,
        seed,
        n_elements,
        p_drop,
        is_training=training,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


class FusedDropoutResidual(nn.Module):
    def __init__(self, p_drop: float = 0.1):
        super().__init__()
        self.p_drop = p_drop

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return fused_dropout_residual(x, residual, self.p_drop, self.training)
