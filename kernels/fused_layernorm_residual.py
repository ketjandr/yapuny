import triton
import triton.language as tl
import torch


@triton.jit
def _fused_layernorm_residual_kernel(
    x_ptr, y_ptr, out_ptr, weight_ptr, bias_ptr,
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
    y = tl.load(y_ptr + ptr_offsets, mask=mask, other=0.0)

    # z is the result of the residual add
    z = x + y

    # calculate normalized embedding from mean and variance
    mean = tl.sum(z) / n_cols
    variance = tl.sum((z - mean) ** 2) / n_cols
    z_norm = (z - mean) * tl.math.rsqrt(variance + eps)

    # apply weight * z_norm + bias
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    out = weight * z_norm + bias

    tl.store(out_ptr + ptr_offsets, out, mask=mask)

def fused_layernorm_residual(x, y, weight, bias, eps=1e-5):
    """
    Python wrapper — reshapes, picks BLOCK_SIZE, launches grid.
    x: (B, T, C) or (B*T, C) — residual
    y: same shape — block output
    weight: (C,)
    bias: (C,)
    Returns: out (same shape as x)
    """

    # flatten to 2D: (B*T, C)
    orig_shape = x.shape # (B, T, C)
    x_flatten = x.reshape(-1, x.shape[-1])
    y_flatten = y.reshape(-1, y.shape[-1])
    n_rows, n_cols = x_flatten.shape

    # Allocate output
    out = torch.empty_like(x_flatten)

    # Pick BLOCK_SIZE: next power of 2 >= n_cols
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Launch grid: one program per row
    grid = (n_rows,)
    _fused_layernorm_residual_kernel[grid](
        x_flatten, y_flatten, out,
        weight, bias,
        n_cols,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out.reshape(orig_shape)
