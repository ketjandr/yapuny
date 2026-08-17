import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_scale_mask_softmax_kernel(
    scores_ptr,
    causal_mask_ptr,
    out_ptr,
    scale,
    T,
    T_k,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < T_k

    # load scores and scale
    ptr_offsets = pid * T_k + col_offsets
    scores = tl.load(scores_ptr + ptr_offsets, mask=mask, other=float("-inf"))
    scores *= scale

    # apply causal mask
    causal_mask_row = pid % T
    causal_mask_offsets = causal_mask_row * T_k + col_offsets
    causal_mask = tl.load(causal_mask_ptr + causal_mask_offsets, mask=mask, other=0.0)
    scores = tl.where(causal_mask == 1, scores, float("-inf"))

    # safe "shifted" softmax
    row_max = tl.max(scores, axis=0)
    scores_minus_max = scores - row_max # shift so we don't overflow
    numerator = tl.exp(scores_minus_max)
    denominator = tl.sum(numerator)
    out = numerator / denominator

    tl.store(out_ptr + ptr_offsets, out, mask=mask)

def fused_scale_mask_softmax(
    scores: torch.Tensor,  # (B, n_head, T, T_k) — raw Q@K^T
    mask: torch.Tensor,    # (1, 1, T, T_k) or (T, T_k) — causal mask (1=keep, 0=mask)
    scale: float,          # 1/sqrt(head_dim)
) -> torch.Tensor:         # (B, n_head, T, T_k) — softmax probabilities
    orig_shape = scores.shape
    B, n_head, T, T_k = orig_shape

    # flatten to 2D: (B*n_head*T, T_k)
    scores_2d = scores.reshape(-1, T_k)
    n_rows = scores_2d.shape[0]

    # flatten mask to 2D: (T, T_k)
    # the mask repeats every T rows (same mask for each batch/head)
    mask_2d = mask.squeeze(0).squeeze(0)  # (T, T_k)

    out = torch.empty_like(scores_2d)

    BLOCK_SIZE = triton.next_power_of_2(T_k)

    # launch grid: one program per row
    grid = (n_rows,)
    _fused_scale_mask_softmax_kernel[grid](
        scores_2d, mask_2d, out,
        scale,
        T,
        T_k,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out.reshape(orig_shape)


class FusedScaleMaskSoftmax(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.scale = 1.0 / (head_dim ** 0.5)

    def forward(self, scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return fused_scale_mask_softmax(scores, mask, self.scale)
