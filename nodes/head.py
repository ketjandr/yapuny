import torch
import torch.nn as nn


class LMHead(nn.Module):
    """Project hidden states to vocab logits (no bias)."""

    def __init__(self, n_embd: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
