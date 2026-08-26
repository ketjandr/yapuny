import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.norm = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)
