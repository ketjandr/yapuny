import torch
import torch.nn as nn
from torch.nn import functional as F


class MLPUp(nn.Module):
    """Up-project: n_embd - 4*n_embd."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLPActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


class MLPDown(nn.Module):
    """Down-project: 4*n_embd - n_embd."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.fc = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
