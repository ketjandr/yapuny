import torch
import torch.nn as nn
from torch.nn import functional as F


class MLPUp(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLPActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


class MLPDown(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.fc = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc(x))
