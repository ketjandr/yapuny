import torch
import torch.nn as nn


class Dropout(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x)
