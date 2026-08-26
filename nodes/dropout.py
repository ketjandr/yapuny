import torch
import torch.nn as nn


class Dropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x)
