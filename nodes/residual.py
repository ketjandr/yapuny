import torch
import torch.nn as nn


class ResidualAdd(nn.Module):
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return x + residual
