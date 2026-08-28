import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from shared.types import CacheType


class QKVProjection(nn.Module):
    """Single linear - split into Q, K, V - reshape to multi-head."""

    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        # combined QKV projection, 1 matmul instead of 3
        self.proj = nn.Linear(n_embd, 3 * n_embd)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, C = x.shape  # batch, sequence length, embedding dim

        qkv = self.proj(x)  # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)  # (B, T, C) each

        # reshape to (B, n_head, T, head_dim) for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        return q, k, v


class KVCache(nn.Module):
    """Concatenates new K, V with cached K, V from previous steps."""

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        cache: CacheType = None,
    ) -> tuple[torch.Tensor, torch.Tensor, CacheType]:
        # concat k, v into cached_k, cached_v
        if cache is not None:
            cached_k, cached_v = cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        return k, v, (k, v)


class AttentionScore(nn.Module):
    """Q @ K^T scaled by 1/sqrt(head_dim)."""

    def __init__(self, head_dim: int):
        super().__init__()
        self.scale = 1.0 / math.sqrt(head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # scaled dot-product attention
        return (q @ k.transpose(-2, -1)) * self.scale  # (B, n_head, T, S)


class CausalMask(nn.Module):
    """Applies lower-triangular causal mask to attention scores."""

    def __init__(self, block_size: int):
        super().__init__()
        # apply causal mask
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[2]
        S = x.shape[3]
        # masking only happens when T > 1 (e.g. when cache is None)
        if T > 1:
            x = x.masked_fill(self.mask[:, :, :T, :S] == 0, float("-inf"))
        return x


class Softmax(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=-1)


class ValueWeightedSum(nn.Module):
    """att_probs @ V - weighted combination of value vectors."""

    def forward(self, att: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return att @ v  # (B, n_head, T, head_dim)


class OutProjection(nn.Module):
    """Merge heads back to (B, T, C) and project."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, n_head, T, head_dim = x.shape
        x = (
            x.transpose(1, 2).contiguous().view(B, T, n_head * head_dim)
        )  # merge heads back (B, T, C)
        return self.proj(x)
