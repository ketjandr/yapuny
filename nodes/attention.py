import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from model.types import CacheType


class QKVProjection(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        # combined QKV projection, 1 matmul instead of 3
        self.proj = nn.Linear(n_embd, 3 * n_embd)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, C = x.shape  # batch, sequence length, embedding dim

        qkv = self.proj(x) # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2) # (B, T, C) each

        # reshape to (B, n_head, T, head_dim) for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        return q, k, v


class CausalAttention(nn.Module):
    """Scaled dot-product attention with causal mask and KV cache.

    Flash attention replaces exactly this node.
    """

    def __init__(self, block_size: int, head_dim: int, dropout: float):
        super().__init__()
        self.head_dim = head_dim
        self.attn_dropout = nn.Dropout(dropout)
        # apply causal mask
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cache: CacheType = None,
    ) -> tuple[torch.Tensor, CacheType]:
        # concat k, v into cached_k, cached_v
        if cache is not None:
            cached_k, cached_v = cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        T = q.shape[2]
        S = k.shape[2]

        # scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim)) # (B, n_head, T, T)

        # masking only happens when T > 1 (e.g. when cache is None)
        if T > 1:
            att = att.masked_fill(self.mask[:, :, :T, :S] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v # (B, n_head, T, head_dim)
        # return the out matrix and new KV cache
        return out, (k, v)


class OutProjection(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, n_head, T, head_dim = x.shape
        x = x.transpose(1, 2).contiguous().view(B, T, n_head * head_dim) # merge heads back (B, T, C)
        return self.resid_dropout(self.proj(x))
