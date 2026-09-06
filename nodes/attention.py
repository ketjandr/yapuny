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


class RoPE(nn.Module):
    """Rotary position embedding: rotates Q and K by a position-dependent angle. Because the
    attention score q_i . k_j then depends only on the offset (i - j), positions are relative, so
    a KVCache can roll (evict old entries) and still be correct past the context window."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        # inverse frequencies for each rotation plane (half of head_dim planes)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # cached cos/sin for the training positions, keyed on (T, device, dtype)
        self._cache_key: tuple | None = None
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _cos_sin(self, positions: torch.Tensor, dtype: torch.dtype):
        freqs = torch.outer(positions.float(), self.inv_freq)  # (T, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, head_dim)
        # (1, 1, T, head_dim)
        return emb.cos()[None, None].to(dtype), emb.sin()[None, None].to(dtype)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k: (B, n_head, T, head_dim); positions: (T,), absolute, can grow unbounded.
        # cos/sin depend only on positions. In training positions are a fixed arange(0, T) reused
        # every step, so cache them and rebuild only when shape/device/dtype changes (never mid-run).
        # Inference passes new positions each step, so recompute there (cheap at T=1).
        key = (positions.shape[0], positions.device, q.dtype)
        if self.training and self._cache_key == key:
            cos, sin = self._cos, self._sin
        else:
            cos, sin = self._cos_sin(positions, q.dtype)
            if self.training:
                self._cache_key, self._cos, self._sin = key, cos, sin
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot.type_as(q), k_rot.type_as(k)


class KVCache(nn.Module):
    """Appends new K, V to the cache. With rotary positions (a RoPE node upstream) the cache can
    roll: once it exceeds block_size it keeps only the most recent block_size entries, so decoding
    past the context window stays O(1) per token instead of recomputing the whole window. For
    absolute position_embedding graphs the generation loop recomputes on overflow instead."""

    def __init__(self, block_size: int):
        super().__init__()
        self.block_size = block_size

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
        # roll: evict the oldest entries so the window never exceeds block_size
        if k.shape[2] > self.block_size:
            k = k[:, :, -self.block_size:, :]
            v = v[:, :, -self.block_size:, :]
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
