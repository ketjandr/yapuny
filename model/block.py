import torch
import torch.nn as nn

from model.attention import CausalSelfAttention
from model.config import GPTConfig
from model.mlp import MLP
from model.types import CacheType


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, cache: CacheType = None) -> tuple[torch.Tensor, CacheType]:
        attn_out, new_cache = self.attn(self.ln1(x), cache)
        x += attn_out
        mlp_out = self.mlp(self.ln2(x))
        x += mlp_out
        return (x, new_cache)
