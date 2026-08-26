import torch
import torch.nn as nn

from model.config import GPTConfig
from model.types import CacheType
from nodes.attention import CausalAttention, OutProjection, QKVProjection
from nodes.mlp import MLPActivation, MLPDown, MLPUp
from nodes.normalization import LayerNorm
from nodes.residual import ResidualAdd


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd)
        self.qkv_proj = QKVProjection(config.n_embd, config.n_head)
        self.attn = CausalAttention(config.block_size, config.n_embd // config.n_head, config.dropout)
        self.out_proj = OutProjection(config.n_embd, config.dropout)
        self.residual1 = ResidualAdd()

        self.ln2 = LayerNorm(config.n_embd)
        self.mlp_up = MLPUp(config.n_embd)
        self.mlp_act = MLPActivation()
        self.mlp_down = MLPDown(config.n_embd, config.dropout)
        self.residual2 = ResidualAdd()

    def forward(self, x: torch.Tensor, cache: CacheType = None) -> tuple[torch.Tensor, CacheType]:
        normed = self.ln1(x)
        q, k, v = self.qkv_proj(normed)
        attn_out, new_cache = self.attn(q, k, v, cache)
        attn_out = self.out_proj(attn_out)
        x = self.residual1(attn_out, x)

        normed = self.ln2(x)
        mlp_out = self.mlp_down(self.mlp_act(self.mlp_up(normed)))
        x = self.residual2(mlp_out, x)

        return (x, new_cache)
