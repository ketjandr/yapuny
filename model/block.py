import torch
import torch.nn as nn

from model.config import GPTConfig
from nodes.attention import (
    AttentionScore,
    CausalMask,
    KVCache,
    OutProjection,
    QKVProjection,
    Softmax,
    ValueWeightedSum,
)
from nodes.dropout import Dropout
from nodes.mlp import MLPActivation, MLPDown, MLPUp
from nodes.normalization import LayerNorm
from nodes.residual import ResidualAdd
from shared.types import CacheType


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        head_dim = config.n_embd // config.n_head

        # attention subgraph
        self.ln1 = LayerNorm(config.n_embd)
        self.qkv_proj = QKVProjection(config.n_embd, config.n_head)
        self.kv_cache = KVCache()
        self.attn_score = AttentionScore(head_dim)
        self.causal_mask = CausalMask(config.block_size)
        self.softmax = Softmax()
        self.attn_dropout = Dropout(config.dropout)
        self.value_weighted_sum = ValueWeightedSum()
        self.out_proj = OutProjection(config.n_embd)
        self.resid_dropout1 = Dropout(config.dropout)
        self.residual1 = ResidualAdd()

        # mlp subgraph
        self.ln2 = LayerNorm(config.n_embd)
        self.mlp_up = MLPUp(config.n_embd)
        self.mlp_act = MLPActivation()
        self.mlp_down = MLPDown(config.n_embd)
        self.resid_dropout2 = Dropout(config.dropout)
        self.residual2 = ResidualAdd()

    def forward(self, x: torch.Tensor, cache: CacheType = None) -> tuple[torch.Tensor, CacheType]:
        # attention
        normed = self.ln1(x)
        q, k, v = self.qkv_proj(normed)
        k, v, new_cache = self.kv_cache(k, v, cache)
        att = self.attn_score(q, k)
        att = self.causal_mask(att)
        att = self.softmax(att)
        att = self.attn_dropout(att)
        attn_out = self.value_weighted_sum(att, v)
        attn_out = self.resid_dropout1(self.out_proj(attn_out))
        x = self.residual1(attn_out, x)

        # mlp
        normed = self.ln2(x)
        mlp_out = self.resid_dropout2(self.mlp_down(self.mlp_act(self.mlp_up(normed))))
        x = self.residual2(mlp_out, x)

        return (x, new_cache)
