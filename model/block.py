import torch.nn as nn

from model.attention import CausalSelfAttention
from model.config import GPTConfig
from model.mlp import MLP


class Block(nn.Module):
    """One transformer block: attention + MLP, each with pre-norm + residual."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
