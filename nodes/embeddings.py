import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.embedding(idx)


class PositionEmbedding(nn.Module):
    def __init__(self, block_size: int, n_embd: int):
        super().__init__()
        self.embedding = nn.Embedding(block_size, n_embd)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.embedding(positions)
