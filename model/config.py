from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 8000
    block_size: int = 256 # max sequence length (context window)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384 # must be divisible by n_head
    dropout: float = 0.1
