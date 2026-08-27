import torch
import torch.nn as nn
from torch.nn import functional as F

from model.block import Block
from model.config import GPTConfig
from nodes.embeddings import PositionEmbedding, TokenEmbedding
from nodes.head import LMHead
from nodes.normalization import LayerNorm
from shared.types import CacheListType


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_emb = TokenEmbedding(config.vocab_size, config.n_embd)
        self.pos_emb = PositionEmbedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd)
        self.lm_head = LMHead(config.n_embd, config.vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor = None,
        caches: CacheListType = None,
    ):
        B, T = idx.shape  # (B, T)
        assert T <= self.config.block_size, "sequence longer than block_size"

        # initialize token embeddings and position embeddings
        if caches is not None and caches[0] is not None:
            start_pos = caches[0][0].shape[2]  # seq_len for K tensor in block 0
        else:
            start_pos = 0

        pos = torch.arange(start_pos, start_pos + T, device=idx.device)  # (T,)

        x = self.token_emb(idx) + self.pos_emb(pos)  # (B, T, C)
        x = self.dropout(x)

        # do this so caches will be an iterable
        if caches is None:
            caches = [None] * len(self.blocks)

        new_caches = []  # list to store new KV caches

        # run all block layers and capture each new KV cache
        for block, cache in zip(self.blocks, caches):
            x, new_cache = block(x, cache)
            new_caches.append(new_cache)

        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, new_caches

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=False):
        if use_cache:
            return self._generate_cached(idx, max_new_tokens, temperature, top_k)
        else:
            return self._generate_naive(idx, max_new_tokens, temperature, top_k)

    def _generate_naive(self, idx, max_new_tokens, temperature, top_k):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)

        return idx

    def _generate_cached(self, idx, max_new_tokens, temperature, top_k):
        # prefill: process entire prompt, build KV caches
        idx_cond = idx[:, -self.config.block_size :]  # (B, T)

        # ensure context window doesn't exceed block_size
        max_new_tokens = min(max_new_tokens, self.config.block_size - idx_cond.shape[1] - 1)

        logits, _, caches = self(idx_cond)
        logits = logits[:, -1, :] / temperature  # (B, vocab_size)

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
        idx = torch.cat((idx, next_id), dim=1)

        # decode loop: one token at a time using KV cache
        for _ in range(max_new_tokens - 1):
            logits, _, caches = self(next_id, caches=caches)  # (B, 1, vocab_size)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, next_id), dim=1)

        return idx
