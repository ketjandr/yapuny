from __future__ import annotations

import torch

from server.compiler.compiler import GraphCompiler
from server.models.graph import GraphSpec


class Worker:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.compiler = GraphCompiler()
        self.model = None

    def compile_graph(self, graph_data: dict):
        graph = GraphSpec.from_dict(graph_data)
        self.model = self.compiler.compile(graph)
        self.model.to(self.device)
        self.model.eval()
        return {"status": "compiled", "device": str(self.device)}

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
    ):
        if self.model is None:
            return {"error": "no model compiled"}

        idx = torch.tensor([prompt_ids], device=self.device)
        tokens = []

        for _ in range(max_new_tokens):
            logits, _, _ = self.model(idx)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            tokens.append(next_id.item())

        return {"tokens": tokens}
