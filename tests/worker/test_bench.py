from __future__ import annotations

import torch

from server.compiler.compiler import GraphCompiler
from server.models.graph import GraphSpec
from tests.server.graph_factory import default_gpt_graph
from worker.bench import (
    ProfileResult,
    profile_graph,
)
from worker.worker import Worker

TINY = dict(n_layer=2, n_head=2, n_embd=32, block_size=16, vocab_size=64)


def _compile(graph_dict: dict) -> tuple:
    compiler = GraphCompiler()
    graph = GraphSpec.from_dict(graph_dict)
    model = compiler.compile(graph)
    model.eval()
    return model, graph


class TestBenchFlag:
    def _make_worker(self):
        model, _ = _compile(default_gpt_graph(**TINY))
        w = Worker.__new__(Worker)
        w.model = model
        w.device = torch.device("cpu")
        return w

    def test_generate_bench_returns_timing(self):
        w = self._make_worker()
        result = w.generate(prompt_ids=[1, 2, 3], max_new_tokens=5, bench=True)
        assert "bench" in result
        assert result["bench"]["prefill_ms"] > 0
        assert result["bench"]["tokens_per_sec"] > 0
        assert result["bench"]["decode_ms_per_token"] > 0
        assert result["bench"]["peak_vram_mb"] is None

    def test_generate_no_bench_by_default(self):
        w = self._make_worker()
        result = w.generate(prompt_ids=[1, 2, 3], max_new_tokens=5)
        assert "bench" not in result
        assert "tokens" in result
        assert len(result["tokens"]) == 5


class TestProfileGraph:
    def test_decode_returns_node_percentages(self):
        model, _ = _compile(default_gpt_graph(**TINY))
        result = profile_graph(
            model,
            torch.device("cpu"),
            mode="decode",
            prompt_tokens=4,
            new_tokens=4,
            batch_size=1,
            warmup=1,
        )
        assert isinstance(result, ProfileResult)
        assert len(result.nodes) > 0
        assert result.total_us > 0
        total_pct = sum(n.pct for n in result.nodes)
        assert abs(total_pct - 100.0) < 0.1
        assert result.nodes[0].pct >= result.nodes[-1].pct
        assert all(n.node_id for n in result.nodes)

    def test_train_returns_node_percentages(self):
        model, _ = _compile(default_gpt_graph(**TINY))
        result = profile_graph(
            model,
            torch.device("cpu"),
            mode="train",
            batch_size=2,
            warmup=1,
        )
        assert isinstance(result, ProfileResult)
        assert len(result.nodes) > 0
        assert not model.training

    def test_profile_flag_reset_after_run(self):
        model, _ = _compile(default_gpt_graph(**TINY))
        assert not model.profile
        profile_graph(
            model,
            torch.device("cpu"),
            mode="decode",
            prompt_tokens=4,
            new_tokens=4,
            batch_size=1,
            warmup=1,
        )
        assert not model.profile

    def test_profile_flag_reset_on_error(self):
        model, _ = _compile(default_gpt_graph(**TINY))
        model.profile = False
        try:
            profile_graph(
                model,
                torch.device("cpu"),
                mode="decode",
                prompt_tokens=4,
                new_tokens=0,
                batch_size=1,
                warmup=0,
            )
        except Exception:
            pass
        assert not model.profile
