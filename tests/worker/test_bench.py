from __future__ import annotations

import torch

from server.compiler.compiler import GraphCompiler
from server.models.graph import GraphSpec
from tests.server.graph_factory import default_gpt_graph
from worker.bench import (
    BenchResult,
    TimingResult,
    bench_graph,
    run_benchmark,
)
from worker.worker import Worker

TINY = dict(n_layer=2, n_head=2, n_embd=32, block_size=16, vocab_size=64)


def _compile(graph_dict: dict) -> tuple:
    compiler = GraphCompiler()
    graph = GraphSpec.from_dict(graph_dict)
    model = compiler.compile(graph)
    model.eval()
    return model, graph


class TestBenchGraph:
    def test_inference_returns_expected_keys(self):
        model, _ = _compile(default_gpt_graph(**TINY))
        result = bench_graph(
            model,
            torch.device("cpu"),
            mode="decode",
            prompt_tokens=4,
            new_tokens=8,
            batch_size=1,
            repeats=3,
            warmup=1,
        )
        assert isinstance(result["prefill_ms"], TimingResult)
        assert isinstance(result["decode_ms_per_token"], TimingResult)
        assert result["tokens_per_sec"] > 0
        assert result["param_count"] > 0
        assert result["weight_bytes"] > 0
        assert result["peak_vram_mb"] is None  # CPU
        assert result.get("steps_per_sec") is None
        assert result.get("forward_ms") is None

    def test_train_returns_expected_keys(self):
        model, _ = _compile(default_gpt_graph(**TINY))
        result = bench_graph(
            model,
            torch.device("cpu"),
            mode="train",
            batch_size=2,
            repeats=3,
            warmup=1,
        )
        assert isinstance(result["forward_ms"], TimingResult)
        assert isinstance(result["backward_ms"], TimingResult)
        assert result["steps_per_sec"] > 0
        assert result["param_count"] > 0
        assert result.get("prefill_ms") is None
        assert result.get("tokens_per_sec") is None

    def test_timing_result_spread(self):
        model, _ = _compile(default_gpt_graph(**TINY))
        result = bench_graph(
            model,
            torch.device("cpu"),
            mode="decode",
            prompt_tokens=4,
            new_tokens=8,
            batch_size=1,
            repeats=5,
            warmup=1,
        )
        tr = result["prefill_ms"]
        assert tr.p05 <= tr.median <= tr.p95
        assert len(tr.samples) == 5

    def test_model_back_to_eval_after_train_bench(self):
        model, _ = _compile(default_gpt_graph(**TINY))
        bench_graph(
            model,
            torch.device("cpu"),
            mode="train",
            batch_size=2,
            repeats=2,
            warmup=1,
        )
        assert not model.training


class TestRunBenchmark:
    def test_single_graph(self):
        model, graph = _compile(default_gpt_graph(**TINY))
        from server.compiler.utils import graph_structure_hash

        result = run_benchmark(
            graphs=[
                {
                    "graph_id": "test-v1",
                    "structure_hash": graph_structure_hash(graph),
                    "model": model,
                    "meta": {"n_layer": 2, "n_embd": 32},
                }
            ],
            device=torch.device("cpu"),
            mode="decode",
            prompt_tokens=4,
            new_tokens=8,
            batch_size=1,
            repeats=3,
            warmup=1,
        )
        assert isinstance(result, BenchResult)
        assert len(result.graphs) == 1
        assert result.graphs[0].graph_id == "test-v1"
        assert result.mode == "decode"
        assert "torch" in result.env

    def test_structure_groups(self):
        model_a, graph_a = _compile(default_gpt_graph(**TINY))
        model_b, graph_b = _compile(default_gpt_graph(**TINY))
        different = dict(TINY, n_layer=1)
        model_c, graph_c = _compile(default_gpt_graph(**different))

        from server.compiler.utils import graph_structure_hash

        hash_ab = graph_structure_hash(graph_a)
        hash_c = graph_structure_hash(graph_c)
        assert hash_ab != hash_c

        result = run_benchmark(
            graphs=[
                {"graph_id": "a", "structure_hash": hash_ab, "model": model_a, "meta": {}},
                {"graph_id": "b", "structure_hash": hash_ab, "model": model_b, "meta": {}},
                {"graph_id": "c", "structure_hash": hash_c, "model": model_c, "meta": {}},
            ],
            device=torch.device("cpu"),
            mode="decode",
            prompt_tokens=4,
            new_tokens=4,
            batch_size=1,
            repeats=2,
            warmup=1,
        )
        assert len(result.structure_groups) == 2
        group_sizes = sorted(len(g) for g in result.structure_groups)
        assert group_sizes == [1, 2]

    def test_train_mode(self):
        model, graph = _compile(default_gpt_graph(**TINY))
        from server.compiler.utils import graph_structure_hash

        result = run_benchmark(
            graphs=[
                {
                    "graph_id": "train-test",
                    "structure_hash": graph_structure_hash(graph),
                    "model": model,
                    "meta": {},
                }
            ],
            device=torch.device("cpu"),
            mode="train",
            batch_size=2,
            repeats=3,
            warmup=1,
        )
        v = result.graphs[0]
        assert v.steps_per_sec > 0
        assert v.forward_ms is not None
        assert v.backward_ms is not None
        assert v.prefill_ms is None


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
