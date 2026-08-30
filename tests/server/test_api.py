import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from server.api.schemas import (
    BenchRunRequest,
    DecodeRequest,
    FusionGroupSchema,
    GenerateRequest,
    GraphMetaSchema,
    GraphRequest,
    NodeSchema,
    PrepareDataRequest,
    TrainRequest,
)
from server.app import app
from tests.server.graph_factory import default_gpt_graph

client = TestClient(app)


class TestFusionGroupSchema:
    def test_valid_group(self):
        fg = FusionGroupSchema(nodes=["a", "b"])
        assert fg.nodes == ["a", "b"]

    def test_kernel_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            FusionGroupSchema(nodes=["a", "b"], kernel="FusedDropoutResidual")

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            FusionGroupSchema(nodes=["a"], name="my group")

    def test_empty_nodes(self):
        fg = FusionGroupSchema(nodes=[])
        assert fg.nodes == []


class TestNodeSchema:
    def test_valid_node(self):
        n = NodeSchema(id="b0_ln1", type="layernorm")
        assert n.config == {}
        assert n.quantized is None

    def test_with_config(self):
        n = NodeSchema(id="b0_ln1", type="layernorm", config={"eps": 1e-5})
        assert n.config["eps"] == 1e-5

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            NodeSchema(id="b0", type="layernorm", position=3)

    def test_missing_required(self):
        with pytest.raises(ValidationError):
            NodeSchema(id="b0")


class TestGraphRequest:
    def test_minimal_valid(self):
        g = GraphRequest(
            nodes=[NodeSchema(id="tok", type="token_embedding")],
            edges=[],
        )
        assert len(g.fusion_groups) == 0
        assert g.meta.n_embd == 384

    def test_custom_meta(self):
        g = GraphRequest(
            nodes=[],
            edges=[],
            meta=GraphMetaSchema(n_embd=128),
        )
        assert g.meta.n_embd == 128

    def test_fusion_group_without_kernel(self):
        g = GraphRequest(
            nodes=[],
            edges=[],
            fusion_groups=[FusionGroupSchema(nodes=["a", "b"])],
        )
        assert g.fusion_groups[0].nodes == ["a", "b"]

    def test_fusion_group_with_kernel_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            GraphRequest(
                nodes=[],
                edges=[],
                fusion_groups=[{"nodes": ["a", "b"], "kernel": "Foo"}],
            )

    def test_extra_top_level_field_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            GraphRequest(
                nodes=[],
                edges=[],
                id="sneaky",
            )

    def test_model_dump_roundtrip(self):
        g = GraphRequest(
            nodes=[NodeSchema(id="tok", type="token_embedding")],
            edges=[],
            fusion_groups=[FusionGroupSchema(nodes=["a", "b"])],
        )
        d = g.model_dump()
        assert d["nodes"][0]["id"] == "tok"
        assert d["fusion_groups"][0] == {"nodes": ["a", "b"]}
        assert "kernel" not in d["fusion_groups"][0]


class TestGenerateRequest:
    def test_valid(self):
        r = GenerateRequest(prompt_ids=[1, 2, 3])
        assert r.max_new_tokens == 50
        assert r.temperature == 1.0
        assert r.bench is False

    def test_bench_flag(self):
        r = GenerateRequest(prompt_ids=[1, 2, 3], bench=True)
        assert r.bench is True

    def test_missing_prompt_ids(self):
        with pytest.raises(ValidationError):
            GenerateRequest()

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            GenerateRequest(prompt_ids=[1], beam_width=5)


class TestDecodeRequest:
    def test_valid(self):
        r = DecodeRequest(token_ids=[1, 2, 3])
        assert r.token_ids == [1, 2, 3]

    def test_missing_token_ids(self):
        with pytest.raises(ValidationError):
            DecodeRequest()


class TestTrainRequest:
    def test_defaults(self):
        r = TrainRequest()
        assert r.max_steps == 2000
        assert r.batch_size == 32
        assert r.learning_rate == 3e-4
        assert r.checkpoint_path is None
        assert r.bench is False

    def test_custom(self):
        r = TrainRequest(max_steps=100, batch_size=8)
        assert r.max_steps == 100
        assert r.batch_size == 8

    def test_bench_flag(self):
        r = TrainRequest(bench=True)
        assert r.bench is True


class TestPrepareDataRequest:
    def test_defaults(self):
        r = PrepareDataRequest()
        assert r.vocab_size == 8000
        assert r.val_fraction == 0.1


# -- route-level tests --


class TestValidateRoute:
    def test_invalid_fusion_pattern_returns_error(self):
        graph = default_gpt_graph(n_layer=1)
        graph["fusion_groups"] = [{"nodes": ["b0_ln1", "b0_resid_drop1"]}]
        resp = client.post("/api/graph/validate", json=graph)
        assert resp.status_code == 200
        body = resp.json()
        assert body["valid"] is False
        assert any("no fusion kernel matches pattern" in e for e in body["errors"])

    def test_fusion_with_kernel_field_rejected(self):
        graph = default_gpt_graph(n_layer=1)
        graph["fusion_groups"] = [
            {"nodes": ["b0_resid_drop1", "b0_res1"], "kernel": "FusedDropoutResidual"},
        ]
        resp = client.post("/api/graph/validate", json=graph)
        assert resp.status_code == 422

    def test_valid_graph_passes(self):
        graph = default_gpt_graph(n_layer=1)
        resp = client.post("/api/graph/validate", json=graph)
        assert resp.status_code == 200
        assert resp.json()["valid"] is True

    def test_unknown_node_in_fusion_returns_error(self):
        graph = default_gpt_graph(n_layer=1)
        graph["fusion_groups"] = [{"nodes": ["ghost", "b0_res1"]}]
        resp = client.post("/api/graph/validate", json=graph)
        assert resp.status_code == 200
        body = resp.json()
        assert body["valid"] is False
        assert any("unknown node" in e for e in body["errors"])


# -- benchmark schemas --


class TestBenchRunRequest:
    def test_defaults(self):
        r = BenchRunRequest(graph_ids=["a"], prompt_ids=[1, 2, 3])
        assert r.max_new_tokens == 50
        assert r.temperature == 1.0
        assert r.top_k is None

    def test_too_many_graph_ids_rejected(self):
        with pytest.raises(ValidationError):
            BenchRunRequest(graph_ids=[f"g{i}" for i in range(6)], prompt_ids=[1])

    def test_empty_graph_ids_rejected(self):
        with pytest.raises(ValidationError):
            BenchRunRequest(graph_ids=[], prompt_ids=[1])

    def test_missing_prompt_ids(self):
        with pytest.raises(ValidationError):
            BenchRunRequest(graph_ids=["a"])

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            BenchRunRequest(graph_ids=["a"], prompt_ids=[1], gpu="A100")


# -- benchmark route tests --


def _save_and_compile(graph_id: str, **kwargs):
    graph = default_gpt_graph(**kwargs)
    graph["id"] = graph_id
    client.post("/api/graph", json=graph)
    client.post("/api/graph/compile", json=graph)
    return graph


class TestGraphListRoute:
    def test_list_empty(self):
        from server.api.routes import _graphs

        _graphs.clear()
        resp = client.get("/api/graph")
        assert resp.status_code == 200
        assert resp.json()["graphs"] == []

    def test_list_after_save(self):
        from server.api.routes import _graphs

        _graphs.clear()
        graph = default_gpt_graph(n_layer=1, n_head=2, n_embd=32, block_size=16, vocab_size=64)
        graph["id"] = "test-list"
        client.post("/api/graph", json=graph)
        resp = client.get("/api/graph")
        body = resp.json()
        assert len(body["graphs"]) == 1
        assert body["graphs"][0]["id"] == "test-list"
        assert "structure_hash" in body["graphs"][0]


def _parse_sse(text: str) -> list[dict]:
    events = []
    for block in text.strip().split("\n\n"):
        event = {}
        for line in block.strip().split("\n"):
            if line.startswith("event: "):
                event["event"] = line[7:]
            elif line.startswith("data: "):
                import json

                event["data"] = json.loads(line[6:])
        if event:
            events.append(event)
    return events


class TestBenchRoutes:
    @pytest.fixture(autouse=True)
    def _clean(self):
        from server.api.routes import _graphs

        _graphs.clear()
        yield
        _graphs.clear()

    def test_streams_single_graph(self):
        _save_and_compile("bench-a", n_layer=1, n_head=2, n_embd=32, block_size=16, vocab_size=64)

        resp = client.post(
            "/api/bench/generate",
            json={
                "graph_ids": ["bench-a"],
                "prompt_ids": [1, 2, 3],
                "max_new_tokens": 4,
            },
        )
        assert resp.status_code == 200
        events = _parse_sse(resp.text)

        starts = [e for e in events if e["event"] == "graph_start"]
        assert len(starts) == 1
        assert starts[0]["data"]["graph_id"] == "bench-a"

        tokens = [e for e in events if e["event"] == "token"]
        assert len(tokens) == 4
        assert all(t["data"]["graph_id"] == "bench-a" for t in tokens)
        assert all("bench" in t["data"] for t in tokens)

        dones = [e for e in events if e["event"] == "done"]
        assert len(dones) >= 1
        assert "env" in dones[-1]["data"]

    def test_unknown_graph_returns_error(self):
        resp = client.post(
            "/api/bench/generate",
            json={
                "graph_ids": ["nonexistent"],
                "prompt_ids": [1, 2, 3],
            },
        )
        events = _parse_sse(resp.text)
        errors = [e for e in events if e["event"] == "error"]
        assert len(errors) == 1
        assert "not found" in errors[0]["data"]["error"]

    def test_multiple_graphs(self):
        _save_and_compile("v1", n_layer=1, n_head=2, n_embd=32, block_size=16, vocab_size=64)
        _save_and_compile("v2", n_layer=2, n_head=2, n_embd=32, block_size=16, vocab_size=64)

        resp = client.post(
            "/api/bench/generate",
            json={
                "graph_ids": ["v1", "v2"],
                "prompt_ids": [1, 2, 3],
                "max_new_tokens": 4,
            },
        )
        events = _parse_sse(resp.text)

        starts = [e for e in events if e["event"] == "graph_start"]
        assert len(starts) == 2
        assert starts[0]["data"]["graph_id"] == "v1"
        assert starts[1]["data"]["graph_id"] == "v2"

        v1_tokens = [e for e in events if e["event"] == "token" and e["data"]["graph_id"] == "v1"]
        v2_tokens = [e for e in events if e["event"] == "token" and e["data"]["graph_id"] == "v2"]
        assert len(v1_tokens) == 4
        assert len(v2_tokens) == 4

        dones = [e for e in events if e["event"] == "done"]
        assert any("env" in d["data"] for d in dones)
