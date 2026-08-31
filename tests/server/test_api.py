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
    ModelGraphRequest,
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
        r = GenerateRequest(id="m", prompt="hello world")
        assert r.max_new_tokens == 50
        assert r.temperature == 1.0
        assert r.bench is False

    def test_bench_flag(self):
        r = GenerateRequest(id="m", prompt="hello", bench=True)
        assert r.bench is True

    def test_missing_prompt(self):
        with pytest.raises(ValidationError):
            GenerateRequest(id="m")

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            GenerateRequest(id="m", prompt="hi", beam_width=5)


class TestDecodeRequest:
    def test_valid(self):
        r = DecodeRequest(id="m", token_ids=[1, 2, 3])
        assert r.token_ids == [1, 2, 3]

    def test_missing_token_ids(self):
        with pytest.raises(ValidationError):
            DecodeRequest(id="m")


class TestTrainRequest:
    def test_defaults(self):
        r = TrainRequest(id="m")
        assert r.max_steps == 2000
        assert r.batch_size == 32
        assert r.learning_rate == 3e-4
        assert r.bench is False

    def test_custom(self):
        r = TrainRequest(id="m", max_steps=100, batch_size=8)
        assert r.max_steps == 100
        assert r.batch_size == 8

    def test_bench_flag(self):
        r = TrainRequest(id="m", bench=True)
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
    def _dummy_graph(self, model_id="m"):
        return ModelGraphRequest(id=model_id, graph=GraphRequest(**default_gpt_graph(
            n_layer=1, n_head=2, n_embd=32, block_size=16, vocab_size=64,
        )))

    def test_defaults(self):
        r = BenchRunRequest(graphs=[self._dummy_graph()], prompt="hello")
        assert r.max_new_tokens == 50
        assert r.temperature == 1.0
        assert r.top_k is None

    def test_too_many_graphs_rejected(self):
        with pytest.raises(ValidationError):
            BenchRunRequest(graphs=[self._dummy_graph() for _ in range(6)], prompt="hi")

    def test_empty_graphs_rejected(self):
        with pytest.raises(ValidationError):
            BenchRunRequest(graphs=[], prompt="hi")

    def test_missing_prompt(self):
        with pytest.raises(ValidationError):
            BenchRunRequest(graphs=[self._dummy_graph()])

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            BenchRunRequest(graphs=[self._dummy_graph()], prompt="hi", gpu="A100")


# -- benchmark route tests --


def _compile(model_id="test-model", **kwargs):
    graph = default_gpt_graph(**kwargs)
    client.post("/api/graph/compile", json={"id": model_id, "graph": graph})
    return graph


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


# bench sits on top of trained models: seed the locker with a package (real state_dict
# + the committed tokenizer) so bench can load weights by id without actually training.
def _seed_package(model_id, graph_dict):
    from data.tokenizer import load_tokenizer
    from server.compiler.compiler import GraphCompiler
    from server.compiler.utils import graph_structure_hash
    from server.models.graph import GraphSpec
    from worker import store
    from worker.worker import TOKENIZER_PATH

    spec = GraphSpec.from_dict(graph_dict)
    model = GraphCompiler().compile(spec)
    tok = load_tokenizer(TOKENIZER_PATH)
    store.save(model_id, tok, model.state_dict(), graph_structure_hash(spec))


@pytest.fixture
def isolated_locker(tmp_path, monkeypatch):
    from worker import store

    monkeypatch.setattr(store, "MODELS_DIR", tmp_path)
    return tmp_path


class TestBenchRoutes:
    # vocab_size matches the committed tokenizer so encoded prompt ids are in range
    def test_streams_single_graph(self, isolated_locker):
        graph_a = default_gpt_graph(n_layer=1, n_head=2, n_embd=32, block_size=16, vocab_size=8000)
        _seed_package("m-a", graph_a)

        resp = client.post(
            "/api/bench/generate",
            json={
                "graphs": [{"id": "m-a", "graph": graph_a}],
                "prompt": "abc",
                "max_new_tokens": 4,
            },
        )
        assert resp.status_code == 200
        events = _parse_sse(resp.text)

        starts = [e for e in events if e["event"] == "graph_start"]
        assert len(starts) == 1
        assert starts[0]["data"]["graph_idx"] == 0

        tokens = [e for e in events if e["event"] == "token"]
        assert len(tokens) == 4
        assert all(t["data"]["graph_idx"] == 0 for t in tokens)
        assert all("bench" in t["data"] for t in tokens)

        dones = [e for e in events if e["event"] == "done"]
        assert len(dones) >= 1
        assert "env" in dones[-1]["data"]

    def test_multiple_graphs(self, isolated_locker):
        graph_v1 = default_gpt_graph(n_layer=1, n_head=2, n_embd=32, block_size=16, vocab_size=8000)
        graph_v2 = default_gpt_graph(n_layer=2, n_head=2, n_embd=32, block_size=16, vocab_size=8000)
        _seed_package("m-v1", graph_v1)
        _seed_package("m-v2", graph_v2)

        resp = client.post(
            "/api/bench/generate",
            json={
                "graphs": [{"id": "m-v1", "graph": graph_v1}, {"id": "m-v2", "graph": graph_v2}],
                "prompt": "abc",
                "max_new_tokens": 4,
            },
        )
        events = _parse_sse(resp.text)

        starts = [e for e in events if e["event"] == "graph_start"]
        assert len(starts) == 2
        assert starts[0]["data"]["graph_idx"] == 0
        assert starts[1]["data"]["graph_idx"] == 1

        g0_tokens = [e for e in events if e["event"] == "token" and e["data"]["graph_idx"] == 0]
        g1_tokens = [e for e in events if e["event"] == "token" and e["data"]["graph_idx"] == 1]
        assert len(g0_tokens) == 4
        assert len(g1_tokens) == 4

        dones = [e for e in events if e["event"] == "done"]
        assert any("env" in d["data"] for d in dones)

    def test_untrained_model_errors(self, isolated_locker):
        graph = default_gpt_graph(n_layer=1, n_head=2, n_embd=32, block_size=16, vocab_size=8000)

        resp = client.post(
            "/api/bench/generate",
            json={
                "graphs": [{"id": "never-trained", "graph": graph}],
                "prompt": "abc",
                "max_new_tokens": 4,
            },
        )
        events = _parse_sse(resp.text)
        errors = [e for e in events if e["event"] == "error"]
        assert any("not trained" in e["data"]["error"] for e in errors)


class TestProfileRoute:
    def test_profile_decode(self):
        _compile(model_id="prof", n_layer=1, n_head=2, n_embd=32, block_size=16, vocab_size=64)
        resp = client.post(
            "/api/bench/profile",
            json={"id": "prof", "mode": "decode", "prompt_tokens": 4, "new_tokens": 4, "warmup": 1},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["nodes"]) > 0
        assert body["total_us"] > 0
        assert "env" in body
        total_pct = sum(n["pct"] for n in body["nodes"])
        assert abs(total_pct - 100.0) < 0.1

    def test_profile_train(self):
        _compile(model_id="prof", n_layer=1, n_head=2, n_embd=32, block_size=16, vocab_size=64)
        resp = client.post(
            "/api/bench/profile",
            json={"id": "prof", "mode": "train", "warmup": 1},
        )
        assert resp.status_code == 200
        assert len(resp.json()["nodes"]) > 0

    def test_profile_uncompiled_model(self):
        resp = client.post("/api/bench/profile", json={"id": "ghost"})
        assert resp.status_code == 400
        assert "not compiled" in resp.json()["detail"]

    def test_profile_defaults(self):
        _compile(model_id="prof", n_layer=1, n_head=2, n_embd=32, block_size=16, vocab_size=64)
        resp = client.post("/api/bench/profile", json={"id": "prof"})
        assert resp.status_code == 200
        assert len(resp.json()["nodes"]) > 0
