import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from server.api.schemas import (
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
                nodes=[], edges=[], id="sneaky",
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

    def test_custom(self):
        r = TrainRequest(max_steps=100, batch_size=8)
        assert r.max_steps == 100
        assert r.batch_size == 8


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
        resp = client.post("/api/validate", json=graph)
        assert resp.status_code == 200
        body = resp.json()
        assert body["valid"] is False
        assert any("no fusion kernel matches pattern" in e for e in body["errors"])

    def test_fusion_with_kernel_field_rejected(self):
        graph = default_gpt_graph(n_layer=1)
        graph["fusion_groups"] = [
            {"nodes": ["b0_resid_drop1", "b0_res1"], "kernel": "FusedDropoutResidual"},
        ]
        resp = client.post("/api/validate", json=graph)
        assert resp.status_code == 422

    def test_valid_graph_passes(self):
        graph = default_gpt_graph(n_layer=1)
        resp = client.post("/api/validate", json=graph)
        assert resp.status_code == 200
        assert resp.json()["valid"] is True

    def test_unknown_node_in_fusion_returns_error(self):
        graph = default_gpt_graph(n_layer=1)
        graph["fusion_groups"] = [{"nodes": ["ghost", "b0_res1"]}]
        resp = client.post("/api/validate", json=graph)
        assert resp.status_code == 200
        body = resp.json()
        assert body["valid"] is False
        assert any("unknown node" in e for e in body["errors"])
