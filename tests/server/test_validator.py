import pytest

from server.compiler.fusion_registry import FUSION_AVAILABLE
from server.compiler.validator import GraphValidator
from server.models.graph import GraphSpec
from tests.server.graph_factory import default_gpt_graph

requires_fusion = pytest.mark.skipif(
    not FUSION_AVAILABLE,
    reason="fusion kernels not available",
)


@pytest.fixture
def validator():
    return GraphValidator()


@pytest.fixture
def valid_graph():
    return GraphSpec.from_dict(default_gpt_graph(n_layer=1))


@pytest.fixture
def valid_graph_dict():
    return default_gpt_graph(n_layer=1)


# -- basic validation --


class TestBasicValidation:
    def test_valid_graph_passes(self, validator, valid_graph):
        result = validator.validate(valid_graph)
        assert result.valid
        assert result.errors == []

    def test_unknown_node_type(self, validator, valid_graph_dict):
        valid_graph_dict["nodes"].append({"id": "bad", "type": "nonexistent_type"})
        graph = GraphSpec.from_dict(valid_graph_dict)
        result = validator.validate(graph)
        assert not result.valid
        assert any("unknown node type" in e for e in result.errors)

    def test_missing_required_node(self, validator, valid_graph_dict):
        valid_graph_dict["nodes"] = [n for n in valid_graph_dict["nodes"] if n["type"] != "lm_head"]
        graph = GraphSpec.from_dict(valid_graph_dict)
        result = validator.validate(graph)
        assert not result.valid
        assert any("missing required node: lm_head" in e for e in result.errors)

    def test_dangling_edge(self, validator, valid_graph_dict):
        valid_graph_dict["edges"].append(
            {
                "from_node": "ghost_node",
                "to_node": "lm_head",
            }
        )
        graph = GraphSpec.from_dict(valid_graph_dict)
        result = validator.validate(graph)
        assert not result.valid
        assert any("ghost_node" in e for e in result.errors)

    def test_cycle_detected(self, validator, valid_graph_dict):
        valid_graph_dict["edges"].append(
            {
                "from_node": "lm_head",
                "to_node": "ln_f",
            }
        )
        graph = GraphSpec.from_dict(valid_graph_dict)
        result = validator.validate(graph)
        assert not result.valid
        assert any("cycle" in e for e in result.errors)

    def test_invalid_port(self, validator, valid_graph_dict):
        valid_graph_dict["edges"].append(
            {
                "from_node": "ln_f",
                "from_port": "nonexistent",
                "to_node": "lm_head",
                "to_port": "x",
            }
        )
        graph = GraphSpec.from_dict(valid_graph_dict)
        result = validator.validate(graph)
        assert not result.valid
        assert any("no output port" in e for e in result.errors)


# -- fusion validation --


@requires_fusion
class TestFusionValidation:
    def test_valid_fusion_passes(self, validator, valid_graph_dict):
        valid_graph_dict["fusion_groups"] = [
            {"nodes": ["b0_resid_drop1", "b0_res1"]},
        ]
        graph = GraphSpec.from_dict(valid_graph_dict)
        result = validator.validate(graph)
        assert result.valid

    def test_no_matching_pattern(self, validator, valid_graph_dict):
        valid_graph_dict["fusion_groups"] = [
            {"nodes": ["b0_ln1", "b0_resid_drop1"]},
        ]
        graph = GraphSpec.from_dict(valid_graph_dict)
        result = validator.validate(graph)
        assert not result.valid
        assert any("no fusion kernel matches pattern" in e for e in result.errors)

    def test_unknown_node_in_fusion(self, validator, valid_graph_dict):
        valid_graph_dict["fusion_groups"] = [
            {"nodes": ["ghost_node", "b0_res1"]},
        ]
        graph = GraphSpec.from_dict(valid_graph_dict)
        result = validator.validate(graph)
        assert not result.valid
        assert any("unknown node" in e for e in result.errors)

    def test_overlapping_fusion_groups(self, validator, valid_graph_dict):
        valid_graph_dict["fusion_groups"] = [
            {"nodes": ["b0_resid_drop1", "b0_res1"]},
            {"nodes": ["b0_res1", "b0_ln2"]},
        ]
        graph = GraphSpec.from_dict(valid_graph_dict)
        result = validator.validate(graph)
        assert not result.valid
        assert any("already in another fusion group" in e for e in result.errors)

    def test_unconnected_chain(self, validator):
        g = default_gpt_graph(n_layer=2)
        g["fusion_groups"] = [
            {"nodes": ["b0_resid_drop1", "b1_res1"]},
        ]
        graph = GraphSpec.from_dict(g)
        result = validator.validate(graph)
        assert not result.valid
        assert any("not connected" in e for e in result.errors)

    def test_three_node_fusion_with_external_consumer(self, validator, valid_graph_dict):
        valid_graph_dict["fusion_groups"] = [
            {"nodes": ["b0_resid_drop1", "b0_res1", "b0_ln2"]},
        ]
        graph = GraphSpec.from_dict(valid_graph_dict)
        result = validator.validate(graph)
        assert not result.valid
        assert any("mid-chain node" in e and "b0_res1" in e for e in result.errors)

    def test_multiple_valid_groups(self, validator):
        g = default_gpt_graph(n_layer=2)
        g["fusion_groups"] = [
            {"nodes": ["b0_resid_drop1", "b0_res1"]},
            {"nodes": ["b1_resid_drop1", "b1_res1"]},
        ]
        graph = GraphSpec.from_dict(g)
        result = validator.validate(graph)
        assert result.valid


# -- fusion resolution --


@requires_fusion
class TestFusionResolution:
    def test_resolves_dropout_residual(self, validator, valid_graph_dict):
        valid_graph_dict["fusion_groups"] = [
            {"nodes": ["b0_resid_drop1", "b0_res1"]},
        ]
        graph = GraphSpec.from_dict(valid_graph_dict)
        resolved = validator.resolve_fusions(graph)
        assert len(resolved) == 1
        nodes, fdef = resolved[0]
        assert nodes == ["b0_resid_drop1", "b0_res1"]
        assert fdef.cls.__name__ == "FusedDropoutResidual"

    def test_resolves_three_node_pattern(self, validator, valid_graph_dict):
        valid_graph_dict["fusion_groups"] = [
            {"nodes": ["b0_resid_drop1", "b0_res1", "b0_ln2"]},
        ]
        graph = GraphSpec.from_dict(valid_graph_dict)
        resolved = validator.resolve_fusions(graph)
        assert len(resolved) == 1
        _, fdef = resolved[0]
        assert fdef.cls.__name__ == "FusedDropoutResidualLayerNorm"

    def test_no_groups_returns_empty(self, validator, valid_graph):
        resolved = validator.resolve_fusions(valid_graph)
        assert resolved == []

    def test_unmatched_pattern_skipped(self, validator, valid_graph_dict):
        valid_graph_dict["fusion_groups"] = [
            {"nodes": ["b0_ln1", "b0_resid_drop1"]},
        ]
        graph = GraphSpec.from_dict(valid_graph_dict)
        resolved = validator.resolve_fusions(graph)
        assert resolved == []


# -- quantization validation --


TINY = dict(n_layer=1, n_head=2, n_embd=32, block_size=16, vocab_size=64)


def _quantize_nodes(graph_dict: dict, node_ids: list[str], mode: str = "w8") -> dict:
    for node in graph_dict["nodes"]:
        if node["id"] in node_ids:
            node["quantized"] = mode
    return graph_dict


class TestQuantizationValidation:
    def test_valid_w8_on_mlp_up(self, validator):
        g = default_gpt_graph(**TINY)
        _quantize_nodes(g, ["b0_mlp_up"], "w8")
        result = validator.validate(GraphSpec.from_dict(g))
        assert result.valid

    def test_valid_w4_on_lm_head(self, validator):
        g = default_gpt_graph(**TINY)
        _quantize_nodes(g, ["lm_head"], "w4")
        result = validator.validate(GraphSpec.from_dict(g))
        assert result.valid

    def test_invalid_mode_rejected(self, validator):
        g = default_gpt_graph(**TINY)
        _quantize_nodes(g, ["b0_mlp_up"], "w16")
        result = validator.validate(GraphSpec.from_dict(g))
        assert not result.valid
        assert any("invalid quantization mode" in e for e in result.errors)

    def test_non_quantizable_node_rejected(self, validator):
        g = default_gpt_graph(**TINY)
        _quantize_nodes(g, ["emb_drop"], "w8")
        result = validator.validate(GraphSpec.from_dict(g))
        assert not result.valid
        assert any("cannot be quantized" in e for e in result.errors)

    def test_quantized_plus_fusion_rejected(self, validator):
        g = default_gpt_graph(**TINY)
        _quantize_nodes(g, ["b0_resid_drop1"], "w8")
        g["fusion_groups"] = [{"nodes": ["b0_resid_drop1", "b0_res1"]}]
        result = validator.validate(GraphSpec.from_dict(g))
        assert not result.valid
        assert any("cannot be both quantized" in e for e in result.errors)

    def test_multiple_nodes_quantized(self, validator):
        g = default_gpt_graph(**TINY)
        _quantize_nodes(g, ["b0_qkv", "b0_out_proj", "b0_mlp_up", "b0_mlp_down", "lm_head"], "w8")
        result = validator.validate(GraphSpec.from_dict(g))
        assert result.valid


# -- warnings --


class TestWarnings:
    def test_no_causal_mask_warning(self, validator, valid_graph_dict):
        valid_graph_dict["nodes"] = [
            n for n in valid_graph_dict["nodes"] if n["type"] != "causal_mask"
        ]
        graph = GraphSpec.from_dict(valid_graph_dict)
        result = validator.validate(graph)
        assert any("causal mask" in w for w in result.warnings)

    def test_no_layernorm_warning(self, validator, valid_graph_dict):
        valid_graph_dict["nodes"] = [
            n for n in valid_graph_dict["nodes"] if n["type"] != "layernorm"
        ]
        graph = GraphSpec.from_dict(valid_graph_dict)
        result = validator.validate(graph)
        assert any("LayerNorm" in w for w in result.warnings)
