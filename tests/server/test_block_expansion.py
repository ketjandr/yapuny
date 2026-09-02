import pytest
import torch

from server.compiler.compiler import GraphCompiler
from server.compiler.utils import expand_blocks, graph_structure_hash, logical_node_id
from server.models.graph import GraphSpec
from tests.server.graph_factory import blocked_gpt_graph, default_gpt_graph

BLOCK_NODE_COUNT = 17  # nodes in one transformer block (make_block_nodes)
PROLOGUE_EPILOGUE = 6  # tok/pos emb, emb_add, emb_drop + ln_f, lm_head


def test_expand_unrolls_to_n_layer():
    expanded = expand_blocks(GraphSpec.from_dict(blocked_gpt_graph(3)))
    assert sum(1 for n in expanded.nodes if n.type == "qkv_proj") == 3
    assert len(expanded.nodes) == PROLOGUE_EPILOGUE + 3 * BLOCK_NODE_COUNT
    assert expanded.block is None  # consumed


def test_no_block_is_noop():
    g = GraphSpec.from_dict(default_gpt_graph(n_layer=2))
    assert expand_blocks(g) is g


def test_expanded_matches_explicit_stack():
    # one block + spec(n_layer=2) must compile to the same arch as a hand-authored 2-block graph
    logical = GraphCompiler().compile(GraphSpec.from_dict(blocked_gpt_graph(2)))
    explicit = GraphCompiler().compile(GraphSpec.from_dict(default_gpt_graph(n_layer=2)))
    assert sum(p.numel() for p in logical.parameters()) == sum(
        p.numel() for p in explicit.parameters()
    )


def test_expanded_forward_runs():
    model = GraphCompiler().compile(GraphSpec.from_dict(blocked_gpt_graph(2)))
    logits, _, _ = model(torch.randint(0, 8000, (2, 16)))
    assert logits.shape == (2, 16, 8000)


def test_n_layer_changes_structure_hash():
    h1 = graph_structure_hash(GraphSpec.from_dict(blocked_gpt_graph(2)))
    h2 = graph_structure_hash(GraphSpec.from_dict(blocked_gpt_graph(4)))
    assert h1 != h2


def test_multi_output_block_rejected():
    # a block whose nodes emit two distinct output tensors is not a single-exit slice
    g = blocked_gpt_graph(2)
    g["block"]["nodes"] = ["b0_ln1", "b0_qkv"]  # qkv emits q,k,v to different downstream nodes
    with pytest.raises(ValueError):
        expand_blocks(GraphSpec.from_dict(g))


def test_logical_node_id_strips_layer_prefix():
    assert logical_node_id("l0_mlp_up") == "mlp_up"
    assert logical_node_id("l12_b0_qkv") == "b0_qkv"
    assert logical_node_id("emb_drop") == "emb_drop"  # non-block node unchanged
    assert logical_node_id("_fused_l0_mlp_up_l0_mlp_act") == "_fused_l0_mlp_up_l0_mlp_act"


def test_block_survives_schema_roundtrip():
    # the block field must survive the Pydantic layer (GraphRequest -> model_dump -> GraphSpec)
    from server.api.schemas import GraphRequest

    graph = GraphSpec.from_dict(GraphRequest(**blocked_gpt_graph(2)).model_dump())
    assert graph.block is not None and graph.meta.n_layer == 2
    assert sum(1 for n in expand_blocks(graph).nodes if n.type == "qkv_proj") == 2


def test_block_weights_reload_across_recompile():
    # unrolling is deterministic, so weights saved from one compile reload into a fresh
    # recompile of the same block graph (the l{L}_* ids match) - never silently reinitialized
    graph = GraphSpec.from_dict(blocked_gpt_graph(2))
    compiler = GraphCompiler()
    trained = compiler.compile(graph)
    reloaded = compiler.compile(graph, pretrained_state=trained.state_dict())
    reloaded_state = reloaded.state_dict()
    assert reloaded_state.keys() == trained.state_dict().keys()
    for key, value in trained.state_dict().items():
        assert torch.equal(value, reloaded_state[key])
