import pytest
import torch

from server.compiler.compiler import GraphCompiler
from server.compiler.fusion_registry import FUSION_AVAILABLE
from server.compiler.utils import graph_structure_hash
from server.models.graph import GraphSpec
from tests.server.graph_factory import default_gpt_graph

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required",
)
requires_fusion = pytest.mark.skipif(
    not FUSION_AVAILABLE, reason="fusion kernels not available",
)

DEVICE = "cuda"
TINY = dict(n_layer=1, n_head=2, n_embd=32, block_size=16, vocab_size=64)


@pytest.fixture
def compiler():
    return GraphCompiler()


@pytest.fixture
def graph():
    return GraphSpec.from_dict(default_gpt_graph(**TINY))


@pytest.fixture
def dummy_input():
    return torch.randint(0, 64, (1, 8), device=DEVICE)


@requires_cuda
class TestCompilation:
    def test_compiles_valid_graph(self, compiler, graph):
        model = compiler.compile(graph)
        assert model is not None

    def test_invalid_graph_raises(self, compiler):
        bad = default_gpt_graph(**TINY)
        bad["nodes"] = [n for n in bad["nodes"] if n["type"] != "lm_head"]
        graph = GraphSpec.from_dict(bad)
        with pytest.raises(ValueError, match="invalid graph"):
            compiler.compile(graph)

    def test_output_shape(self, compiler, graph, dummy_input):
        model = compiler.compile(graph).to(DEVICE)
        model.eval()
        logits, loss, _ = model(dummy_input)
        assert logits.shape == (1, 8, 64)
        assert loss is None

    def test_output_with_targets(self, compiler, graph, dummy_input):
        model = compiler.compile(graph).to(DEVICE)
        model.eval()
        targets = torch.randint(0, 64, (1, 8), device=DEVICE)
        logits, loss, _ = model(dummy_input, targets)
        assert loss is not None
        assert loss.item() > 0


@requires_cuda
@requires_fusion
class TestFusionCompilation:
    def test_fused_compiles(self, compiler):
        g = default_gpt_graph(**TINY)
        g["fusion_groups"] = [{"nodes": ["b0_resid_drop1", "b0_res1"]}]
        graph = GraphSpec.from_dict(g)
        model = compiler.compile(graph)
        assert model is not None

    def test_fused_output_matches_unfused(self, compiler, dummy_input):
        unfused_graph = GraphSpec.from_dict(default_gpt_graph(**TINY))
        unfused_model = compiler.compile(unfused_graph).to(DEVICE)
        unfused_model.eval()

        with torch.no_grad():
            unfused_out, _, _ = unfused_model(dummy_input)

        state = unfused_model.state_dict()

        g = default_gpt_graph(**TINY)
        g["fusion_groups"] = [{"nodes": ["b0_resid_drop1", "b0_res1"]}]
        fused_graph = GraphSpec.from_dict(g)
        fused_model = compiler.compile(fused_graph, pretrained_state=state).to(DEVICE)
        fused_model.eval()

        with torch.no_grad():
            fused_out, _, _ = fused_model(dummy_input)

        torch.testing.assert_close(unfused_out, fused_out, atol=1e-3, rtol=1e-3)


@requires_cuda
class TestPretrainedState:
    def test_loads_pretrained_weights(self, compiler, graph, dummy_input):
        model1 = compiler.compile(graph).to(DEVICE)
        model1.eval()
        state = model1.state_dict()

        with torch.no_grad():
            out1, _, _ = model1(dummy_input)

        model2 = compiler.compile(graph, pretrained_state=state).to(DEVICE)
        model2.eval()

        with torch.no_grad():
            out2, _, _ = model2(dummy_input)

        torch.testing.assert_close(out1, out2)

    def test_pretrained_skips_init(self, compiler, graph):
        model_fresh = compiler.compile(graph)
        state = model_fresh.state_dict()
        model_loaded = compiler.compile(graph, pretrained_state=state)

        for key in state:
            torch.testing.assert_close(
                model_fresh.state_dict()[key], model_loaded.state_dict()[key],
            )

    @requires_fusion
    def test_pretrained_with_different_fusion(self, compiler):
        # larger n_embd to avoid numerical divergence in fused layernorm reduction
        cfg = dict(n_layer=1, n_head=2, n_embd=128, block_size=16, vocab_size=64)
        inp = torch.randint(0, 64, (1, 8), device=DEVICE)

        unfused_graph = GraphSpec.from_dict(default_gpt_graph(**cfg))
        unfused_model = compiler.compile(unfused_graph).to(DEVICE)
        unfused_model.eval()
        state = unfused_model.state_dict()

        with torch.no_grad():
            baseline, _, _ = unfused_model(inp)

        g1 = default_gpt_graph(**cfg)
        g1["fusion_groups"] = [{"nodes": ["b0_resid_drop1", "b0_res1"]}]
        fused1 = compiler.compile(GraphSpec.from_dict(g1), pretrained_state=state)
        fused1.to(DEVICE).eval()

        g2 = default_gpt_graph(**cfg)
        g2["fusion_groups"] = [
            {"nodes": ["b0_resid_drop1", "b0_res1", "b0_ln2"]},
        ]
        fused2 = compiler.compile(GraphSpec.from_dict(g2), pretrained_state=state)
        fused2.to(DEVICE).eval()

        with torch.no_grad():
            out1, _, _ = fused1(inp)
            out2, _, _ = fused2(inp)

        torch.testing.assert_close(baseline, out1, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(baseline, out2, atol=1e-3, rtol=1e-3)


class TestStructureHash:
    def test_same_graph_same_hash(self):
        g1 = GraphSpec.from_dict(default_gpt_graph(**TINY))
        g2 = GraphSpec.from_dict(default_gpt_graph(**TINY))
        assert graph_structure_hash(g1) == graph_structure_hash(g2)

    def test_different_meta_different_hash(self):
        g1 = GraphSpec.from_dict(default_gpt_graph(**TINY))
        modified = dict(TINY, n_embd=64)
        g2 = GraphSpec.from_dict(default_gpt_graph(**modified))
        assert graph_structure_hash(g1) != graph_structure_hash(g2)

    def test_fusion_groups_excluded(self):
        g1 = GraphSpec.from_dict(default_gpt_graph(**TINY))
        d2 = default_gpt_graph(**TINY)
        d2["fusion_groups"] = [{"nodes": ["b0_resid_drop1", "b0_res1"]}]
        g2 = GraphSpec.from_dict(d2)
        assert graph_structure_hash(g1) == graph_structure_hash(g2)
