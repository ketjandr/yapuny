import pytest
import torch

from server.compiler.compiler import GraphCompiler
from server.compiler.fusion_registry import FUSION_AVAILABLE
from server.compiler.quantization_registry import QUANTIZATION_AVAILABLE
from server.compiler.utils import graph_structure_hash
from server.models.graph import GraphSpec
from tests.server.graph_factory import default_gpt_graph

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required",
)
requires_fusion = pytest.mark.skipif(
    not FUSION_AVAILABLE,
    reason="fusion kernels not available",
)
requires_quantization = pytest.mark.skipif(
    not QUANTIZATION_AVAILABLE,
    reason="Triton/CUDA not available",
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
                model_fresh.state_dict()[key],
                model_loaded.state_dict()[key],
            )

    @requires_fusion
    def test_pretrained_with_different_fusion(self, compiler, dummy_input):
        unfused_graph = GraphSpec.from_dict(default_gpt_graph(**TINY))
        unfused_model = compiler.compile(unfused_graph).to(DEVICE)
        unfused_model.eval()
        state = unfused_model.state_dict()

        with torch.no_grad():
            baseline, _, _ = unfused_model(dummy_input)

        g1 = default_gpt_graph(**TINY)
        g1["fusion_groups"] = [{"nodes": ["b0_resid_drop1", "b0_res1"]}]
        fused1 = compiler.compile(GraphSpec.from_dict(g1), pretrained_state=state)
        fused1.to(DEVICE).eval()

        g2 = default_gpt_graph(**TINY)
        g2["fusion_groups"] = [{"nodes": ["b0_resid_drop2", "b0_res2"]}]
        fused2 = compiler.compile(GraphSpec.from_dict(g2), pretrained_state=state)
        fused2.to(DEVICE).eval()

        with torch.no_grad():
            out1, _, _ = fused1(dummy_input)
            out2, _, _ = fused2(dummy_input)

        torch.testing.assert_close(baseline, out1, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(baseline, out2, atol=1e-3, rtol=1e-3)

    @requires_fusion
    def test_three_node_fusion_rejected_when_mid_has_consumers(self, compiler):
        g = default_gpt_graph(**TINY)
        g["fusion_groups"] = [
            {"nodes": ["b0_resid_drop1", "b0_res1", "b0_ln2"]},
        ]
        with pytest.raises(ValueError, match="mid-chain node"):
            compiler.compile(GraphSpec.from_dict(g))


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


# -- quantization compilation --


def _quantize_nodes(graph_dict: dict, node_ids: list[str], mode: str = "w8") -> dict:
    for node in graph_dict["nodes"]:
        if node["id"] in node_ids:
            node["quantized"] = mode
    return graph_dict


class TestQuantizationCompiler:
    def test_rejects_without_cuda(self):
        if QUANTIZATION_AVAILABLE:
            pytest.skip("Triton available, cannot test rejection")
        g = default_gpt_graph(**TINY)
        _quantize_nodes(g, ["b0_mlp_up"], "w8")
        compiler = GraphCompiler()
        with pytest.raises(ValueError, match="CUDA GPU"):
            compiler.compile(GraphSpec.from_dict(g))

    @requires_quantization
    def test_w8_compiles_and_runs(self, compiler):
        graph = GraphSpec.from_dict(default_gpt_graph(**TINY))
        model = compiler.compile(graph)
        model.eval()
        pretrained = model.state_dict()

        g = default_gpt_graph(**TINY)
        _quantize_nodes(g, ["b0_mlp_up", "b0_mlp_down"], "w8")
        q_model = compiler.compile(GraphSpec.from_dict(g), pretrained_state=pretrained)
        q_model.eval().cuda()

        idx = torch.randint(0, TINY["vocab_size"], (1, 4), device="cuda")
        logits, _, _ = q_model(idx)
        assert logits.shape == (1, 4, TINY["vocab_size"])

    @requires_quantization
    def test_w4_compiles_and_runs(self, compiler):
        graph = GraphSpec.from_dict(default_gpt_graph(**TINY))
        model = compiler.compile(graph)
        model.eval()
        pretrained = model.state_dict()

        g = default_gpt_graph(**TINY)
        _quantize_nodes(g, ["b0_mlp_up"], "w4")
        q_model = compiler.compile(GraphSpec.from_dict(g), pretrained_state=pretrained)
        q_model.eval().cuda()

        idx = torch.randint(0, TINY["vocab_size"], (1, 4), device="cuda")
        logits, _, _ = q_model(idx)
        assert logits.shape == (1, 4, TINY["vocab_size"])

    @requires_quantization
    def test_quantized_output_close_to_fp32(self, compiler):
        graph = GraphSpec.from_dict(default_gpt_graph(**TINY))
        model = compiler.compile(graph)
        model.eval().cuda()
        pretrained = model.state_dict()

        g = default_gpt_graph(**TINY)
        _quantize_nodes(g, ["b0_mlp_up"], "w8")
        q_model = compiler.compile(GraphSpec.from_dict(g), pretrained_state=pretrained)
        q_model.eval().cuda()

        idx = torch.randint(0, TINY["vocab_size"], (1, 4), device="cuda")
        with torch.no_grad():
            fp32_logits, _, _ = model(idx)
            q_logits, _, _ = q_model(idx)

        assert torch.allclose(fp32_logits, q_logits, atol=0.5)

    @requires_quantization
    def test_all_linear_nodes_quantized(self, compiler):
        graph = GraphSpec.from_dict(default_gpt_graph(**TINY))
        model = compiler.compile(graph)
        model.eval()
        pretrained = model.state_dict()

        g = default_gpt_graph(**TINY)
        _quantize_nodes(g, ["b0_qkv", "b0_out_proj", "b0_mlp_up", "b0_mlp_down", "lm_head"], "w8")
        q_model = compiler.compile(GraphSpec.from_dict(g), pretrained_state=pretrained)
        q_model.eval().cuda()

        idx = torch.randint(0, TINY["vocab_size"], (1, 4), device="cuda")
        logits, _, _ = q_model(idx)
        assert logits.shape == (1, 4, TINY["vocab_size"])

    @requires_quantization
    def test_quantized_reduces_weight_bytes(self, compiler):
        graph = GraphSpec.from_dict(default_gpt_graph(**TINY))
        model = compiler.compile(graph)
        fp32_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

        g = default_gpt_graph(**TINY)
        _quantize_nodes(g, ["b0_qkv", "b0_out_proj", "b0_mlp_up", "b0_mlp_down", "lm_head"], "w8")
        q_model = compiler.compile(
            GraphSpec.from_dict(g),
            pretrained_state=model.state_dict(),
        )

        q_bytes = sum(p.numel() * p.element_size() for p in q_model.parameters())
        q_buf_bytes = sum(b.numel() * b.element_size() for b in q_model.buffers())
        total_q = q_bytes + q_buf_bytes

        assert total_q < fp32_bytes

    @requires_quantization
    def test_train_rejects_quantized_model(self, compiler):
        from worker.worker import Worker

        graph = GraphSpec.from_dict(default_gpt_graph(**TINY))
        model = compiler.compile(graph)
        model.eval()
        pretrained = model.state_dict()

        g_dict = default_gpt_graph(**TINY)
        _quantize_nodes(g_dict, ["b0_mlp_up"], "w8")
        q_graph = GraphSpec.from_dict(g_dict)
        q_model = compiler.compile(q_graph, pretrained_state=pretrained)
        q_model.eval()

        w = Worker.__new__(Worker)
        w.model = q_model
        w.graph = q_graph
        w.device = torch.device("cpu")
        w.training = False
        w.train_state = None
        w._weight_store = {}
        w._structure_hash = None

        result = w.train(max_steps=10)
        assert "error" in result
        assert "quantized" in result["error"]
