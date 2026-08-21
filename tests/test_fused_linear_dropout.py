import pytest
import torch
import torch.nn.functional as F

from kernels.fused_linear_dropout import FusedLinearDropout, fused_linear_dropout

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEVICE = "cuda"
B, T, C = 1, 256, 384
N = 1536
P_DROP = 0.1


@pytest.fixture
def setup():
    x = torch.randn(B, T, C, device=DEVICE)
    weight = torch.randn(N, C, device=DEVICE)
    bias = torch.randn(N, device=DEVICE)
    return x, weight, bias


def vanilla_linear_dropout(x, weight, bias, p_drop, training):
    return F.dropout(F.linear(x, weight, bias), p=p_drop, training=training)


class TestCorrectness:
    def test_eval_matches_pytorch(self, setup):
        """With training=False, dropout is identity so output is matmul+bias."""
        x, weight, bias = setup
        expected = vanilla_linear_dropout(x, weight, bias, P_DROP, training=False)
        actual = fused_linear_dropout(x, weight, bias, P_DROP, training=False)
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)

    def test_output_shape(self, setup):
        x, weight, bias = setup
        out = fused_linear_dropout(x, weight, bias, P_DROP, training=True)
        assert out.shape == (B, T, N)

    def test_2d_input(self, setup):
        x, weight, bias = setup
        x_2d = x.reshape(-1, C)
        expected = vanilla_linear_dropout(x_2d, weight, bias, P_DROP, training=False)
        actual = fused_linear_dropout(x_2d, weight, bias, P_DROP, training=False)
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)

    def test_train_some_elements_dropped(self, setup):
        x, weight, bias = setup
        out_train = fused_linear_dropout(x, weight, bias, p_drop=0.5, training=True)
        out_eval = fused_linear_dropout(x, weight, bias, p_drop=0.5, training=False)
        diff = (out_train - out_eval).abs()
        assert (diff > 1e-6).any(), "dropout should modify some elements"

    def test_train_drop_rate_approx(self, setup):
        """Check that ~p_drop fraction of outputs are zeroed."""
        x = torch.randn(1, 1, C, device=DEVICE)
        weight = torch.eye(C, device=DEVICE)  # identity matmul
        bias = torch.zeros(C, device=DEVICE)
        out = fused_linear_dropout(x, weight, bias, p_drop=P_DROP, training=True)
        n_zero = (out.abs() < 1e-6).sum().item()
        drop_rate = n_zero / out.numel()
        assert 0.05 < drop_rate < 0.15, f"expected ~10% drop rate, got {drop_rate:.2%}"

    def test_train_scaling(self, setup):
        """Kept values should be scaled by 1/(1-p) relative to eval output."""
        x = torch.ones(1, 1, C, device=DEVICE)
        weight = torch.eye(C, device=DEVICE)
        bias = torch.zeros(C, device=DEVICE)
        out = fused_linear_dropout(x, weight, bias, p_drop=P_DROP, training=True)
        # with identity matmul and ones input, eval output = 1.0
        kept = out[out.abs() > 1e-6]
        expected_scale = 1.0 / (1.0 - P_DROP)
        torch.testing.assert_close(
            kept, torch.full_like(kept, expected_scale), atol=1e-5, rtol=1e-5
        )

    def test_module_wrapper(self, setup):
        x, _, _ = setup
        module = FusedLinearDropout(C, N, P_DROP).to(DEVICE)
        module.eval()
        expected = F.linear(x, module.weight, module.bias)
        actual = module(x)
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)


class TestBenchmark:
    @pytest.mark.parametrize("seq_len", [32, 128, 256])
    def test_fused(self, seq_len, benchmark):
        x = torch.randn(B, seq_len, C, device=DEVICE)
        weight = torch.randn(N, C, device=DEVICE)
        bias = torch.randn(N, device=DEVICE)

        def run_fused():
            torch.cuda.synchronize()
            out = fused_linear_dropout(x, weight, bias, P_DROP, training=True)
            torch.cuda.synchronize()
            return out

        benchmark(run_fused)

    @pytest.mark.parametrize("seq_len", [32, 128, 256])
    def test_vanilla(self, seq_len, benchmark):
        x = torch.randn(B, seq_len, C, device=DEVICE)
        weight = torch.randn(N, C, device=DEVICE)
        bias = torch.randn(N, device=DEVICE)

        def run_vanilla():
            torch.cuda.synchronize()
            out = vanilla_linear_dropout(x, weight, bias, P_DROP, training=True)
            torch.cuda.synchronize()
            return out

        benchmark(run_vanilla)
