import pytest
import torch
import torch.nn.functional as F

from kernels.fusion.fused_dropout_residual import FusedDropoutResidual, fused_dropout_residual

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEVICE = "cuda"
B, T, C = 1, 256, 384
P_DROP = 0.1


@pytest.fixture
def setup():
    x = torch.randn(B, T, C, device=DEVICE)
    residual = torch.randn(B, T, C, device=DEVICE)
    return x, residual


def vanilla_dropout_residual(x, residual, p_drop, training):
    return x + F.dropout(residual, p=p_drop, training=training)


class TestCorrectness:
    def test_eval_matches_pytorch(self, setup):
        """With training=False, dropout is identity so output is exact."""
        x, residual = setup
        expected = vanilla_dropout_residual(x, residual, P_DROP, training=False)
        actual = fused_dropout_residual(x, residual, P_DROP, training=False)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_eval_is_plain_add(self, setup):
        x, residual = setup
        out = fused_dropout_residual(x, residual, P_DROP, training=False)
        torch.testing.assert_close(out, x + residual, atol=1e-6, rtol=1e-6)

    def test_train_output_shape(self, setup):
        x, residual = setup
        out = fused_dropout_residual(x, residual, P_DROP, training=True)
        assert out.shape == (B, T, C)

    def test_train_some_elements_dropped(self, setup):
        x, residual = setup
        # use p_drop=0.5 for clearer signal
        out = fused_dropout_residual(x, residual, p_drop=0.5, training=True)
        plain_add = x + residual
        # some elements should differ from plain add (dropout applied)
        diff = (out - plain_add).abs()
        assert (diff > 1e-6).any(), "dropout should modify some elements"

    def test_train_drop_rate_approx(self, setup):
        """Check that ~p_drop fraction of residual contributions are zeroed."""
        x = torch.zeros(1, 1, 100_000, device=DEVICE)
        residual = torch.ones(1, 1, 100_000, device=DEVICE)
        out = fused_dropout_residual(x, residual, p_drop=P_DROP, training=True)
        # with x=0: out = 0 + dropout(ones). Dropped positions = 0, kept = 1/(1-p)
        n_zero = (out.abs() < 1e-6).sum().item()
        drop_rate = n_zero / out.numel()
        assert 0.05 < drop_rate < 0.15, f"expected ~10% drop rate, got {drop_rate:.2%}"

    def test_train_scaling(self, setup):
        """Kept values should be scaled by 1/(1-p)."""
        x = torch.zeros(1, 1, 100_000, device=DEVICE)
        residual = torch.ones(1, 1, 100_000, device=DEVICE)
        out = fused_dropout_residual(x, residual, p_drop=P_DROP, training=True)
        kept = out[out.abs() > 1e-6]
        expected_scale = 1.0 / (1.0 - P_DROP)
        torch.testing.assert_close(
            kept, torch.full_like(kept, expected_scale), atol=1e-5, rtol=1e-5
        )

    def test_module_wrapper(self, setup):
        x, residual = setup
        module = FusedDropoutResidual(P_DROP).to(DEVICE)
        module.eval()
        expected = x + residual
        actual = module(x, residual)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


class TestBenchmark:
    @pytest.mark.parametrize("seq_len", [32, 128, 256])
    def test_fused(self, seq_len, benchmark):
        x = torch.randn(B, seq_len, C, device=DEVICE)
        residual = torch.randn(B, seq_len, C, device=DEVICE)

        def run_fused():
            torch.cuda.synchronize()
            out = fused_dropout_residual(x, residual, P_DROP, training=True)
            torch.cuda.synchronize()
            return out

        benchmark(run_fused)

    @pytest.mark.parametrize("seq_len", [32, 128, 256])
    def test_vanilla(self, seq_len, benchmark):
        x = torch.randn(B, seq_len, C, device=DEVICE)
        residual = torch.randn(B, seq_len, C, device=DEVICE)

        def run_vanilla():
            torch.cuda.synchronize()
            out = vanilla_dropout_residual(x, residual, P_DROP, training=True)
            torch.cuda.synchronize()
            return out

        benchmark(run_vanilla)
