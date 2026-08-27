import pytest
import torch
import torch.nn.functional as F

from kernels.fusion.fused_linear_dropout_residual import (
    FusedLinearDropoutResidual,
    fused_linear_dropout_residual,
)

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEVICE = "cuda"
B, T, C = 1, 256, 384
N = 384  # output projects back to C (like MLP linear2)
P_DROP = 0.1


@pytest.fixture
def setup():
    x = torch.randn(B, T, C, device=DEVICE)
    weight = torch.randn(N, C, device=DEVICE)
    bias = torch.randn(N, device=DEVICE)
    x_skip = torch.randn(B, T, N, device=DEVICE)
    return x, weight, bias, x_skip


def vanilla_linear_dropout_residual(x, weight, bias, x_skip, p_drop, training):
    return x_skip + F.dropout(F.linear(x, weight, bias), p=p_drop, training=training)


class TestCorrectness:
    def test_eval_matches_pytorch(self, setup):
        """With training=False, dropout is identity so output is exact."""
        x, weight, bias, x_skip = setup
        expected = vanilla_linear_dropout_residual(x, weight, bias, x_skip, P_DROP, training=False)
        actual = fused_linear_dropout_residual(x, weight, bias, x_skip, P_DROP, training=False)
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)

    def test_output_shape(self, setup):
        x, weight, bias, x_skip = setup
        out = fused_linear_dropout_residual(x, weight, bias, x_skip, P_DROP, training=True)
        assert out.shape == (B, T, N)

    def test_2d_input(self, setup):
        x, weight, bias, x_skip = setup
        x_2d = x.reshape(-1, C)
        skip_2d = x_skip.reshape(-1, N)
        expected = vanilla_linear_dropout_residual(
            x_2d, weight, bias, skip_2d, P_DROP, training=False
        )
        actual = fused_linear_dropout_residual(x_2d, weight, bias, skip_2d, P_DROP, training=False)
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)

    def test_train_some_elements_differ(self, setup):
        x, weight, bias, x_skip = setup
        out_train = fused_linear_dropout_residual(
            x, weight, bias, x_skip, p_drop=0.5, training=True
        )
        out_eval = fused_linear_dropout_residual(
            x, weight, bias, x_skip, p_drop=0.5, training=False
        )
        diff = (out_train - out_eval).abs()
        assert (diff > 1e-3).any(), "dropout should modify some elements"

    def test_train_drop_rate_approx(self):
        """Check that ~p_drop fraction of linear outputs are zeroed before residual add."""
        x = torch.ones(1, 1, C, device=DEVICE)
        weight = torch.eye(C, device=DEVICE)  # identity matmul (N=C)
        bias = torch.zeros(C, device=DEVICE)
        x_skip = torch.zeros(1, 1, C, device=DEVICE)
        out = fused_linear_dropout_residual(x, weight, bias, x_skip, p_drop=P_DROP, training=True)
        # x_skip=0, identity matmul, input=1 → out = dropout(1)
        n_zero = (out.abs() < 1e-6).sum().item()
        drop_rate = n_zero / out.numel()
        assert 0.05 < drop_rate < 0.15, f"expected ~10% drop rate, got {drop_rate:.2%}"

    def test_module_wrapper(self, setup):
        x, _, _, x_skip = setup
        module = FusedLinearDropoutResidual(C, N, P_DROP).to(DEVICE)
        module.eval()
        expected = x_skip + F.linear(x, module.weight, module.bias)
        actual = module(x, x_skip)
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)


class TestBenchmark:
    @pytest.mark.parametrize("seq_len", [32, 128, 256])
    def test_fused(self, seq_len, benchmark):
        x = torch.randn(B, seq_len, C, device=DEVICE)
        weight = torch.randn(N, C, device=DEVICE)
        bias = torch.randn(N, device=DEVICE)
        x_skip = torch.randn(B, seq_len, N, device=DEVICE)

        def run_fused():
            torch.cuda.synchronize()
            out = fused_linear_dropout_residual(x, weight, bias, x_skip, P_DROP, training=True)
            torch.cuda.synchronize()
            return out

        benchmark(run_fused)

    @pytest.mark.parametrize("seq_len", [32, 128, 256])
    def test_vanilla(self, seq_len, benchmark):
        x = torch.randn(B, seq_len, C, device=DEVICE)
        weight = torch.randn(N, C, device=DEVICE)
        bias = torch.randn(N, device=DEVICE)
        x_skip = torch.randn(B, seq_len, N, device=DEVICE)

        def run_vanilla():
            torch.cuda.synchronize()
            out = vanilla_linear_dropout_residual(x, weight, bias, x_skip, P_DROP, training=True)
            torch.cuda.synchronize()
            return out

        benchmark(run_vanilla)
