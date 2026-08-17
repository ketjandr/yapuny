import pytest
import torch
import torch.nn as nn

from kernels.fused_layernorm_residual import FusedLayerNormResidual, fused_layernorm_residual

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEVICE = "cuda"
B, T, C = 4, 256, 384


@pytest.fixture
def setup():
    x = torch.randn(B, T, C, device=DEVICE)
    y = torch.randn(B, T, C, device=DEVICE)
    ln = nn.LayerNorm(C).to(DEVICE)
    return x, y, ln


class TestCorrectness:
    def test_matches_pytorch(self, setup):
        x, y, ln = setup
        expected = ln(x + y)
        actual = fused_layernorm_residual(x, y, ln.weight, ln.bias)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_2d_input(self, setup):
        x, y, ln = setup
        x_2d = x.reshape(-1, C)
        y_2d = y.reshape(-1, C)
        expected = ln(x_2d + y_2d)
        actual = fused_layernorm_residual(x_2d, y_2d, ln.weight, ln.bias)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_module_wrapper(self, setup):
        x, y, ln = setup
        fused = FusedLayerNormResidual(C).to(DEVICE)
        # copy weights from reference
        fused.weight.data.copy_(ln.weight.data)
        fused.bias.data.copy_(ln.bias.data)
        expected = ln(x + y)
        actual = fused(x, y)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_zeros(self):
        x = torch.zeros(B, T, C, device=DEVICE)
        y = torch.zeros(B, T, C, device=DEVICE)
        ln = nn.LayerNorm(C).to(DEVICE)
        expected = ln(x + y)
        actual = fused_layernorm_residual(x, y, ln.weight, ln.bias)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


class TestBenchmark:
    @pytest.mark.parametrize("seq_len", [32, 128, 256])
    def test_fused(self, seq_len, benchmark):
        x = torch.randn(B, seq_len, C, device=DEVICE)
        y = torch.randn(B, seq_len, C, device=DEVICE)
        ln = nn.LayerNorm(C).to(DEVICE)

        def run_fused():
            torch.cuda.synchronize()
            out = fused_layernorm_residual(x, y, ln.weight, ln.bias)
            torch.cuda.synchronize()
            return out

        benchmark(run_fused)

    @pytest.mark.parametrize("seq_len", [32, 128, 256])
    def test_vanilla(self, seq_len, benchmark):
        x = torch.randn(B, seq_len, C, device=DEVICE)
        y = torch.randn(B, seq_len, C, device=DEVICE)
        ln = nn.LayerNorm(C).to(DEVICE)

        def run_vanilla():
            torch.cuda.synchronize()
            out = ln(x + y)
            torch.cuda.synchronize()
            return out

        benchmark(run_vanilla)
