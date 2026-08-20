import pytest
import torch
import torch.nn.functional as F

from kernels.fused_linear_gelu import FusedLinearGELU, fused_linear_gelu

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEVICE = "cuda"
B, T, C = 1, 256, 384
N = 1536  # out_features (4 * C, matching MLP fc1)


@pytest.fixture
def setup():
    x = torch.randn(B, T, C, device=DEVICE)
    weight = torch.randn(N, C, device=DEVICE)
    bias = torch.randn(N, device=DEVICE)
    return x, weight, bias


def vanilla_linear_gelu(x, weight, bias):
    return F.gelu(F.linear(x, weight, bias))


class TestCorrectness:
    def test_matches_pytorch(self, setup):
        x, weight, bias = setup
        expected = vanilla_linear_gelu(x, weight, bias)
        actual = fused_linear_gelu(x, weight, bias)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    def test_output_shape(self, setup):
        x, weight, bias = setup
        out = fused_linear_gelu(x, weight, bias)
        assert out.shape == (B, T, N)

    def test_2d_input(self, setup):
        x, weight, bias = setup
        x_2d = x.reshape(-1, C)
        expected = vanilla_linear_gelu(x_2d, weight, bias)
        actual = fused_linear_gelu(x_2d, weight, bias)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    def test_module_wrapper(self, setup):
        x, _, _ = setup
        module = FusedLinearGELU(C, N).to(DEVICE)
        expected = vanilla_linear_gelu(x, module.weight, module.bias)
        actual = module(x)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    def test_small_dims(self):
        x = torch.randn(1, 4, 16, device=DEVICE)
        weight = torch.randn(32, 16, device=DEVICE)
        bias = torch.randn(32, device=DEVICE)
        expected = vanilla_linear_gelu(x, weight, bias)
        actual = fused_linear_gelu(x, weight, bias)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


class TestBenchmark:
    @pytest.mark.parametrize("seq_len", [32, 128, 256])
    def test_fused(self, seq_len, benchmark):
        x = torch.randn(B, seq_len, C, device=DEVICE)
        weight = torch.randn(N, C, device=DEVICE)
        bias = torch.randn(N, device=DEVICE)

        def run_fused():
            torch.cuda.synchronize()
            out = fused_linear_gelu(x, weight, bias)
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
            out = vanilla_linear_gelu(x, weight, bias)
            torch.cuda.synchronize()
            return out

        benchmark(run_vanilla)
