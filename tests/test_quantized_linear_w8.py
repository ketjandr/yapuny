import pytest
import torch
import torch.nn.functional as F

from kernels.quantize import dequantize_weights_int8, quantize_weights_int8
from kernels.quantized_linear_w8 import QuantizedLinearW8, quantized_linear_w8

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


def vanilla_linear(x, weight, bias):
    return F.linear(x, weight, bias)


def quantized_manual(x, weight, bias):
    w_q, scale = quantize_weights_int8(weight)
    w_dq = dequantize_weights_int8(w_q, scale)
    return F.linear(x, w_dq, bias)


class TestCorrectness:
    def test_matches_manual(self, setup):
        x, weight, bias = setup
        w_q, scale = quantize_weights_int8(weight)
        expected = quantized_manual(x, weight, bias)
        actual = quantized_linear_w8(x, w_q, scale.squeeze(-1), bias)
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)

    def test_output_shape(self, setup):
        x, weight, bias = setup
        w_q, scale = quantize_weights_int8(weight)
        out = quantized_linear_w8(x, w_q, scale.squeeze(-1), bias)
        assert out.shape == (B, T, N)

    def test_2d_input(self, setup):
        x, weight, bias = setup
        x_2d = x.reshape(-1, C)
        w_q, scale = quantize_weights_int8(weight)
        expected = quantized_manual(x_2d, weight, bias)
        actual = quantized_linear_w8(x_2d, w_q, scale.squeeze(-1), bias)
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)

    def test_module_wrapper(self, setup):
        x, weight, bias = setup
        w_q, scale = quantize_weights_int8(weight)
        module = QuantizedLinearW8(C, N).to(DEVICE)
        module.w_q.copy_(w_q)
        module.scale.copy_(scale.squeeze(-1))
        module.bias.copy_(bias)
        expected = quantized_manual(x, weight, bias)
        actual = module(x)
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)

    def test_small_dims(self):
        x = torch.randn(1, 4, 16, device=DEVICE)
        weight = torch.randn(32, 16, device=DEVICE)
        bias = torch.randn(32, device=DEVICE)
        w_q, scale = quantize_weights_int8(weight)
        expected = quantized_manual(x, weight, bias)
        actual = quantized_linear_w8(x, w_q, scale.squeeze(-1), bias)
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)


class TestAccuracy:
    def test_w8_reconstruction_error(self):
        weight = torch.randn(N, C, device=DEVICE)
        w_q, scale = quantize_weights_int8(weight)
        w_dq = dequantize_weights_int8(w_q, scale)
        max_err = (weight - w_dq).abs().max().item()
        max_scale = scale.max().item()
        assert max_err < max_scale / 2 + 1e-6, f"max error {max_err} exceeds scale/2 {max_scale/2}"

    def test_w8_vs_fp32_relative_error(self, setup):
        x, weight, bias = setup
        y_fp32 = vanilla_linear(x, weight, bias)
        y_w8 = quantized_manual(x, weight, bias)
        rel_err = (y_fp32 - y_w8).norm() / y_fp32.norm()
        assert rel_err < 0.05, f"relative error {rel_err:.4f} exceeds 5%"

    def test_w8_zero_bias(self):
        x = torch.randn(B, T, C, device=DEVICE)
        weight = torch.randn(N, C, device=DEVICE)
        bias = torch.zeros(N, device=DEVICE)
        w_q, scale = quantize_weights_int8(weight)
        expected = quantized_manual(x, weight, bias)
        actual = quantized_linear_w8(x, w_q, scale.squeeze(-1), bias)
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)


class TestBenchmark:
    @pytest.mark.parametrize("seq_len", [32, 128, 256])
    def test_w8_kernel(self, seq_len, benchmark):
        x = torch.randn(B, seq_len, C, device=DEVICE)
        weight = torch.randn(N, C, device=DEVICE)
        bias = torch.randn(N, device=DEVICE)
        w_q, scale = quantize_weights_int8(weight)
        scale_1d = scale.squeeze(-1)

        def run_kernel():
            torch.cuda.synchronize()
            out = quantized_linear_w8(x, w_q, scale_1d, bias)
            torch.cuda.synchronize()
            return out

        benchmark(run_kernel)

    @pytest.mark.parametrize("seq_len", [32, 128, 256])
    def test_vanilla(self, seq_len, benchmark):
        x = torch.randn(B, seq_len, C, device=DEVICE)
        weight = torch.randn(N, C, device=DEVICE)
        bias = torch.randn(N, device=DEVICE)

        def run_vanilla():
            torch.cuda.synchronize()
            out = vanilla_linear(x, weight, bias)
            torch.cuda.synchronize()
            return out

        benchmark(run_vanilla)
