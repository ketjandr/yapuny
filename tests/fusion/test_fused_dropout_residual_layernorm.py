import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels.fusion.fused_dropout_residual_layernorm import (
    FusedDropoutResidualLayerNorm,
    fused_dropout_residual_layernorm,
)

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEVICE = "cuda"
B, T, C = 1, 256, 384
P_DROP = 0.1


@pytest.fixture
def setup():
    x = torch.randn(B, T, C, device=DEVICE)
    residual = torch.randn(B, T, C, device=DEVICE)
    ln = nn.LayerNorm(C).to(DEVICE)
    return x, residual, ln


def vanilla_dropout_residual_layernorm(x, residual, weight, bias, p_drop, training):
    return F.layer_norm(
        x + F.dropout(residual, p=p_drop, training=training),
        [weight.shape[0]],
        weight,
        bias,
    )


class TestCorrectness:
    def test_eval_matches_pytorch(self, setup):
        """With training=False, dropout is identity so output is exact."""
        x, residual, ln = setup
        expected = ln(x + residual)
        actual = fused_dropout_residual_layernorm(
            x, residual, ln.weight, ln.bias, P_DROP, training=False
        )
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_2d_input(self, setup):
        x, residual, ln = setup
        x_2d = x.reshape(-1, C)
        res_2d = residual.reshape(-1, C)
        expected = ln(x_2d + res_2d)
        actual = fused_dropout_residual_layernorm(
            x_2d, res_2d, ln.weight, ln.bias, P_DROP, training=False
        )
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_output_shape(self, setup):
        x, residual, ln = setup
        out = fused_dropout_residual_layernorm(
            x, residual, ln.weight, ln.bias, P_DROP, training=True
        )
        assert out.shape == (B, T, C)

    def test_train_output_differs_from_eval(self, setup):
        x, residual, ln = setup
        out_train = fused_dropout_residual_layernorm(
            x, residual, ln.weight, ln.bias, p_drop=0.5, training=True
        )
        out_eval = fused_dropout_residual_layernorm(
            x, residual, ln.weight, ln.bias, p_drop=0.5, training=False
        )
        diff = (out_train - out_eval).abs()
        assert (diff > 1e-4).any(), "dropout should cause differences"

    def test_module_wrapper(self, setup):
        x, residual, ln = setup
        fused = FusedDropoutResidualLayerNorm(C, P_DROP).to(DEVICE)
        fused.weight.data.copy_(ln.weight.data)
        fused.bias.data.copy_(ln.bias.data)
        fused.eval()
        expected = ln(x + residual)
        actual = fused(x, residual)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


class TestBenchmark:
    @pytest.mark.parametrize("seq_len", [32, 128, 256])
    def test_fused(self, seq_len, benchmark):
        x = torch.randn(B, seq_len, C, device=DEVICE)
        residual = torch.randn(B, seq_len, C, device=DEVICE)
        ln = nn.LayerNorm(C).to(DEVICE)

        def run_fused():
            torch.cuda.synchronize()
            out = fused_dropout_residual_layernorm(
                x, residual, ln.weight, ln.bias, P_DROP, training=True
            )
            torch.cuda.synchronize()
            return out

        benchmark(run_fused)

    @pytest.mark.parametrize("seq_len", [32, 128, 256])
    def test_vanilla(self, seq_len, benchmark):
        x = torch.randn(B, seq_len, C, device=DEVICE)
        residual = torch.randn(B, seq_len, C, device=DEVICE)
        ln = nn.LayerNorm(C).to(DEVICE)

        def run_vanilla():
            torch.cuda.synchronize()
            out = vanilla_dropout_residual_layernorm(
                x, residual, ln.weight, ln.bias, P_DROP, training=True
            )
            torch.cuda.synchronize()
            return out

        benchmark(run_vanilla)
