import pytest
import torch
import torch.nn.functional as F

from kernels.flash_attention import FlashAttention, flash_attention

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEVICE = "cuda"
B, H, T, D = 1, 6, 256, 64


@pytest.fixture
def setup():
    torch.manual_seed(42)
    q = torch.randn(B, H, T, D, device=DEVICE)
    k = torch.randn(B, H, T, D, device=DEVICE)
    v = torch.randn(B, H, T, D, device=DEVICE)
    return q, k, v


def sdpa_attention(q, k, v, is_causal=True):
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


def naive_attention(q, k, v, is_causal=True):
    scale = q.shape[-1] ** -0.5
    s = q @ k.transpose(-2, -1) * scale
    if is_causal:
        mask = torch.tril(torch.ones(s.shape[-2], s.shape[-1], device=s.device))
        s = s.masked_fill(mask == 0, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return p @ v


class TestCorrectness:
    def test_causal_matches_pytorch(self, setup):
        q, k, v = setup
        expected = sdpa_attention(q, k, v, is_causal=True)
        actual = flash_attention(q, k, v, is_causal=True)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    def test_non_causal_matches_pytorch(self, setup):
        q, k, v = setup
        expected = sdpa_attention(q, k, v, is_causal=False)
        actual = flash_attention(q, k, v, is_causal=False)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    def test_small_seq_len(self):
        torch.manual_seed(42)
        q = torch.randn(1, 2, 8, D, device=DEVICE)
        k = torch.randn(1, 2, 8, D, device=DEVICE)
        v = torch.randn(1, 2, 8, D, device=DEVICE)
        expected = sdpa_attention(q, k, v, is_causal=True)
        actual = flash_attention(q, k, v, is_causal=True)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    def test_seq_len_not_multiple_of_block(self):
        torch.manual_seed(42)
        seq_len = 100  # not a multiple of BLOCK_M=64 or BLOCK_N=64
        q = torch.randn(1, 2, seq_len, D, device=DEVICE)
        k = torch.randn(1, 2, seq_len, D, device=DEVICE)
        v = torch.randn(1, 2, seq_len, D, device=DEVICE)
        expected = sdpa_attention(q, k, v, is_causal=True)
        actual = flash_attention(q, k, v, is_causal=True)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    def test_multi_batch(self):
        torch.manual_seed(42)
        q = torch.randn(4, H, T, D, device=DEVICE)
        k = torch.randn(4, H, T, D, device=DEVICE)
        v = torch.randn(4, H, T, D, device=DEVICE)
        expected = sdpa_attention(q, k, v, is_causal=True)
        actual = flash_attention(q, k, v, is_causal=True)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    def test_module_wrapper(self, setup):
        q, k, v = setup
        module = FlashAttention(is_causal=True)
        expected = sdpa_attention(q, k, v, is_causal=True)
        actual = module(q, k, v)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


class TestBenchmark:
    @pytest.mark.parametrize("seq_len", [64, 128, 256])
    def test_flash(self, seq_len, benchmark):
        torch.manual_seed(42)
        q = torch.randn(B, H, seq_len, D, device=DEVICE)
        k = torch.randn(B, H, seq_len, D, device=DEVICE)
        v = torch.randn(B, H, seq_len, D, device=DEVICE)

        def run_flash():
            torch.cuda.synchronize()
            out = flash_attention(q, k, v, is_causal=True)
            torch.cuda.synchronize()
            return out

        benchmark(run_flash)

    @pytest.mark.parametrize("seq_len", [64, 128, 256])
    def test_sdpa(self, seq_len, benchmark):
        torch.manual_seed(42)
        q = torch.randn(B, H, seq_len, D, device=DEVICE)
        k = torch.randn(B, H, seq_len, D, device=DEVICE)
        v = torch.randn(B, H, seq_len, D, device=DEVICE)

        def run_sdpa():
            torch.cuda.synchronize()
            out = sdpa_attention(q, k, v, is_causal=True)
            torch.cuda.synchronize()
            return out

        benchmark(run_sdpa)

    @pytest.mark.parametrize("seq_len", [64, 128, 256])
    def test_naive(self, seq_len, benchmark):
        torch.manual_seed(42)
        q = torch.randn(B, H, seq_len, D, device=DEVICE)
        k = torch.randn(B, H, seq_len, D, device=DEVICE)
        v = torch.randn(B, H, seq_len, D, device=DEVICE)

        def run_naive():
            torch.cuda.synchronize()
            out = naive_attention(q, k, v, is_causal=True)
            torch.cuda.synchronize()
            return out

        benchmark(run_naive)
