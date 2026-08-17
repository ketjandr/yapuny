import pytest
import torch
import torch.nn.functional as F

from kernels.fused_scale_mask_softmax import FusedScaleMaskSoftmax, fused_scale_mask_softmax

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEVICE = "cuda"
B, N_HEAD, T, T_K, HEAD_DIM = 1, 6, 256, 256, 64
SCALE = 1.0 / (HEAD_DIM ** 0.5)


@pytest.fixture
def setup():
    scores = torch.randn(B, N_HEAD, T, T_K, device=DEVICE)
    mask = torch.tril(torch.ones(T, T_K, device=DEVICE)).unsqueeze(0).unsqueeze(0)
    return scores, mask


def vanilla_scale_mask_softmax(scores, mask, scale):
    scores = scores * scale
    scores = scores.masked_fill(mask == 0, float("-inf"))
    return F.softmax(scores, dim=-1)


class TestCorrectness:
    def test_matches_pytorch(self, setup):
        scores, mask = setup
        expected = vanilla_scale_mask_softmax(scores, mask, SCALE)
        actual = fused_scale_mask_softmax(scores, mask, SCALE)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_rows_sum_to_one(self, setup):
        scores, mask = setup
        out = fused_scale_mask_softmax(scores, mask, SCALE)
        row_sums = out.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4)

    def test_masked_positions_are_zero(self, setup):
        scores, mask = setup
        out = fused_scale_mask_softmax(scores, mask, SCALE)
        masked = out.masked_select(mask == 0)
        torch.testing.assert_close(masked, torch.zeros_like(masked), atol=1e-6, rtol=0)

    def test_module_wrapper(self, setup):
        scores, mask = setup
        module = FusedScaleMaskSoftmax(HEAD_DIM).to(DEVICE)
        expected = vanilla_scale_mask_softmax(scores, mask, SCALE)
        actual = module(scores, mask)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_small_T(self):
        scores = torch.randn(B, N_HEAD, 4, 4, device=DEVICE)
        mask = torch.tril(torch.ones(4, 4, device=DEVICE)).unsqueeze(0).unsqueeze(0)
        expected = vanilla_scale_mask_softmax(scores, mask, SCALE)
        actual = fused_scale_mask_softmax(scores, mask, SCALE)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


class TestBenchmark:
    @pytest.mark.parametrize("seq_len", [32, 128, 256])
    def test_fused(self, seq_len, benchmark):
        scores = torch.randn(B, N_HEAD, seq_len, seq_len, device=DEVICE)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=DEVICE)).unsqueeze(0).unsqueeze(0)

        def run_fused():
            torch.cuda.synchronize()
            out = fused_scale_mask_softmax(scores, mask, SCALE)
            torch.cuda.synchronize()
            return out

        benchmark(run_fused)

    @pytest.mark.parametrize("seq_len", [32, 128, 256])
    def test_vanilla(self, seq_len, benchmark):
        scores = torch.randn(B, N_HEAD, seq_len, seq_len, device=DEVICE)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=DEVICE)).unsqueeze(0).unsqueeze(0)

        def run_vanilla():
            torch.cuda.synchronize()
            out = vanilla_scale_mask_softmax(scores, mask, SCALE)
            torch.cuda.synchronize()
            return out

        benchmark(run_vanilla)
