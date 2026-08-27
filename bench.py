"""
Usage:
    python bench.py
    python bench.py --checkpoint path/to/yapuny.pt
"""

import argparse
import time

import torch

from data.tokenizer import encode, load_tokenizer
from model.gpt import GPT

CHECKPOINT_PATH = "checkpoints/yapuny.pt"
PROMPT = "The "
GEN_LENGTHS = [32, 64, 128, 256]
WARMUP_RUNS = 2
BENCH_RUNS = 5


def load_model(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def bench_generate(model, idx, max_new_tokens, device):
    """Time a single generate() call, returns elapsed seconds."""
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.generate(idx.clone(), max_new_tokens=max_new_tokens)
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(description="Benchmark Yapuny inference")
    parser.add_argument(
        "--checkpoint", type=str, default=CHECKPOINT_PATH, help="Path to checkpoint"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)
    tok = load_tokenizer()

    prompt_ids = encode(tok, PROMPT)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    print(f"Prompt: {PROMPT!r} ({len(prompt_ids)} tokens)")
    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")
    print()

    # Header
    print(f"{'Tokens':>8} {'Total (s)':>10} {'ms/token':>10} {'tok/s':>10}")
    print("-" * 42)

    for n_tokens in GEN_LENGTHS:
        # Warmup
        for _ in range(WARMUP_RUNS):
            model.generate(idx.clone(), max_new_tokens=n_tokens)
            if device == "cuda":
                torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(BENCH_RUNS):
            elapsed = bench_generate(model, idx, n_tokens, device)
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        ms_per_token = (avg_time / n_tokens) * 1000
        tokens_per_sec = n_tokens / avg_time

        print(f"{n_tokens:>8} {avg_time:>10.3f} {ms_per_token:>10.2f} {tokens_per_sec:>10.1f}")

    print()
    print("(Baseline: naive generate, no KV cache)")


if __name__ == "__main__":
    main()
