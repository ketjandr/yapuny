"""
Generate text from a trained Yapuny checkpoint.

Usage:
    python generate.py                          # default prompt
    python generate.py --prompt "Once upon"     # custom prompt
    python generate.py --tokens 200             # generate 200 tokens
"""

import argparse

import torch

from data.tokenizer import decode, encode, load_tokenizer
from model.gpt import GPT

CHECKPOINT_PATH = "checkpoints/yapuny.pt"


def generate():
    parser = argparse.ArgumentParser(description="Generate text with Yapuny")
    parser.add_argument("--prompt", type=str, default="The ", help="Starting text")
    parser.add_argument("--tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling (0 = disabled)")
    parser.add_argument(
        "--checkpoint", type=str, default=CHECKPOINT_PATH, help="Path to checkpoint"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model from checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load tokenizer
    tok = load_tokenizer()

    # Encode prompt
    prompt_ids = encode(tok, args.prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Generate
    top_k = args.top_k if args.top_k > 0 else None
    output = model.generate(
        idx, max_new_tokens=args.tokens, temperature=args.temperature, top_k=top_k
    )

    # Decode and print
    generated_ids = output[0].tolist()
    text = decode(tok, generated_ids)
    print(text)


if __name__ == "__main__":
    generate()
