from pathlib import Path

import numpy as np
import torch

from model.config import GPTConfig
from model.gpt import GPT

# ---- config (hardcode for now, move to configs/*.yaml later) ----
DATA_DIR = Path(__file__).parent / "data"
BLOCK_SIZE = 256
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
MAX_STEPS = 2000
EVAL_INTERVAL = 200
EVAL_ITERS = 50
CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "yapuny.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def get_batch(split: str):
    """Load a random batch of (input, target) sequences from the memmapped .bin file."""
    path = DATA_DIR / ("train.bin" if split == "train" else "val.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")

    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(data[i:i + BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + BLOCK_SIZE].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    # Average loss from multiple batches
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for i in range(EVAL_ITERS):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    config = GPTConfig(block_size=BLOCK_SIZE)
    model = GPT(config).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for step in range(MAX_STEPS):
        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model)
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        x, y = get_batch("train")
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    CHECKPOINT_PATH.parent.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config}, CHECKPOINT_PATH)
    print(f"Saved checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
