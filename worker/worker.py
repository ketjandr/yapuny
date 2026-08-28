from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from data.tokenizer import (
    decode,
    encode,
    load_tokenizer,
    save_tokenizer,
    train_tokenizer,
)
from server.compiler.compiler import GraphCompiler
from server.models.graph import GraphSpec

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
TOKENIZER_PATH = DATA_DIR / "yapuny_tokenizer.json"
MAX_CORPUS_BYTES = 10 * 1024 * 1024  # 10 MB cap


class Worker:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.compiler = GraphCompiler()
        self.model = None
        self.training = False
        self.train_state = None

    def upload_corpus(self, content: bytes, filename: str):
        if len(content) > MAX_CORPUS_BYTES:
            return {"error": f"corpus too large ({len(content)} bytes, max {MAX_CORPUS_BYTES})"}

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        corpus_path = RAW_DIR / "corpus.txt"
        corpus_path.write_bytes(content)

        text = content.decode("utf-8", errors="replace")
        char_count = len(text)
        line_count = text.count("\n") + 1

        return {
            "status": "uploaded",
            "filename": filename,
            "size_bytes": len(content),
            "chars": char_count,
            "lines": line_count,
        }

    def prepare_data(self, vocab_size: int = 8000, val_fraction: float = 0.1):
        corpus_path = RAW_DIR / "corpus.txt"
        if not corpus_path.exists():
            return {"error": "no corpus uploaded - upload a corpus first"}

        # train tokenizer on corpus
        tok = train_tokenizer(corpus_path, vocab_size=vocab_size)
        save_tokenizer(tok, TOKENIZER_PATH)

        # tokenize corpus
        text = corpus_path.read_text(encoding="utf-8")
        ids = encode(tok, text)

        # train/val split
        split_idx = int(len(ids) * (1 - val_fraction))
        train_ids = ids[:split_idx]
        val_ids = ids[split_idx:]

        actual_vocab = tok.get_vocab_size()
        dtype = np.uint16 if actual_vocab < 65536 else np.uint32

        train_arr = np.array(train_ids, dtype=dtype)
        val_arr = np.array(val_ids, dtype=dtype)

        train_arr.tofile(DATA_DIR / "train.bin")
        val_arr.tofile(DATA_DIR / "val.bin")

        return {
            "status": "prepared",
            "vocab_size": actual_vocab,
            "train_tokens": len(train_arr),
            "val_tokens": len(val_arr),
        }

    def compile_graph(self, graph_data: dict):
        graph = GraphSpec.from_dict(graph_data)
        self.model = self.compiler.compile(graph)
        self.model.to(self.device)
        self.model.eval()
        return {"status": "compiled", "device": str(self.device)}

    def train(
        self,
        max_steps: int = 2000,
        batch_size: int = 32,
        learning_rate: float = 3e-4,
        eval_interval: int = 200,
        eval_iters: int = 50,
        checkpoint_path: str | None = None,
    ):
        if self.model is None:
            return {"error": "no model compiled"}

        if self.training:
            return {"error": "training already in progress"}

        block_size = self.model.meta["block_size"]
        self.model.train()
        self.training = True
        self.train_state = {
            "step": 0,
            "max_steps": max_steps,
            "train_loss": None,
            "val_loss": None,
            "status": "running",
        }

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        try:
            for step in range(max_steps):
                if not self.training:
                    self.train_state["status"] = "stopped"
                    break

                # eval
                if step % eval_interval == 0:
                    losses = self._estimate_loss(block_size, batch_size, eval_iters)
                    self.train_state.update(
                        {
                            "step": step,
                            "train_loss": losses["train"],
                            "val_loss": losses["val"],
                        }
                    )

                # train step
                x, y = self._get_batch("train", block_size, batch_size)
                _, loss, _ = self.model(x, y)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            self.train_state["step"] = max_steps
            self.train_state["status"] = "completed"

            # final eval
            losses = self._estimate_loss(block_size, batch_size, eval_iters)
            self.train_state["train_loss"] = losses["train"]
            self.train_state["val_loss"] = losses["val"]

            # save checkpoint
            if checkpoint_path is None:
                checkpoint_path = str(
                    Path(__file__).resolve().parent.parent / "checkpoints" / "yapuny_graph.pt"
                )
            path = Path(checkpoint_path)
            path.parent.mkdir(exist_ok=True)
            torch.save({"model": self.model.state_dict()}, path)
            self.train_state["checkpoint"] = str(path)

        finally:
            self.training = False
            self.model.eval()

        return self.train_state

    def stop_training(self):
        if not self.training:
            return {"error": "no training in progress"}
        self.training = False
        return {"status": "stopping"}

    def get_train_status(self):
        if self.train_state is None:
            return {"status": "idle"}
        return self.train_state

    def _get_batch(self, split: str, block_size: int, batch_size: int):
        path = DATA_DIR / ("train.bin" if split == "train" else "val.bin")
        data = np.memmap(path, dtype=np.uint16, mode="r")

        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack(
            [torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]
        )
        return x.to(self.device), y.to(self.device)

    @torch.no_grad()
    def _estimate_loss(self, block_size: int, batch_size: int, eval_iters: int):
        self.model.eval()
        out = {}
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for i in range(eval_iters):
                x, y = self._get_batch(split, block_size, batch_size)
                _, loss, _ = self.model(x, y)
                losses[i] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
    ):
        if self.model is None:
            return {"error": "no model compiled"}

        idx = torch.tensor([prompt_ids], device=self.device)
        tokens = []

        for _ in range(max_new_tokens):
            logits, _, _ = self.model(idx)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            tokens.append(next_id.item())

        return {"tokens": tokens}

    def decode_tokens(self, token_ids: list[int]):
        if not TOKENIZER_PATH.exists():
            return {"error": "no tokenizer found - prepare data first"}

        tok = load_tokenizer(TOKENIZER_PATH)
        text = decode(tok, token_ids)
        return {"text": text}
