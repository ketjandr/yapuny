from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from data.tokenizer import (
    decode,
    encode,
    save_tokenizer,
    train_tokenizer,
)
from server.compiler.compiler import GraphCompiler, GraphModule, cache_length
from server.compiler.utils import (
    expand_blocks,
    graph_full_hash,
    graph_structure_hash,
    has_inference_opts,
    strip_inference_opts,
)
from server.models.graph import GraphSpec
from worker import store

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
TOKENIZER_PATH = DATA_DIR / "yapuny_tokenizer.json"
MAX_CORPUS_BYTES = 10 * 1024 * 1024  # 10 MB cap


def _bench_slot(model_id: str, max_steps: int) -> dict:
    """A per-model progress slot in a benchmark run's streamed state."""
    return {
        "id": model_id,
        "status": "pending",
        "phase": None,  # "tokenizing" | "training" while this model is the current one
        "step": 0,
        "max_steps": max_steps,
        "train_loss": None,
        "steps_per_sec": None,
        "curve": [],  # downsampled [step, loss] history, so a reload keeps the graph
        "bench": None,
    }


@dataclass
class ModelCacheEntry:
    """One compiled model held in memory, addressed by its frontend model id.
    tokenizer is None until the model is trained (also serves as the 'trained' flag)."""

    full_hash: str
    structure_hash: str
    graph: GraphSpec
    model: GraphModule
    tokenizer: Tokenizer | None


class Worker:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.compiler = GraphCompiler()
        # compiled-model cache (in-memory, per worker), addressed by model id.
        # This is the worker's only model state - there is no implicit "active" model.
        self.cache: dict[str, ModelCacheEntry] = {}
        self.training = False
        self.training_id = None
        self.train_state = None  # single-model run (POST /train/stream)
        self.bench_state = None  # multi-model benchmark run (POST /train/bench)

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

    def compile_model(self, model_id: str, graph_data: dict):
        """Build the runnable model, loading trained weights from the
        locker if the architecture matches, and cache it by id."""
        graph = GraphSpec.from_dict(graph_data)
        full_hash = graph_full_hash(graph)
        struct_hash = graph_structure_hash(graph)

        pkg = store.load(model_id)
        if pkg is not None and pkg.structure_hash == struct_hash:
            model = self.compiler.compile(graph, pretrained_state=pkg.weights)
            tokenizer = pkg.tokenizer
            weights = "loaded"
        else:
            model = self.compiler.compile(graph)
            tokenizer = None
            weights = "reinitialized"

        model.to(self.device)
        model.eval()

        self.cache[model_id] = ModelCacheEntry(full_hash, struct_hash, graph, model, tokenizer)

        return {
            "status": "compiled",
            "weights": weights,
            "trained": tokenizer is not None,
            "device": str(self.device),
            "model_info": self._model_info(model, graph),
        }

    def model_status(self, model_id: str, graph_data: dict):
        """Readiness check for a graph context (canvas) switch."""
        full_hash = graph_full_hash(GraphSpec.from_dict(graph_data))

        entry = self.cache.get(model_id)
        if entry is None or entry.full_hash != full_hash:
            return {"status": "needs_compile"}

        return {
            "status": "ready",
            "trained": entry.tokenizer is not None,
            "model_info": self._model_info(entry.model, entry.graph),
        }

    def delete_model(self, model_id: str):
        self.cache.pop(model_id, None)
        existed = store.delete(model_id)
        return {"status": "deleted" if existed else "not_found"}

    def list_models(self):
        return store.list_ids()

    def _model_info(self, model, graph: GraphSpec) -> dict:
        param_count = sum(p.numel() for p in model.parameters())
        weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        meta = model.meta
        quantized = {n.id: n.quantized for n in graph.nodes if n.quantized}

        return {
            "param_count": param_count,
            "weight_bytes": weight_bytes,
            "vocab_size": meta["vocab_size"],
            "block_size": meta["block_size"],
            "n_layer": meta.get("n_layer"),
            "n_head": meta.get("n_head"),
            "n_embd": meta.get("n_embd"),
            "quantized_nodes": quantized if quantized else None,
        }

    def train(
        self,
        model_id: str,
        max_steps: int = 2000,
        batch_size: int = 32,
        learning_rate: float = 3e-4,
        bench: bool = False,
    ):
        # guards set an error train_state (not just return) so /train/stream, which polls
        # train_state, sees a terminal status and ends instead of hanging on a run that never began
        def _fail(msg: str):
            self.train_state = {"status": "error", "error": msg}
            return self.train_state

        entry = self.cache.get(model_id)
        if entry is None:
            return _fail("model not compiled - compile first")

        if self.training:
            return _fail("training already in progress")

        if not (RAW_DIR / "corpus.txt").exists():
            return _fail("no corpus - upload a corpus first")

        has_opts, plain_graph, model = self._compile_fresh(entry)
        self.training = True
        self.training_id = model_id
        self.train_state = {
            "step": 0,
            "max_steps": max_steps,
            "train_loss": None,
            "steps_per_sec": None,
            "curve": [],  # downsampled [step, loss] history, so a reload keeps the graph
            "status": "running",
            "phase": "tokenizing",  # each model trains its own tokenizer before stepping
        }
        if bench and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        try:
            # train this model's own tokenizer at its vocab_size, then tokenize the corpus
            tokenizer, train_data, _ = self._tokenize_corpus(model.meta["vocab_size"])
            self.train_state["phase"] = "training"

            completed, fwd, bwd = self._train_loop(
                model, train_data, self.train_state, max_steps, batch_size, learning_rate, bench
            )
            # only a full run marks completion; a stopped run keeps its "stopped" status and step.
            # The save is gated on completion, so a stopped run never overwrites existing weights.
            if not completed:
                self.train_state["status"] = "stopped"
            else:
                self.train_state["step"] = max_steps
                self.train_state["status"] = "completed"
                if bench and fwd:
                    self.train_state["bench"] = self._final_bench(model, fwd, bwd)
                self._persist(model_id, model, plain_graph, has_opts, entry, tokenizer)
                self.train_state["saved"] = model_id
        except Exception as e:
            # surface the failure as a terminal state so the stream ends instead of hanging
            self.train_state["status"] = "error"
            self.train_state["error"] = str(e)
        finally:
            self.training = False
            self.training_id = None

    def train_bench(
        self,
        model_ids: list[str],
        max_steps: int = 500,
        batch_size: int = 16,
        learning_rate: float = 3e-4,
    ):
        """Train several already-compiled models one after another, benchmarking each. Progress for
        all models lives in self.bench_state (streamed by /train/bench); models train sequentially,
        so `current` marks the one in flight. Stop is cooperative via self.training."""

        def _fail(msg: str):
            self.bench_state = {"status": "error", "error": msg, "current": 0, "models": []}

        if self.training:
            _fail("training already in progress")
            return
        if not (RAW_DIR / "corpus.txt").exists():
            _fail("no corpus - upload a corpus first")
            return

        entries = []
        for mid in model_ids:
            e = self.cache.get(mid)
            if e is None:
                _fail("all selected models must be compiled first")
                return
            entries.append((mid, e))

        self.training = True
        self.bench_state = {
            "status": "running",
            "current": 0,
            "models": [_bench_slot(mid, max_steps) for mid, _ in entries],
        }
        # models that share a vocab_size (e.g. opt variants of one graph) reuse one tokenizer
        tok_cache: dict[int, tuple] = {}

        try:
            for i, (mid, entry) in enumerate(entries):
                if not self.training:
                    break
                self.bench_state["current"] = i
                slot = self.bench_state["models"][i]
                slot["status"] = "running"

                has_opts, plain_graph, model = self._compile_fresh(entry)
                vocab = model.meta["vocab_size"]
                if vocab not in tok_cache:
                    slot["phase"] = "tokenizing"
                    tok_cache[vocab] = self._tokenize_corpus(vocab)
                tokenizer, train_data, _ = tok_cache[vocab]
                slot["phase"] = "training"

                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)

                completed, fwd, bwd = self._train_loop(
                    model, train_data, slot, max_steps, batch_size, learning_rate, bench=True
                )
                if not completed:
                    slot["status"] = "stopped"
                    break

                slot["step"] = max_steps
                if fwd:
                    slot["bench"] = self._final_bench(model, fwd, bwd)
                self._persist(mid, model, plain_graph, has_opts, entry, tokenizer)
                slot["status"] = "done"

            self.bench_state["status"] = "completed" if self.training else "stopped"
        except Exception as e:
            self.bench_state["status"] = "error"
            self.bench_state["error"] = str(e)
        finally:
            self.training = False

    def _tokenize_corpus(self, vocab_size: int, val_fraction: float = 0.1):
        """Train a fresh BPE tokenizer on the corpus at `vocab_size`, then tokenize + split it.
        Returns (tokenizer, train_ids, val_ids)."""
        corpus_path = RAW_DIR / "corpus.txt"
        tokenizer = train_tokenizer(corpus_path, vocab_size=vocab_size)
        ids = encode(tokenizer, corpus_path.read_text(encoding="utf-8"))
        split = int(len(ids) * (1 - val_fraction))
        dtype = np.uint16 if tokenizer.get_vocab_size() < 65536 else np.uint32
        return tokenizer, np.array(ids[:split], dtype=dtype), np.array(ids[split:], dtype=dtype)

    def _compile_fresh(self, entry: ModelCacheEntry):
        """Compile a fresh (randomly initialized) training model from an entry's graph. Fusion/quant
        are inference-only (no backward kernels), so we train the stripped plain graph."""
        has_opts = has_inference_opts(entry.graph)
        plain_graph = strip_inference_opts(entry.graph) if has_opts else entry.graph
        model = self.compiler.compile(plain_graph).to(self.device)
        return has_opts, plain_graph, model

    def _train_loop(self, model, train_data, state, max_steps, batch_size, learning_rate, bench):
        """Train `model` for up to max_steps on `train_data` (a token-id array), writing live
        progress (step, train_loss, steps_per_sec) into `state`. Returns (done, fwd, bwd)."""
        block_size = model.meta["block_size"]
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        fwd_times: list[float] = []
        bwd_times: list[float] = []
        completed = True

        for step in range(max_steps):
            if not self.training:
                completed = False
                break

            x, y = self._get_batch(train_data, block_size, batch_size)

            if bench:
                t0 = self._stamp()
            _, loss, _ = model(x, y)
            if bench:
                t_fwd = self._stamp()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if bench:
                t_bwd = self._stamp()
            optimizer.step()

            lv = loss.item()
            state["step"] = step + 1
            state["train_loss"] = lv

            if step % max(1, max_steps // 200) == 0 or step == max_steps - 1:
                state["curve"].append([step + 1, lv])
            if bench:
                fwd_times.append((t_fwd - t0) * 1000)
                bwd_times.append((t_bwd - t_fwd) * 1000)
                recent = [f + b for f, b in zip(fwd_times[-10:], bwd_times[-10:])]
                state["steps_per_sec"] = 1000.0 / (sum(recent) / len(recent))

        model.eval()
        return completed, fwd_times, bwd_times

    def _final_bench(self, model, fwd_times: list[float], bwd_times: list[float]) -> dict:
        """Aggregate per-step timings + a per-node train profile into the benchmark summary."""
        from dataclasses import asdict

        from worker.bench import _timing_result, profile_graph

        fwd = _timing_result(fwd_times)
        bwd = _timing_result(bwd_times)
        step_ms = fwd.median + bwd.median
        peak_vram = None
        if self.device.type == "cuda":
            peak_vram = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
        profile = profile_graph(model=model, device=self.device, mode="train", warmup=1)
        return {
            "forward_ms": fwd.median,
            "backward_ms": bwd.median,
            "steps_per_sec": 1000.0 / step_ms if step_ms > 0 else 0,
            "peak_vram_mb": peak_vram,
            "profile": {"nodes": [asdict(n) for n in profile.nodes], "total_us": profile.total_us},
        }

    def _persist(self, model_id, model, plain_graph, has_opts, entry: ModelCacheEntry, tokenizer):
        """Commit trained weights + this model's own tokenizer to the locker, and refresh the
        in-memory cache for inference."""
        unfused = self._unfused_state(model, plain_graph)
        store.save(model_id, tokenizer, unfused, entry.structure_hash)
        if has_opts:
            # rebuild the fused/quant inference model on the freshly trained weights
            inference_model = self.compiler.compile(entry.graph, pretrained_state=unfused)
            inference_model.to(self.device)
            inference_model.eval()
            self.cache[model_id] = ModelCacheEntry(
                entry.full_hash, entry.structure_hash, entry.graph, inference_model, tokenizer
            )
        else:
            # no opts: the freshly trained model IS the inference model
            entry.model = model
            entry.tokenizer = tokenizer

    def stop_training(self):
        if not self.training:
            return {"error": "no training in progress"}
        self.training = False
        return {"status": "stopping"}

    def get_train_status(self):
        # training_id lets the frontend tell whether an in-progress run is for the model it has open
        # (so it can reattach to the live stream after a reload); it clears when a run ends
        if self.train_state is None:
            return {"status": "idle", "training_id": None}
        return {**self.train_state, "training_id": self.training_id}

    def get_bench_status(self):
        if self.bench_state is None:
            return {"status": "idle"}
        return self.bench_state

    def _unfused_state(self, model, graph: GraphSpec) -> dict:
        """Return trained weights in unfused form so any graph variant can reload them.
        Today training always passes a plain graph, so this hits the fast path below.
        TODO: fused training (train fused -> unfuse for storage) will exercise the fusion branch;
        it needs backward kernels for the fusion ops, which don't exist yet."""
        if not graph.fusion_groups:
            return model.state_dict()

        from copy import deepcopy

        # the compiled models carry the unrolled ids, so map over the expanded graph's
        # nodes/fusion groups (no-op when there is no block, so non-block is unchanged)
        expanded = expand_blocks(graph)
        node_types = {n.id: n.type for n in expanded.nodes}

        fused_node_ids = set()
        for fg in expanded.fusion_groups:
            fused_node_ids.update(fg.nodes)

        unfused_graph = deepcopy(graph)
        unfused_graph.fusion_groups = []
        unfused_model = self.compiler.compile(unfused_graph)

        for nid in unfused_model.node_modules:
            if nid not in fused_node_ids:
                src = model.node_modules[nid]
                dst = unfused_model.node_modules[nid]
                dst.load_state_dict(src.state_dict())

        for fg in expanded.fusion_groups:
            fused_id = "_fused_" + "_".join(fg.nodes)
            fused_module = model.node_modules[fused_id]

            unfused_nodes = {node_types[nid]: unfused_model.node_modules[nid] for nid in fg.nodes}

            if hasattr(fused_module, "save_to_nodes"):
                fused_module.save_to_nodes(unfused_nodes)

        return unfused_model.state_dict()

    def _stamp(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def _get_batch(self, data: np.ndarray, block_size: int, batch_size: int):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack(
            [torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]
        )
        return x.to(self.device), y.to(self.device)

    def _sample(self, logits, temperature, top_k):
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def generate(
        self,
        model_id: str,
        prompt_ids: list[int],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        bench: bool = False,
    ):
        result = None
        for event in self.generate_stream(
            model_id=model_id,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            bench=bench,
        ):
            if event["event"] == "error":
                return event["data"]
            if event["event"] == "done":
                result = event["data"]
        return result

    def generate_stream(
        self,
        model_id: str,
        prompt_ids: list[int],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        bench: bool = False,
    ):
        entry = self.cache.get(model_id)
        if entry is None:
            yield {"event": "error", "data": {"error": "model not compiled"}}
            return
        yield from self._stream_tokens(
            entry.model, entry.tokenizer, prompt_ids, max_new_tokens, temperature, top_k, bench
        )

    @torch.no_grad()
    def _stream_tokens(
        self,
        model,
        tok,
        prompt_ids: list[int],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        bench: bool = False,
    ):
        """Pure generation over an explicit (model, tokenizer)."""
        idx = torch.tensor([prompt_ids], device=self.device)
        block_size = model.meta["block_size"]
        tokens = []

        if bench:
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            t_prefill_start = self._stamp()

        # prefill - whole prompt in one pass, seeds the cache if the graph has one
        logits, _, caches = model(idx[:, -block_size:])

        if bench:
            t_prefill_end = self._stamp()
            t_decode_start = t_prefill_end
            prefill_ms = (t_prefill_end - t_prefill_start) * 1000
            yield {"event": "prefill", "data": {"prefill_ms": prefill_ms}}

        for step in range(max_new_tokens):
            next_id = self._sample(logits, temperature, top_k)
            idx = torch.cat((idx, next_id), dim=1)
            token = next_id.item()
            tokens.append(token)

            text = decode(tok, [token]) if tok else None
            event = {"event": "token", "data": {"token": token, "text": text, "step": step}}

            if bench:
                elapsed = (self._stamp() - t_decode_start) * 1000
                count = len(tokens)
                event["data"]["bench"] = {
                    "tokens_per_sec": count / (elapsed / 1000) if elapsed > 0 else 0,
                    "elapsed_ms": elapsed,
                }

            yield event

            if step == max_new_tokens - 1:
                break  # have every token - skip the unused final forward

            cached_len = cache_length(caches)
            if cached_len is not None and cached_len < block_size:
                # decode - feed only the new token, reuse cached k/v
                logits, _, caches = model(next_id, caches=caches)
            else:
                # no kv_cache node, or context window full - recompute the window
                logits, _, _ = model(idx[:, -block_size:])

        full_text = decode(tok, tokens) if tok else None
        done_data = {"tokens": tokens, "text": full_text}

        if bench:
            t_decode_end = self._stamp()
            decode_ms = (t_decode_end - t_decode_start) * 1000
            num_tokens = len(tokens)
            peak_vram = None
            if self.device.type == "cuda":
                peak_vram = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
            done_data["bench"] = {
                "prefill_ms": prefill_ms,
                "decode_ms_per_token": decode_ms / num_tokens if num_tokens > 0 else 0,
                "tokens_per_sec": num_tokens / (decode_ms / 1000) if decode_ms > 0 else 0,
                "peak_vram_mb": peak_vram,
            }

        yield {"event": "done", "data": done_data}

        if bench:
            from dataclasses import asdict

            from worker.bench import profile_graph

            result = profile_graph(
                model=model,
                device=self.device,
                mode="decode",
                prompt_tokens=len(prompt_ids),
                new_tokens=max_new_tokens,
                warmup=1,
            )
            yield {
                "event": "profile",
                "data": {
                    "nodes": [asdict(n) for n in result.nodes],
                    "total_us": result.total_us,
                },
            }

    def _tokenizer(self, model_id: str):
        """Resolve a compiled model's tokenizer, raising if not compiled/trained."""
        entry = self.cache.get(model_id)
        if entry is None:
            raise ValueError("model not compiled - compile first")
        if entry.tokenizer is None:
            raise ValueError("model has no tokenizer - train it first")
        return entry.tokenizer

    def decode_tokens(self, model_id: str, token_ids: list[int]):
        try:
            tok = self._tokenizer(model_id)
        except ValueError as e:
            return {"error": str(e)}
        return {"text": decode(tok, token_ids)}

    def encode_prompt(self, model_id: str, text: str) -> list[int]:
        return encode(self._tokenizer(model_id), text)

    def profile(self, model_id: str, mode: str, prompt_tokens: int, new_tokens: int, warmup: int):
        from worker.bench import profile_graph

        entry = self.cache.get(model_id)
        if entry is None:
            return None
        return profile_graph(
            model=entry.model,
            device=self.device,
            mode=mode,
            prompt_tokens=prompt_tokens,
            new_tokens=new_tokens,
            warmup=warmup,
        )
