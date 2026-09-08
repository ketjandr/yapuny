# Yapuny

Visual GPT builder. Design transformer architectures as node graphs, compile them, train on your own corpus, generate text, and benchmark models live, all from the browser.

The frontend is a React canvas where you wire up transformer components (embeddings, attention, MLP, normalization, etc.) into a dataflow graph. The backend compiles the graph into a runnable PyTorch model, handles training (with live SSE loss streaming), tokenization, inference with/without KV-cache, and optional Triton-based operator fusion and weight quantization.

**Try it out: https://yapuny.vercel.app**

## Architecture

```
browser (React + React Flow)
   |
   v
worker (FastAPI, single-tenant)      <-- self-hosted: direct CORS
   or
gateway (FastAPI reverse proxy)      <-- shared tier: one worker subprocess per session
   |-> worker child 0 (port N)
   |-> worker child 1 (port N+1)
   ...
```

**Self-hosted path**: browser talks directly to a local or remote worker over CORS.

**Shared path**: browser talks to a gateway that spawns an isolated throwaway worker per `X-Yapuny-Session` header, idle-reaped after 10 min.

## Graph compiler

The compiler (`server/compiler/`) takes a `GraphSpec` (nodes, edges, meta, optional block + fusion groups) and produces a `GraphModule` (an `nn.Module` subclass):

1. **Block expansion** - the `block` field marks the repeated transformer layer; `expand_blocks()` unrolls it `n_layer` times, chaining each layer's output to the next and broadcasting `_input` edges (e.g. positions for RoPE).
2. **Flow pruning** - `flow_subgraph()` keeps only nodes on an `_input -> _output` path. Orphan/dead-end nodes are dropped silently.
3. **Validation** - cycle detection, required nodes, port compatibility, input satisfaction, fusion chain connectivity, quantization/fusion conflict checks.
4. **Instantiation** - each node's `build_args` are evaluated against graph meta (`n_embd`, `n_head`, etc.) to produce constructor kwargs. Pretrained weights are loaded per-module when the structure hash matches a saved package.
5. **Quantization** - marked nodes get their `nn.Linear` swapped for a Triton quantized linear (W8 or W4).
6. **Fusion** - resolved fusion groups replace sequences of nodes with a single fused Triton kernel, rewiring edges around the replacement.
Execution: `GraphModule.forward()` walks the topological order, gathers each node's inputs from the edge-routed value dict, runs the node, and stores outputs. KV-cache nodes are special-cased (two outputs + cache state). Loss is cross-entropy when targets are provided.

### Node types

| Category | Nodes |
|---|---|
| Embedding | `token_embedding`, `position_embedding` |
| Attention | `qkv_proj`, `rope`, `kv_cache`, `attention_score`, `causal_mask`, `softmax`, `value_weighted_sum`, `out_proj`, `flash_attention` |
| MLP | `mlp_up`, `mlp_activation`, `mlp_down` |
| Normalization | `layernorm` |
| Misc | `dropout`, `residual_add`, `lm_head` |

`flash_attention` replaces the `attention_score -> causal_mask -> softmax -> value_weighted_sum` chain with a single Triton kernel (forward: tiled online softmax; backward: falls back to reference torch attention until the backward kernel is written).

## Triton fusion kernels

All kernels live in `kernels/fusion/` and `kernels/quantization/`. They require a CUDA GPU and Triton >= 3.7. On CPU or when Triton is unavailable, the model runs with standard PyTorch ops, so no code changes needed, the registry just skips registration.

### Fusion patterns

The fusion registry (`server/compiler/fusion_registry.py`) subtitutes contiguous node chains matching these patterns with fused kernels at compile time:

| Pattern | Kernel | What it fuses |
|---|---|---|
| `residual_add -> layernorm` | `FusedResidualLayerNorm` | Elementwise add + layer norm in one pass |
| `attention_score -> causal_mask -> softmax` | `FusedScaleMaskSoftmax` | Scale, apply causal mask, and softmax in a single row-wise kernel |
| `mlp_up -> mlp_activation` | `FusedLinearGELU` | Tiled matmul + bias + GELU (tanh approximation with clamped exp) |
| `dropout -> residual_add` | `FusedDropoutResidual` | Bernoulli mask + elementwise add |
| `mlp_down -> dropout` | `FusedLinearDropout` | Tiled matmul + bias + dropout mask |
| `dropout -> residual_add -> layernorm` | `FusedDropoutResidualLayerNorm` | Three-op chain in one kernel |
| `mlp_down -> dropout -> residual_add` | `FusedLinearDropoutResidual` | Matmul + dropout + residual in one kernel |

Longer patterns are matched greedily first. Each fused module exposes `load_from_nodes()` / `save_to_nodes()` so trained weights transfer in and out of the fused representation.

### Flash attention

`kernels/fusion/flash_attention.py` is a tiled forward kernel implementing the FlashAttention algorithm:

- Online softmax (running max + sum correction per Q tile) to avoid materializing the full T x T attention matrix.
- Causal masking aligned to the end of K/V (`q_offset + (T_k - T_q) >= k_offset`), which handles both prefill (`T_q = T`) and decode (`T_q = 1, T_k = cached window`).
- Block sizes: `BLOCK_M=64, BLOCK_N=64`, `HEAD_DIM` rounded to next power of 2 (max 128).
- Grid: `(ceil(T_q / BLOCK_M), B * H)`.
- Backward: currently uses a reference torch implementation via `torch.autograd.Function`; native Triton backward is TODO.

### Weight quantization

`kernels/quantization/` provides INT8 and INT4 post-training quantization with custom Triton matmul kernels:

- **W8** (`quantized_linear_w8.py`): per-row symmetric quantization. Scale = max(|row|) / 127. Tiled matmul dequantizes inside the reduction loop (`w_q.to(float32) * scale`).
- **W4** (`quantized_linear_w4.py`): two INT4 values packed per `uint8` byte. Unpacked with nibble masking and sign-extended (`(nibble ^ 8).to(int8) - 8`). Two half-width dot products per K-tile (even/lo + odd/hi). Scale = max(|row|) / 7.

Quantizable node types: `qkv_proj`, `out_proj`, `mlp_up`, `mlp_down`, `lm_head` (any node with an `nn.Linear` marked in the registry's `quantize_attr`).

## Benchmarking

Three benchmark modes, all streamed as SSE to the frontend:

### Training benchmark (`POST /api/train/bench`)

Trains multiple compiled models sequentially on the same corpus/hyperparams and compares them. Per-model metrics:
- **steps/sec** - rolling average over the last 10 steps (wall-clock, includes GPU sync via `loss.item()`)
- **forward_ms / backward_ms** - median per-step timing (CUDA events on GPU, `perf_counter` on CPU)
- **peak_vram_mb** - `torch.cuda.max_memory_allocated`
- **loss curve** - downsampled `[step, loss]` pairs (up to 200 points) for live plotting
- **per-node profile** - a train-mode `profile_graph` run after training finishes (see below)

### Inference benchmark (`POST /api/generate/bench`)

Runs autoregressive generation across multiple trained models on the same prompt. Per-model metrics:
- **prefill_ms** - time for the initial forward pass (prompt encoding)
- **decode_ms_per_token** - total decode wall time / generated tokens
- **tokens/sec** - inverse of above
- **peak_vram_mb**

Warmup (3 tokens) runs before timing to JIT-compile any Triton kernels.

### Node profiling (`POST /api/bench/profile`)

Per-node execution time breakdown for a single model in either `train` or `decode` mode. Uses CUDA event pairs (`record` on pre/post forward hooks, `elapsed_time` after `synchronize`) on GPU, or `perf_counter` deltas on CPU. Reports each node's self-time in microseconds and its percentage of total. Node ids are mapped back to their logical (per-block) id via `logical_node_id()` so unrolled layers group correctly.

## Tokenizer

BPE via HuggingFace `tokenizers` (`data/tokenizer.py`). Trained per-model at training start from the active corpus. The tokenizer ships with each saved model package so vocab differences across architectures coexist. Special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`.

## Model storage

Trained models are persisted as packages in `models/<id>/` (overridable via `YAPUNY_MODELS_DIR`):

```
models/<uuid>/
  tokenizer.json   # HF tokenizer
  weights.pt       # torch state_dict
  meta.json        # structure_hash + status (writing -> trained)
```

Writes are atomic: `meta.json` is written as `status: "writing"` first, then updated to `status: "trained"` after weights + tokenizer are flushed. A crash mid-save leaves an invalid package that reads as absent.

## API

All routes are under `/api` and optionally token-protected (`WORKER_TOKEN` env var).

- **Graph**: `POST /api/graph/validate`, `POST /api/graph/compile`
- **Model**: `POST /api/model/status`, `GET /api/models`, `DELETE /api/model/{id}`
- **Train**: `POST /api/train/stream` (SSE), `GET /api/train/follow`, `POST /api/train/stop`, `POST /api/train/bench` (multi-model SSE)
- **Generate**: `POST /api/generate/stream` (SSE), `POST /api/generate/stop`, `POST /api/generate/bench` (multi-model compare)
- **Data**: `POST /api/data/upload`, `GET /api/data/corpus`, `POST /api/data/corpus`, `DELETE /api/data/corpus`
- **Fusion**: `GET /api/fusion/available`, `POST /api/fusion/suggest`
- **Quantization**: `GET /api/quantization/available`
- **Benchmark**: `POST /api/bench/profile` (per-node timing via CUDA events)

Training and inference are mutually exclusive on the GPU (concurrent requests return 409).

## Dev setup

### Prerequisites

- Python >= 3.9
- Node.js >= 18
- CUDA GPU + Triton >= 3.7 for fusion/quantization/flash-attention (optional, everything else works on CPU)

### Backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"          # add [gpu] on Linux with CUDA for Triton kernels
```

### Frontend

```bash
cd frontend
npm install
npm run dev                      # Vite dev server on :5173, proxies /api to :8000
```

### Run the worker

```bash
# from repo root, with venv active:
uvicorn server.app:app --host 0.0.0.0 --port 8000
# or via the CLI entry point:
yapuny
```

### Tests

```bash
pytest                           # server + worker tests (CPU)
pytest tests/fusion/             # fusion kernel tests (needs CUDA)
pytest tests/quantization/       # quantization kernel tests (needs CUDA)
```

## Install as a standalone worker

One-line install using [uv](https://github.com/astral-sh/uv) (no Docker, no venv management):

**macOS / Linux:**
```bash
curl -LsSf https://raw.githubusercontent.com/ketjandr/yapuny/main/install.sh | sh
```

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/ketjandr/yapuny/main/install.ps1 | iex
```

This installs `uv` (if needed) + the worker with its own Python 3.11. GPU (CUDA + Triton) is auto-detected on Linux; Windows native installs skip Triton (use WSL2 for fusion kernels). Re-running the command updates if there's a new version (SHA-based check), otherwise just starts the worker.

After install, start with: `yapuny`

## Gateway (shared tier)

The gateway (`gateway/app.py`) is a FastAPI reverse proxy that spawns one worker subprocess per browser session. Used for the free shared CPU worker.

Key env vars:
- `GW_MAX_WORKERS` - max concurrent sessions (default 6)
- `GW_MAX_PER_IP` - sessions per client IP (default 2)
- `GW_IDLE_TIMEOUT` - reap idle sessions after N seconds (default 600)
- `FRONTEND_ORIGIN` - allowed CORS origins
- `GW_MAX_STEPS`, `GW_MAX_BATCH`, `GW_MAX_N_EMBD`, `GW_MAX_BLOCK`, `GW_MAX_VOCAB`, `GW_MAX_LAYERS` - compute caps injected into child workers

