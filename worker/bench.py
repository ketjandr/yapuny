from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

import torch

from server.compiler.compiler import GraphModule, cache_length


@dataclass
class TimingResult:
    median: float
    p05: float
    p95: float
    samples: list[float] = field(repr=False)


@dataclass
class GraphResult:
    graph_id: str
    structure_hash: str
    meta: dict
    param_count: int
    weight_bytes: int
    peak_vram_mb: float | None
    prefill_ms: TimingResult | None = None
    decode_ms_per_token: TimingResult | None = None
    tokens_per_sec: float | None = None
    steps_per_sec: float | None = None
    forward_ms: TimingResult | None = None
    backward_ms: TimingResult | None = None


@dataclass
class BenchResult:
    mode: str
    env: dict
    graphs: list[GraphResult]
    structure_groups: list[list[str]]


def _use_cuda_events(device: torch.device) -> bool:
    return device.type == "cuda"


def _collect_env(device: torch.device) -> dict:
    env = {"torch": torch.__version__, "device": str(device)}
    if device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(device)
        env["sm_count"] = torch.cuda.get_device_properties(device).multi_processor_count
        try:
            import triton
            env["triton"] = triton.__version__
        except ImportError:
            env["triton"] = None
    return env


def _param_count(model: GraphModule) -> int:
    return sum(p.numel() for p in model.parameters())


def _weight_bytes(model: GraphModule) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def _percentile(data: list[float], pct: float) -> float:
    s = sorted(data)
    k = (len(s) - 1) * pct / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + frac * (s[hi] - s[lo])


def _timing_result(samples: list[float]) -> TimingResult:
    return TimingResult(
        median=statistics.median(samples),
        p05=_percentile(samples, 5),
        p95=_percentile(samples, 95),
        samples=samples,
    )


def _bench_inference(
    model: GraphModule,
    device: torch.device,
    prompt_tokens: int,
    new_tokens: int,
    batch_size: int,
    repeats: int,
    warmup: int,
) -> dict:
    block_size = model.meta["block_size"]
    vocab_size = model.meta["vocab_size"]
    use_events = _use_cuda_events(device)

    def make_input():
        return torch.randint(0, vocab_size, (batch_size, prompt_tokens), device=device)

    # warmup
    for _ in range(warmup):
        idx = make_input()
        logits, _, caches = model(idx[:, -block_size:])
        for _ in range(min(new_tokens, 4)):
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cached_len = cache_length(caches)
            if cached_len is not None and cached_len < block_size:
                logits, _, caches = model(next_id, caches=caches)
            else:
                idx = torch.cat((idx, next_id), dim=1)
                logits, _, _ = model(idx[:, -block_size:])
    if use_events:
        torch.cuda.synchronize(device)

    prefill_times = []
    decode_times = []

    for _ in range(repeats):
        idx = make_input()

        # prefill
        if use_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            logits, _, caches = model(idx[:, -block_size:])
            end.record()
            torch.cuda.synchronize(device)
            prefill_times.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            logits, _, caches = model(idx[:, -block_size:])
            prefill_times.append((time.perf_counter() - t0) * 1000)

        # decode
        if use_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        decode_t0 = None if use_events else time.perf_counter()

        for step in range(new_tokens):
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat((idx, next_id), dim=1)
            if step == new_tokens - 1:
                break
            cached_len = cache_length(caches)
            if cached_len is not None and cached_len < block_size:
                logits, _, caches = model(next_id, caches=caches)
            else:
                logits, _, _ = model(idx[:, -block_size:])

        if use_events:
            end.record()
            torch.cuda.synchronize(device)
            decode_times.append(start.elapsed_time(end))
        else:
            decode_times.append((time.perf_counter() - decode_t0) * 1000)

    per_token = [t / new_tokens for t in decode_times]
    median_per_token = statistics.median(per_token)

    return {
        "prefill_ms": _timing_result(prefill_times),
        "decode_ms_per_token": _timing_result(per_token),
        "tokens_per_sec": 1000.0 / median_per_token if median_per_token > 0 else 0,
    }


def _bench_train(
    model: GraphModule,
    device: torch.device,
    batch_size: int,
    repeats: int,
    warmup: int,
) -> dict:
    block_size = model.meta["block_size"]
    vocab_size = model.meta["vocab_size"]
    use_events = _use_cuda_events(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    def make_batch():
        x = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
        y = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
        return x, y

    model.train()

    # warmup
    for _ in range(warmup):
        x, y = make_batch()
        _, loss, _ = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if use_events:
        torch.cuda.synchronize(device)

    forward_times = []
    backward_times = []
    step_times = []

    for _ in range(repeats):
        x, y = make_batch()

        if use_events:
            fwd_start = torch.cuda.Event(enable_timing=True)
            fwd_end = torch.cuda.Event(enable_timing=True)
            bwd_start = torch.cuda.Event(enable_timing=True)
            bwd_end = torch.cuda.Event(enable_timing=True)

            fwd_start.record()
            _, loss, _ = model(x, y)
            fwd_end.record()

            optimizer.zero_grad(set_to_none=True)

            bwd_start.record()
            loss.backward()
            bwd_end.record()

            optimizer.step()
            torch.cuda.synchronize(device)

            forward_times.append(fwd_start.elapsed_time(fwd_end))
            backward_times.append(bwd_start.elapsed_time(bwd_end))
            step_times.append(
                fwd_start.elapsed_time(fwd_end) + bwd_start.elapsed_time(bwd_end)
            )
        else:
            t0 = time.perf_counter()
            _, loss, _ = model(x, y)
            t_fwd = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            t_bwd = time.perf_counter()

            optimizer.step()

            forward_times.append((t_fwd - t0) * 1000)
            backward_times.append((t_bwd - t_fwd) * 1000)
            step_times.append((t_bwd - t0) * 1000)

    model.eval()

    median_step = statistics.median(step_times)

    return {
        "forward_ms": _timing_result(forward_times),
        "backward_ms": _timing_result(backward_times),
        "steps_per_sec": 1000.0 / median_step if median_step > 0 else 0,
    }


def bench_graph(
    model: GraphModule,
    device: torch.device,
    mode: str,
    prompt_tokens: int = 64,
    new_tokens: int = 128,
    batch_size: int = 1,
    repeats: int = 20,
    warmup: int = 5,
) -> dict:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if mode == "train":
        metrics = _bench_train(model, device, batch_size, repeats, warmup)
    else:
        metrics = _bench_inference(
            model, device, prompt_tokens, new_tokens, batch_size, repeats, warmup,
        )

    peak_vram = None
    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return {
        "param_count": _param_count(model),
        "weight_bytes": _weight_bytes(model),
        "peak_vram_mb": peak_vram,
        **metrics,
    }


def run_benchmark(
    graphs: list[dict],
    device: torch.device,
    mode: str = "decode",
    prompt_tokens: int = 64,
    new_tokens: int = 128,
    batch_size: int = 1,
    repeats: int = 20,
    warmup: int = 5,
) -> BenchResult:
    """
    Run a benchmark across multiple compiled graphs.

    Each entry in graphs must have:
      - "graph_id": str
      - "structure_hash": str
      - "model": compiled GraphModule (already on device)
      - "meta": dict of graph meta
    """
    env = _collect_env(device)

    # group by structure hash
    hash_groups: dict[str, list[str]] = {}
    for g in graphs:
        h = g["structure_hash"]
        hash_groups.setdefault(h, []).append(g["graph_id"])
    structure_groups = list(hash_groups.values())

    results = []
    for g in graphs:
        metrics = bench_graph(
            model=g["model"],
            device=device,
            mode=mode,
            prompt_tokens=prompt_tokens,
            new_tokens=new_tokens,
            batch_size=batch_size,
            repeats=repeats,
            warmup=warmup,
        )

        results.append(GraphResult(
            graph_id=g["graph_id"],
            structure_hash=g["structure_hash"],
            meta=g["meta"],
            param_count=metrics["param_count"],
            weight_bytes=metrics["weight_bytes"],
            peak_vram_mb=metrics["peak_vram_mb"],
            prefill_ms=metrics.get("prefill_ms"),
            decode_ms_per_token=metrics.get("decode_ms_per_token"),
            tokens_per_sec=metrics.get("tokens_per_sec"),
            steps_per_sec=metrics.get("steps_per_sec"),
            forward_ms=metrics.get("forward_ms"),
            backward_ms=metrics.get("backward_ms"),
        ))

    return BenchResult(
        mode=mode,
        env=env,
        graphs=results,
        structure_groups=structure_groups,
    )
