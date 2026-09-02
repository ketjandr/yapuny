from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

import torch

from server.compiler.compiler import GraphModule
from server.compiler.utils import logical_node_id


@dataclass
class TimingResult:
    median: float
    p05: float
    p95: float
    samples: list[float] = field(repr=False)


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


def _profile_nodes(model: GraphModule, run, device: torch.device) -> dict[str, float]:
    """Time each node in microseconds via forward hooks - CUDA events on GPU, wall clock on CPU."""
    use_cuda = device.type == "cuda"
    times: dict[str, float] = {}
    starts: dict[str, object] = {}
    pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
    handles = []

    def hooks_for(nid):
        if use_cuda:

            def pre(module, args):
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                starts[nid] = ev

            def post(module, args, output):
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                pending.append((nid, starts[nid], ev))
        else:

            def pre(module, args):
                starts[nid] = time.perf_counter()

            def post(module, args, output):
                times[nid] = times.get(nid, 0) + (time.perf_counter() - starts[nid]) * 1e6

        return pre, post

    for nid, module in model.node_modules.items():
        pre, post = hooks_for(nid)
        handles.append(module.register_forward_pre_hook(pre))
        handles.append(module.register_forward_hook(post))

    try:
        run()
        if use_cuda:
            torch.cuda.synchronize(device)
            for nid, start, end in pending:
                times[nid] = times.get(nid, 0) + start.elapsed_time(end) * 1000  # ms to us
    finally:
        for h in handles:
            h.remove()

    return times


@dataclass
class NodeProfile:
    node_id: str  # the real (unrolled) module id, e.g. l0_mlp_up
    logical_id: str  # the per-block id it maps to, e.g. mlp_up (for grouping/display)
    self_us: float
    pct: float


@dataclass
class ProfileResult:
    nodes: list[NodeProfile]
    total_us: float
    env: dict


def profile_graph(
    model: GraphModule,
    device: torch.device,
    mode: str = "decode",
    prompt_tokens: int = 64,
    new_tokens: int = 128,
    batch_size: int = 1,
    warmup: int = 3,
) -> ProfileResult:
    block_size = model.meta["block_size"]
    vocab_size = model.meta["vocab_size"]
    use_cuda = device.type == "cuda"

    if mode == "train":
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        def run():
            x = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
            y = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
            _, loss, _ = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    else:
        model.eval()

        def run():
            idx = torch.randint(0, vocab_size, (batch_size, prompt_tokens), device=device)
            logits, _, caches = model(idx[:, -block_size:])
            for step in range(new_tokens):
                next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                idx = torch.cat((idx, next_id), dim=1)
                if step == new_tokens - 1:
                    break
                logits, _, _ = model(idx[:, -block_size:])

    for _ in range(warmup):
        run()
    if use_cuda:
        torch.cuda.synchronize(device)

    node_times = _profile_nodes(model, run, device)

    total = sum(node_times.values()) or 1.0
    nodes = sorted(
        [
            NodeProfile(
                node_id=nid, logical_id=logical_node_id(nid), self_us=us, pct=us / total * 100
            )
            for nid, us in node_times.items()
        ],
        key=lambda n: n.self_us,
        reverse=True,
    )

    if mode == "train":
        model.eval()

    return ProfileResult(nodes=nodes, total_us=total, env=_collect_env(device))
