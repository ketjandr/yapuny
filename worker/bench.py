from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import torch

from server.compiler.compiler import GraphModule


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


@dataclass
class NodeProfile:
    node_id: str
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

    model.profile = True

    try:
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

        activities = [torch.profiler.ProfilerActivity.CPU]
        if use_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        with torch.profiler.profile(activities=activities) as prof:
            run()
            if use_cuda:
                torch.cuda.synchronize(device)

        node_times: dict[str, float] = {}
        for evt in prof.key_averages():
            if evt.key.startswith("node::"):
                node_id = evt.key[6:]
                us = evt.self_cuda_time_total if use_cuda else evt.self_cpu_time_total
                node_times[node_id] = node_times.get(node_id, 0) + us

        total = sum(node_times.values()) or 1.0
        nodes = sorted(
            [
                NodeProfile(node_id=nid, self_us=us, pct=us / total * 100)
                for nid, us in node_times.items()
            ],
            key=lambda n: n.self_us,
            reverse=True,
        )

        if mode == "train":
            model.eval()

        return ProfileResult(nodes=nodes, total_us=total, env=_collect_env(device))

    finally:
        model.profile = False
