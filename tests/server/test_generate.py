import torch

from server.models.graph import GraphSpec
from tests.server.graph_factory import default_gpt_graph
from worker.worker import Worker

TINY = dict(n_layer=2, n_head=2, n_embd=32, block_size=16, vocab_size=64)
PROMPT = [1, 2, 3, 4]


def _worker(graph_dict):
    w = Worker()
    w.graph = GraphSpec.from_dict(graph_dict)
    w.model = w.compiler.compile(w.graph).to(w.device)
    w.model.eval()
    return w


def _no_kv_cache(graph_dict):
    """Strip kv_cache nodes, rewiring score/vws to read k/v straight from qkv."""
    cache_ids = {n["id"] for n in graph_dict["nodes"] if n["type"] == "kv_cache"}
    graph_dict["nodes"] = [n for n in graph_dict["nodes"] if n["id"] not in cache_ids]

    # qkv -> kvcache edges tell us which qkv feeds each cache
    source = {
        e["to_node"]: e["from_node"] for e in graph_dict["edges"] if e["to_node"] in cache_ids
    }

    rewired = []
    for e in graph_dict["edges"]:
        if e["to_node"] in cache_ids:
            continue  # drop qkv -> cache
        if e["from_node"] in cache_ids:
            # cache -> consumer becomes qkv -> consumer on the same port
            e = dict(e, from_node=source[e["from_node"]])
        rewired.append(e)
    graph_dict["edges"] = rewired
    return graph_dict


class TestKVCacheGeneration:
    def test_cached_matches_uncached(self):
        """Same weights + same seed must give identical tokens with and without cache."""
        g = default_gpt_graph(**TINY)
        cached = _worker(g)
        state = cached.model.state_dict()

        uncached = _worker(_no_kv_cache(default_gpt_graph(**TINY)))
        uncached.model = uncached.compiler.compile(
            uncached.graph,
            pretrained_state=state,
        ).to(uncached.device)
        uncached.model.eval()

        torch.manual_seed(0)
        a = cached.generate(PROMPT, max_new_tokens=8, temperature=1.0)

        torch.manual_seed(0)
        b = uncached.generate(PROMPT, max_new_tokens=8, temperature=1.0)

        assert a["tokens"] == b["tokens"]

    def test_cache_is_actually_used(self):
        """Decode steps must feed one token, not the whole prefix."""
        w = _worker(default_gpt_graph(**TINY))

        widths = []
        original = w.model.forward

        def spy(idx, targets=None, caches=None):
            widths.append(idx.shape[1])
            return original(idx, targets, caches)

        w.model.forward = spy
        w.generate(PROMPT, max_new_tokens=5)

        assert widths[0] == len(PROMPT)  # prefill sees the prompt
        assert widths[1:] == [1] * (len(widths) - 1)  # decode sees one token

    def test_no_kv_cache_node_does_not_crash(self):
        """Empty cache dict is falsy but not None - must not raise StopIteration."""
        w = _worker(_no_kv_cache(default_gpt_graph(**TINY)))
        out = w.generate(PROMPT, max_new_tokens=4)
        assert len(out["tokens"]) == 4

    def test_generation_past_block_size(self):
        """Cache must stop growing at block_size instead of overflowing pos embedding."""
        w = _worker(default_gpt_graph(**TINY))
        out = w.generate(PROMPT, max_new_tokens=TINY["block_size"] + 8)
        assert len(out["tokens"]) == TINY["block_size"] + 8

    def test_token_count_and_range(self):
        w = _worker(default_gpt_graph(**TINY))
        out = w.generate(PROMPT, max_new_tokens=12)
        assert len(out["tokens"]) == 12
        assert all(0 <= t < TINY["vocab_size"] for t in out["tokens"])
