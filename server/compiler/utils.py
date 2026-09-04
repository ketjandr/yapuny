from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict

from server.models.graph import EdgeSpec, FusionGroup, GraphSpec, NodeSpec


def eval_build_args(build_args: dict[str, str], meta: dict) -> dict:
    """Evaluate build_args expressions against graph meta."""
    return {param: eval(expr, {"__builtins__": {}}, meta) for param, expr in build_args.items()}


def topo_sort(graph: GraphSpec) -> list[str]:
    """Topological sort via Kahn's algorithm."""
    # build an adjacency list (unidirectional)
    adj = defaultdict(list)
    # in_degree: how many incoming edges (dependencies) a node has
    in_degree = defaultdict(int)
    node_ids = [n.id for n in graph.nodes]

    # initialize all nodes to 0 incoming edges
    for nid in node_ids:
        in_degree[nid] = 0

    # count incoming edges for each node (skip pseudo-endpoints: _input source, _output sink)
    for edge in graph.edges:
        if edge.from_node == "_input" or edge.to_node == "_output":
            continue
        adj[edge.from_node].append(edge.to_node)
        in_degree[edge.to_node] += 1

    # start with nodes that have no dependencies (in_degree == 0)
    # invariant: queue always contains nodes with in_degree == 0
    queue = [nid for nid in node_ids if in_degree[nid] == 0]
    order = []

    while queue:
        # pick next ready node (all its inputs are already processed)
        node = queue.pop(0)
        order.append(node)
        # "remove" this node by decrementing its neighbors' in_degree
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            # neighbor has no more unprocessed dependencies, it's ready
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # if we couldn't order all nodes, there's a cycle
    if len(order) != len(node_ids):
        raise ValueError("cycle detected during topological sort")

    return order


def _block_key(graph: GraphSpec) -> list[str]:
    # the block spec changes the network structure (unrolled n_layer times), so it's identity
    return sorted(graph.block.nodes) if graph.block else []


def graph_structure_hash(graph: GraphSpec) -> str:
    # excludes inference-based transforms like fusion and quantization
    nodes = sorted([(n.id, n.type, sorted(n.config.items())) for n in graph.nodes])
    edges = sorted([(e.from_node, e.from_port, e.to_node, e.to_port) for e in graph.edges])
    meta = sorted(asdict(graph.meta).items())
    blob = json.dumps([nodes, edges, meta, _block_key(graph)], default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def graph_full_hash(graph: GraphSpec) -> str:
    # includes all train and inference-based transforms
    nodes = sorted([(n.id, n.type, n.quantized, sorted(n.config.items())) for n in graph.nodes])
    edges = sorted([(e.from_node, e.from_port, e.to_node, e.to_port) for e in graph.edges])
    meta = sorted(asdict(graph.meta).items())
    fusion = sorted(sorted(fg.nodes) for fg in graph.fusion_groups)
    blob = json.dumps([nodes, edges, meta, fusion, _block_key(graph)], default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def expand_blocks(graph: GraphSpec) -> GraphSpec:
    """Unroll graph.block meta.n_layer times, chaining each layer's output to the next."""
    n = graph.meta.n_layer
    if not graph.block or n <= 1:
        return graph

    block = list(graph.block.nodes)
    b = set(block)

    def rid(layer: int, nid: str) -> str:
        return f"l{layer}_{nid}"

    internal, in_edges, out_edges, external = [], [], [], []
    for e in graph.edges:
        f_in, t_in = e.from_node in b, e.to_node in b
        if f_in and t_in:
            internal.append(e)
        elif t_in:
            in_edges.append(e)
        elif f_in:
            out_edges.append(e)
        else:
            external.append(e)

    in_srcs = {(e.from_node, e.from_port) for e in in_edges}
    out_srcs = {(e.from_node, e.from_port) for e in out_edges}
    if len(in_srcs) != 1:
        raise ValueError(f"block must have exactly one input tensor, got {len(in_srcs)}")
    if len(out_srcs) != 1:
        raise ValueError(f"block must have exactly one output tensor, got {len(out_srcs)}")
    in_src = next(iter(in_srcs))
    out_node, out_port = next(iter(out_srcs))

    node_by_id = {nd.id: nd for nd in graph.nodes}
    new_nodes = [nd for nd in graph.nodes if nd.id not in b]
    for layer in range(n):
        for nid in block:
            src = node_by_id[nid]
            new_nodes.append(
                NodeSpec(
                    id=rid(layer, nid),
                    type=src.type,
                    config=dict(src.config),
                    quantized=src.quantized,
                )
            )

    new_edges = list(external)
    for layer in range(n):
        for e in internal:
            new_edges.append(
                EdgeSpec(rid(layer, e.from_node), rid(layer, e.to_node), e.from_port, e.to_port)
            )
        # layer 0 takes the block's external input; later layers take the previous layer's output
        entry_node, entry_port = in_src if layer == 0 else (rid(layer - 1, out_node), out_port)
        for e in in_edges:
            new_edges.append(EdgeSpec(entry_node, rid(layer, e.to_node), entry_port, e.to_port))
    # the last layer's output feeds the epilogue
    for e in out_edges:
        new_edges.append(EdgeSpec(rid(n - 1, out_node), e.to_node, out_port, e.to_port))

    new_fusion = []
    for fg in graph.fusion_groups:
        fg_set = set(fg.nodes)
        if fg_set <= b:
            new_fusion.extend(FusionGroup([rid(layer, x) for x in fg.nodes]) for layer in range(n))
        elif fg_set & b:
            raise ValueError("fusion group straddles the block boundary")
        else:
            new_fusion.append(FusionGroup(list(fg.nodes)))

    return GraphSpec(
        nodes=new_nodes, edges=new_edges, fusion_groups=new_fusion, meta=graph.meta, block=None
    )


def _reachable(adj: dict[str, list[str]], start: str) -> set[str]:
    """DFS reachable set from start over adjacency adj (start itself is not included)."""
    seen, stack = set(), [start]
    while stack:
        for nxt in adj[stack.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def _subgraph(graph: GraphSpec, keep: set[str]) -> GraphSpec:
    """The subgraph induced by node-id set keep, retaining the pseudo-endpoint edges
    (_input -> ..., ... -> _output). Fusion groups are kept as-is so a group referencing a pruned
    node still surfaces in _check_fusion_groups rather than silently vanishing."""
    nodes = [n for n in graph.nodes if n.id in keep]
    edges = [
        e
        for e in graph.edges
        if (e.from_node in keep or e.from_node == "_input")
        and (e.to_node in keep or e.to_node == "_output")
    ]
    return GraphSpec(
        nodes=nodes, edges=edges, fusion_groups=graph.fusion_groups, meta=graph.meta, block=None
    )


def flow_subgraph(graph: GraphSpec) -> GraphSpec:
    """Nodes on some _input -> _output path (reachable from input AND reaching output); orphan and
    dead-end nodes are dropped. Idempotent. Assumes blocks are already expanded. Used to gate 'is
    there a complete path?' and to build only the reachable model."""
    successors, predecessors = defaultdict(list), defaultdict(list)
    for e in graph.edges:
        successors[e.from_node].append(e.to_node)
        predecessors[e.to_node].append(e.from_node)
    keep = _reachable(successors, "_input") & _reachable(predecessors, "_output")
    return _subgraph(graph, keep)


def output_cone(graph: GraphSpec) -> GraphSpec:
    """The subgraph the output depends on: every node that reaches _output, with its incoming
    edges."""
    predecessors = defaultdict(list)
    for e in graph.edges:
        predecessors[e.to_node].append(e.from_node)
    return _subgraph(graph, _reachable(predecessors, "_output"))


def logical_node_id(nid: str) -> str:
    """Map an unrolled node id back to its logical (per-block) id by stripping every l{layer}_
    prefix expand_blocks adds - including the copies embedded inside a fused kernel's id."""
    return re.sub(r"l\d+_", "", nid)


def has_inference_opts(graph: GraphSpec) -> bool:
    """Whether the graph carries inference-only transforms (fusion or quantization)."""
    return bool(graph.fusion_groups) or any(n.quantized for n in graph.nodes)


def strip_inference_opts(graph: GraphSpec) -> GraphSpec:
    """Derive the training graph by stripping inference-only operations."""
    plain = deepcopy(graph)
    plain.fusion_groups = []
    for n in plain.nodes:
        n.quantized = None
    return plain
