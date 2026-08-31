from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict

from server.models.graph import GraphSpec


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

    # count incoming edges for each node
    for edge in graph.edges:
        if edge.from_node == "_input":
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


def graph_structure_hash(graph: GraphSpec) -> str:
    # excludes inference-based transforms like fusion and quantization
    nodes = sorted([(n.id, n.type, sorted(n.config.items())) for n in graph.nodes])
    edges = sorted([(e.from_node, e.from_port, e.to_node, e.to_port) for e in graph.edges])
    meta = sorted(asdict(graph.meta).items())
    blob = json.dumps([nodes, edges, meta], default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def graph_full_hash(graph: GraphSpec) -> str:
    # includes all train and inference-based transforms
    nodes = sorted([(n.id, n.type, n.quantized, sorted(n.config.items())) for n in graph.nodes])
    edges = sorted([(e.from_node, e.from_port, e.to_node, e.to_port) for e in graph.edges])
    meta = sorted(asdict(graph.meta).items())
    fusion = sorted(sorted(fg.nodes) for fg in graph.fusion_groups)
    blob = json.dumps([nodes, edges, meta, fusion], default=str).encode()
    return hashlib.sha256(blob).hexdigest()
