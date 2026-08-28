from collections import defaultdict
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.nn import functional as F

from server.compiler.fusion_registry import apply_fusion
from server.compiler.node_registry import NODE_REGISTRY, get_build_kwargs
from server.compiler.validator import GraphValidator
from server.models.graph import GraphSpec


class GraphModule(nn.Module):
    """A compiled graph - executes nodes in topological order, routing tensors via edges."""

    def __init__(
        self,
        modules: dict[str, nn.Module],
        topo_order: list[str],
        edges: list[tuple[str, str, str, str]],  # (from_id, from_port, to_id, to_port)
        node_types: dict[str, str],
        node_outputs: dict[str, list[str]],
        meta: dict,
    ):
        super().__init__()
        self.node_modules = nn.ModuleDict(modules)
        self.topo_order = topo_order
        self.edges = edges
        self.node_types = node_types
        self.node_outputs = node_outputs
        self.meta = meta

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, caches=None):
        B, T = idx.shape
        values = {}

        # compute position indices
        if caches is not None and caches.get("kv") is not None:
            first_cache = next(iter(caches["kv"].values()))
            start_pos = first_cache[0].shape[2]
        else:
            start_pos = 0

        positions = torch.arange(start_pos, start_pos + T, device=idx.device)

        # seed graph inputs
        values[("_input", "idx")] = idx
        values[("_input", "positions")] = positions

        if caches is None:
            caches = {"kv": {}}

        new_caches = {}

        for node_id in self.topo_order:
            node_type = self.node_types[node_id]
            module = self.node_modules[node_id]

            # gather inputs from edges
            inputs = {}
            for from_id, from_port, to_id, to_port in self.edges:
                if to_id == node_id:
                    inputs[to_port] = values[(from_id, from_port)]

            # kv_cache needs special handling for cache state
            if node_type == "kv_cache":
                cache_val = caches["kv"].get(node_id)
                result = module(inputs["k"], inputs["v"], cache_val)
                k_out, v_out, new_cache = result
                values[(node_id, "k")] = k_out
                values[(node_id, "v")] = v_out
                new_caches[node_id] = new_cache
                continue

            # call forward with kwargs so port names match parameter names
            result = module(**inputs)

            # store outputs keyed by port name
            outputs = self.node_outputs[node_id]
            if isinstance(result, tuple):  # multiple tensor outputs
                for port, val in zip(outputs, result):
                    values[(node_id, port)] = val
            else:  # single tensor output
                values[(node_id, outputs[0])] = result

        # find lm_head output for logits
        lm_head_ids = [nid for nid, nt in self.node_types.items() if nt == "lm_head"]
        logits = values[(lm_head_ids[0], "out")]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, {"kv": new_caches}


class GraphCompiler:
    def __init__(self):
        self.validator = GraphValidator()

    def compile(self, graph: GraphSpec) -> GraphModule:
        result = self.validator.validate(graph)
        if not result.valid:
            raise ValueError(f"invalid graph: {result.errors}")

        meta = asdict(graph.meta)
        modules = {}
        node_types = {}

        # instantiate each node
        for node in graph.nodes:
            node_def = NODE_REGISTRY[node.type]
            kwargs = get_build_kwargs(node.type, meta)
            kwargs.update(node.config)  # override global defaults
            modules[node.id] = node_def.cls(**kwargs)
            node_types[node.id] = node.type

        # build edge list
        edges = [(e.from_node, e.from_port, e.to_node, e.to_port) for e in graph.edges]

        # topological sort
        topo_order = self._topo_sort(graph)

        # build output port mapping for each node
        node_outputs = {node.id: list(NODE_REGISTRY[node.type].outputs) for node in graph.nodes}

        # apply user-specified fusion groups (already validated by GraphValidator)
        if graph.fusion_groups:
            from server.compiler.fusion_registry import FUSION_BY_KERNEL

            groups = [(fg.nodes, FUSION_BY_KERNEL[fg.kernel]) for fg in graph.fusion_groups]
            modules, node_types, topo_order, edges, node_outputs = apply_fusion(
                groups, modules, node_types, topo_order, edges, node_outputs, meta
            )

        # weight init
        model = GraphModule(modules, topo_order, edges, node_types, node_outputs, meta)
        model.apply(self._init_weights)

        return model

    def _topo_sort(self, graph: GraphSpec) -> list[str]:
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

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
