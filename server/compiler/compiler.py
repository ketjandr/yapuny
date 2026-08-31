from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn as nn
from torch.nn import functional as F

from server.compiler.fusion_registry import apply_fusion
from server.compiler.node_registry import NODE_REGISTRY, get_build_kwargs
from server.compiler.utils import topo_sort
from server.compiler.validator import GraphValidator
from server.models.graph import GraphSpec


def cache_length(caches: dict | None) -> int | None:
    """Cached sequence length, or None if the graph has no kv_cache node."""
    kv = (caches or {}).get("kv")
    if not kv:
        return None
    k, _ = next(iter(kv.values()))
    return k.shape[2]


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

    def _exec_node(self, node_id, node_type, module, inputs, caches, new_caches, values):
        """Run one node and store its outputs."""
        if node_type == "kv_cache":
            cache_val = caches["kv"].get(node_id)
            k_out, v_out, new_cache = module(inputs["k"], inputs["v"], cache_val)
            values[(node_id, "k")] = k_out
            values[(node_id, "v")] = v_out
            new_caches[node_id] = new_cache
            return

        result = module(**inputs)  # call forward on this node

        outputs = self.node_outputs[node_id]
        if isinstance(result, tuple):
            for port, val in zip(outputs, result):
                values[(node_id, port)] = val
        else:
            values[(node_id, outputs[0])] = result

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, caches=None):
        B, T = idx.shape
        values = {}

        # position indices continue from whatever is already cached
        start_pos = cache_length(caches) or 0

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

            self._exec_node(node_id, node_type, module, inputs, caches, new_caches, values)

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

    def compile(
        self,
        graph: GraphSpec,
        pretrained_state: dict | None = None,
    ) -> GraphModule:
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
        topo_order = topo_sort(graph)

        # build output port mapping for each node
        node_outputs = {node.id: list(NODE_REGISTRY[node.type].outputs) for node in graph.nodes}

        if pretrained_state is not None:
            # no structural change - load trained weights into freshly instantiated modules
            # pretrained_state keys are namespaced like "node_modules.b0_mlp_up.fc.weight",
            # so we strip the prefix to get what each module's load_state_dict expects
            for node_id, module in modules.items():
                prefix = f"node_modules.{node_id}."
                node_state = {
                    param_name[len(prefix) :]: param_tensor
                    for param_name, param_tensor in pretrained_state.items()
                    if param_name.startswith(prefix)
                }
                if node_state:
                    module.load_state_dict(node_state)

        # quantize marked nodes
        quantized_nodes = {n.id: (n.type, n.quantized) for n in graph.nodes if n.quantized}
        if quantized_nodes:
            from server.compiler.quantization_registry import quantize_module

            for node_id, (ntype, mode) in quantized_nodes.items():
                quantize_module(modules[node_id], ntype, mode)

        # resolve fusion kernels and apply them to the graph
        transfer_weights = pretrained_state is not None
        resolved_fusions = self.validator.resolve_fusions(graph)
        if resolved_fusions:
            modules, node_types, topo_order, edges, node_outputs = apply_fusion(
                resolved_fusions,
                modules,
                node_types,
                topo_order,
                edges,
                node_outputs,
                meta,
                transfer_weights=transfer_weights,
            )

        model = GraphModule(modules, topo_order, edges, node_types, node_outputs, meta)

        if pretrained_state is None:
            model.apply(self._init_weights)

        return model

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif hasattr(module, "init_weights"):
            module.init_weights()
