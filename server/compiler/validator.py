from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from server.compiler.fusion_registry import FUSION_BY_PATTERN, FusionDef
from server.compiler.node_registry import NODE_REGISTRY
from server.models.graph import GraphSpec


@dataclass
class ValidationResult:
    errors: list[str]
    warnings: list[str]

    @property
    def valid(self) -> bool:
        return len(self.errors) == 0


REQUIRED_NODE_TYPES = {
    "token_embedding",
    "position_embedding",
    "qkv_proj",
    "attention_score",
    "softmax",
    "value_weighted_sum",
    "out_proj",
    "residual_add",
    "mlp_up",
    "mlp_activation",
    "mlp_down",
    "lm_head",
}


class GraphValidator:
    def validate(self, graph: GraphSpec) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        self._check_unknown_types(graph, errors)
        self._check_required_nodes(graph, errors)
        self._check_cycles(graph, errors)
        self._check_dangling_edges(graph, errors)
        self._check_port_connections(graph, errors)
        self._check_quantization(graph, errors)
        self._check_fusion_groups(graph, errors)
        self._check_optional_warnings(graph, warnings)

        return ValidationResult(errors=errors, warnings=warnings)

    def resolve_fusions(
        self,
        graph: GraphSpec,
    ) -> list[tuple[list[str], FusionDef]]:
        """Match fusion groups to registry entries by node type pattern."""
        node_types = {n.id: n.type for n in graph.nodes}
        resolved = []
        for fg in graph.fusion_groups:
            pattern = tuple(node_types[nid] for nid in fg.nodes)
            fdef = FUSION_BY_PATTERN.get(pattern)
            if fdef:
                resolved.append((fg.nodes, fdef))
        return resolved

    def _check_unknown_types(self, graph: GraphSpec, errors: list[str]):
        for node in graph.nodes:
            if node.type not in NODE_REGISTRY:
                errors.append(f"unknown node type: {node.type} (node {node.id})")

    def _check_required_nodes(self, graph: GraphSpec, errors: list[str]):
        present = {n.type for n in graph.nodes}
        for req in REQUIRED_NODE_TYPES:
            if req not in present:
                errors.append(f"missing required node: {req}")

    def _check_cycles(self, graph: GraphSpec, errors: list[str]):
        # build an adjacency list (unidirectional)
        adj = defaultdict(list)
        for edge in graph.edges:
            adj[edge.from_node].append(edge.to_node)

        curr_path = set()  # current path of nodes
        visited = set()  # fully explored nodes

        # helper to conduct dfs (True means a cycle is present)
        def dfs(nid):
            if nid in visited:
                return False

            curr_path.add(nid)  # add the current node to the path

            # inspect the other to_nodes
            for to_nid in adj[nid]:
                if to_nid in curr_path:
                    return True  # cycle detected
                if dfs(to_nid):
                    return True  # cycle detected

            curr_path.discard(nid)  # remove current node from the path
            visited.add(nid)  # remember dfs-ing current node found no cycles

            return False

        # conduct dfs on all nodes (skipping visited)
        for node in graph.nodes:
            if dfs(node.id):
                errors.append("cycle detected in graph")
                return

    def _check_dangling_edges(self, graph: GraphSpec, errors: list[str]):
        node_ids = {n.id for n in graph.nodes}
        node_ids.add("_input")  # pseudo-node for graph inputs (idx, positions)
        for edge in graph.edges:
            if edge.from_node not in node_ids:
                errors.append(f"edge references nonexistent node: {edge.from_node}")
            if edge.to_node not in node_ids:
                errors.append(f"edge references nonexistent node: {edge.to_node}")

    def _check_port_connections(self, graph: GraphSpec, errors: list[str]):
        # map node id -> node type (e.g. "block_0_qkv" -> "qkv_proj")
        node_types = {n.id: n.type for n in graph.nodes}
        for edge in graph.edges:
            # _input is a pseudo-node with no registry entry, skip it
            if edge.from_node == "_input":
                continue
            # skip edges referencing unknown nodes (caught by _check_dangling_edges)
            if edge.from_node not in node_types or edge.to_node not in node_types:
                continue

            from_type = node_types[edge.from_node]
            to_type = node_types[edge.to_node]

            # skip unknown types (caught by _check_unknown_types)
            if from_type not in NODE_REGISTRY or to_type not in NODE_REGISTRY:
                continue

            from_def = NODE_REGISTRY[from_type]
            to_def = NODE_REGISTRY[to_type]

            # verify the edge's ports actually exist on those node types
            if edge.from_port not in from_def.outputs:
                errors.append(
                    f"node {edge.from_node} ({from_type}) has no output port '{edge.from_port}'"
                )
            if edge.to_port not in to_def.inputs:
                errors.append(f"node {edge.to_node} ({to_type}) has no input port '{edge.to_port}'")

    def _check_quantization(self, graph: GraphSpec, errors: list[str]):
        from server.compiler.quantization_registry import QUANT_MODES, QUANTIZABLE_NODES

        quantized_ids = set()
        for node in graph.nodes:
            if node.quantized is None:
                continue
            if node.quantized not in QUANT_MODES:
                errors.append(
                    f"node {node.id}: invalid quantization mode '{node.quantized}'"
                    f" (must be one of {', '.join(sorted(QUANT_MODES))})"
                )
            if node.type not in QUANTIZABLE_NODES:
                errors.append(
                    f"node {node.id} ({node.type}): cannot be quantized"
                    f" (quantizable types: {', '.join(sorted(QUANTIZABLE_NODES))})"
                )
            quantized_ids.add(node.id)

        # TODO: support quantized weights inside fused kernels (dequantize before tl.dot)
        if quantized_ids and graph.fusion_groups:
            for fg in graph.fusion_groups:
                overlap = quantized_ids & set(fg.nodes)
                if overlap:
                    errors.append(
                        f"node(s) {', '.join(overlap)} cannot be both quantized"
                        " and in a fusion group"
                    )

    def _check_fusion_groups(self, graph: GraphSpec, errors: list[str]):
        if not graph.fusion_groups:
            return

        node_types = {n.id: n.type for n in graph.nodes}

        # build successor map for chain connectivity check
        successors: dict[str, list[str]] = defaultdict(list)
        for edge in graph.edges:
            if edge.from_node == "_input":
                continue
            successors[edge.from_node].append(edge.to_node)

        # check for overlapping nodes across fusion groups
        seen_nodes: set[str] = set()
        for fg in graph.fusion_groups:
            overlap = seen_nodes & set(fg.nodes)
            if overlap:
                overlap_str = ", ".join(overlap)
                label = ", ".join(fg.nodes)
                errors.append(
                    f"fusion [{label}]: node(s) {overlap_str} already in another fusion group"
                )
                continue
            seen_nodes.update(fg.nodes)

        for fg in graph.fusion_groups:
            # resolve kernel from node type pattern
            node_pattern = tuple(node_types[nid] for nid in fg.nodes if nid in node_types)
            if len(node_pattern) != len(fg.nodes):
                missing = [nid for nid in fg.nodes if nid not in node_types]
                for nid in missing:
                    errors.append(f"fusion references unknown node: {nid}")
                continue

            matched = FUSION_BY_PATTERN.get(node_pattern)

            if matched is None:
                types_str = " -> ".join(node_pattern)
                errors.append(f"no fusion kernel matches pattern ({types_str})")
                continue

            kernel_name = matched.cls.__name__

            # check nodes form a connected chain
            chain_ok = True
            for i in range(len(fg.nodes) - 1):
                if fg.nodes[i + 1] not in successors[fg.nodes[i]]:
                    errors.append(
                        f"fusion {kernel_name}: {fg.nodes[i]} -> {fg.nodes[i + 1]} not connected"
                    )
                    chain_ok = False
                    break
            if not chain_ok:
                continue

            # non-final chain nodes must not have external consumers
            chain_set = set(fg.nodes)
            for nid in fg.nodes[:-1]:
                external = [s for s in successors[nid] if s not in chain_set]
                if external:
                    errors.append(
                        f"fusion {kernel_name}: mid-chain node {nid} "
                        f"has external consumer(s) {', '.join(external)}"
                    )

    def _check_optional_warnings(self, graph: GraphSpec, warnings: list[str]):
        present = {n.type for n in graph.nodes}
        if "causal_mask" not in present:
            warnings.append("causal mask removed - model sees future tokens (breaks AR)")
        if "layernorm" not in present:
            warnings.append("no LayerNorm in pipeline - training may destabilize")

        # 3 dropouts per block is normal (attn, resid-attn, resid-mlp) + 1 emb
        n_layer = sum(1 for n in graph.nodes if n.type == "qkv_proj")
        expected_dropout = 3 * n_layer + 1
        dropout_count = sum(1 for n in graph.nodes if n.type == "dropout")
        if dropout_count > expected_dropout + 3:
            warnings.append(f"{dropout_count} dropout nodes - may over-regularize")
