from collections import defaultdict
from dataclasses import dataclass

from server.compiler.registry import NODE_REGISTRY
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
        errors = []
        warnings = []

        self._check_unknown_types(graph, errors)
        self._check_required_nodes(graph, errors)
        self._check_cycles(graph, errors)
        self._check_dangling_edges(graph, errors)
        self._check_port_connections(graph, errors)
        self._check_optional_warnings(graph, warnings)

        return ValidationResult(errors=errors, warnings=warnings)

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
        adjacency_list = defaultdict(list)
        for edge in graph.edges:
            adjacency_list[edge.from_node].append(edge.to_node)

        curr_path = set()  # current path of nodes
        visited = set()  # fully explored nodes

        # helper to conduct dfs (True means a cycle is present)
        def dfs(node_id):
            if node_id in visited:
                return False

            curr_path.add(node_id)  # add the current node to the path

            # inspect the other to_nodes
            for to_node_id in adjacency_list[node_id]:
                if to_node_id in curr_path:
                    return True  # cycle detected
                if dfs(to_node_id):
                    return True  # cycle detected

            curr_path.discard(node_id)  # remove current node from the path
            visited.add(node_id)  # remember dfs-ing current node found no cycles

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
