from __future__ import annotations

from dataclasses import dataclass

try:
    from kernels.fusion.fused_dropout_residual import FusedDropoutResidual
    from kernels.fusion.fused_dropout_residual_layernorm import FusedDropoutResidualLayerNorm
    from kernels.fusion.fused_linear_dropout import FusedLinearDropout
    from kernels.fusion.fused_linear_dropout_residual import FusedLinearDropoutResidual
    from kernels.fusion.fused_linear_gelu import FusedLinearGELU
    from kernels.fusion.fused_residual_layernorm import FusedResidualLayerNorm
    from kernels.fusion.fused_scale_mask_softmax import FusedScaleMaskSoftmax

    FUSION_AVAILABLE = True
except ImportError:
    FUSION_AVAILABLE = False


@dataclass
class FusionDef:
    pattern: tuple[str, ...]
    cls: type
    inputs: list[str]
    outputs: list[str]
    build_args: list[str]


if FUSION_AVAILABLE:
    FUSION_REGISTRY: list[FusionDef] = [
        FusionDef(
            pattern=("residual_add", "layernorm"),
            cls=FusedResidualLayerNorm,
            inputs=["x", "residual"],
            outputs=["out"],
            build_args=["n_embd"],
        ),
        FusionDef(
            pattern=("attention_score", "causal_mask", "softmax"),
            cls=FusedScaleMaskSoftmax,
            inputs=["q", "k"],
            outputs=["out"],
            build_args=["head_dim"],
        ),
        FusionDef(
            pattern=("mlp_up", "mlp_activation"),
            cls=FusedLinearGELU,
            inputs=["x"],
            outputs=["out"],
            build_args=["n_embd"],
        ),
        FusionDef(
            pattern=("dropout", "residual_add"),
            cls=FusedDropoutResidual,
            inputs=["x", "residual"],
            outputs=["out"],
            build_args=["dropout"],
        ),
        FusionDef(
            pattern=("mlp_down", "dropout"),
            cls=FusedLinearDropout,
            inputs=["x"],
            outputs=["out"],
            build_args=["n_embd", "dropout"],
        ),
        FusionDef(
            pattern=("dropout", "residual_add", "layernorm"),
            cls=FusedDropoutResidualLayerNorm,
            inputs=["x", "residual"],
            outputs=["out"],
            build_args=["n_embd", "dropout"],
        ),
        FusionDef(
            pattern=("mlp_down", "dropout", "residual_add"),
            cls=FusedLinearDropoutResidual,
            inputs=["x", "residual"],
            outputs=["out"],
            build_args=["n_embd", "dropout"],
        ),
    ]
else:
    FUSION_REGISTRY = []

FUSION_BY_KERNEL: dict[str, FusionDef] = {f.cls.__name__: f for f in FUSION_REGISTRY}


def detect_fusion_groups(
    topo_order: list[str],
    node_types: dict[str, str],
    edges: list[tuple[str, str, str, str]],
) -> list[tuple[list[str], FusionDef]]:
    """Auto-detect contiguous node sequences that match a fusion pattern."""
    # build successor map: node_id -> [(next_node_id, from_port, to_port)]
    successors: dict[str, list[str]] = {}
    for from_id, from_port, to_id, to_port in edges:
        if from_id == "_input":
            continue
        successors.setdefault(from_id, []).append(to_id)

    # build predecessor count
    in_count: dict[str, int] = {}
    for from_id, from_port, to_id, to_port in edges:
        if from_id == "_input":
            continue
        in_count[to_id] = in_count.get(to_id, 0) + 1

    # track which nodes are already fused
    fused_nodes: set[str] = set()
    groups: list[tuple[list[str], FusionDef]] = []

    # try longer patterns first (greedy)
    sorted_registry = sorted(FUSION_REGISTRY, key=lambda f: len(f.pattern), reverse=True)

    for start_idx, start_node in enumerate(topo_order):
        if start_node in fused_nodes:
            continue

        for fdef in sorted_registry:
            pattern = fdef.pattern
            if node_types.get(start_node) != pattern[0]:
                continue

            # try to match the full pattern as a linear chain
            chain = [start_node]
            current = start_node
            matched = True

            for i in range(1, len(pattern)):
                # current node must have exactly one successor in the chain
                succs = successors.get(current, [])
                next_node = None
                for s in succs:
                    if node_types.get(s) == pattern[i] and s not in fused_nodes:
                        # successor must have no other inputs from outside the chain
                        # (except residual inputs which are external by design)
                        next_node = s
                        break

                if next_node is None:
                    matched = False
                    break

                chain.append(next_node)
                current = next_node

            if matched and len(chain) == len(pattern):
                # verify no chain node is already fused
                if not any(n in fused_nodes for n in chain):
                    groups.append((chain, fdef))
                    fused_nodes.update(chain)
                    break

    return groups


def apply_fusion(
    groups: list[tuple[list[str], FusionDef]],
    modules: dict[str, object],
    node_types: dict[str, str],
    topo_order: list[str],
    edges: list[tuple[str, str, str, str]],
    node_outputs: dict[str, list[str]],
    meta: dict,
) -> tuple[dict, dict, list, list, dict]:
    """Replace fused node groups with single fused modules, rewire edges."""
    new_modules = dict(modules)
    new_types = dict(node_types)
    new_topo = list(topo_order)
    new_edges = list(edges)
    new_outputs = dict(node_outputs)

    for chain, fdef in groups:
        fused_id = "_fused_" + "_".join(chain)
        first_node = chain[0]
        last_node = chain[-1]

        # build kwargs for the fused kernel
        derived = {**meta, "head_dim": meta["n_embd"] // meta["n_head"]}
        kwargs = {arg: derived[arg] for arg in fdef.build_args}

        # special handling for linear-based fusions that need in/out features
        if hasattr(fdef.cls, '__init__'):
            import inspect
            sig = inspect.signature(fdef.cls.__init__)
            params = list(sig.parameters.keys())
            if 'in_features' in params:
                kwargs['in_features'] = meta['n_embd']
            if 'out_features' in params:
                # mlp_up expands 4x, mlp_down contracts back
                if 'mlp_up' in fdef.pattern:
                    kwargs['out_features'] = 4 * meta['n_embd']
                else:
                    kwargs['out_features'] = meta['n_embd']
            if 'p_drop' in params:
                kwargs['p_drop'] = meta.get('dropout', 0.1)
                kwargs.pop('dropout', None)

        # instantiate fused module
        new_modules[fused_id] = fdef.cls(**kwargs)
        new_types[fused_id] = fused_id

        new_outputs[fused_id] = list(fdef.outputs)

        # remove old modules
        for nid in chain:
            del new_modules[nid]
            del new_types[nid]
            del new_outputs[nid]

        # rewire edges: inputs to first node -> inputs to fused node
        rewired_edges = []
        for from_id, from_port, to_id, to_port in new_edges:
            if to_id == first_node and from_id not in chain:
                rewired_edges.append((from_id, from_port, fused_id, to_port))
            elif to_id in chain and from_id in chain:
                continue  # internal edge, drop it
            elif to_id in chain and from_id not in chain:
                # external input to a middle/last node (e.g. residual)
                rewired_edges.append((from_id, from_port, fused_id, to_port))
            elif from_id == last_node and to_id not in chain:
                rewired_edges.append((fused_id, "out", to_id, to_port))
            elif from_id in chain and to_id not in chain:
                # output from middle node to outside - route through fused
                rewired_edges.append((fused_id, "out", to_id, to_port))
            else:
                rewired_edges.append((from_id, from_port, to_id, to_port))

        new_edges = rewired_edges

        # fix topo order: replace chain with fused node
        first_idx = new_topo.index(first_node)
        for nid in chain:
            new_topo.remove(nid)
        new_topo.insert(first_idx, fused_id)

    return new_modules, new_types, new_topo, new_edges, new_outputs
