from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NodeSpec:
    id: str
    type: str
    config: dict = field(default_factory=dict)
    quantized: str | None = None  # "w8" or "w4"


@dataclass
class EdgeSpec:
    from_node: str
    to_node: str
    from_port: str = "out"
    to_port: str = "x"


@dataclass
class FusionGroup:
    nodes: list[str]


@dataclass
class GraphMeta:
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    dropout: float = 0.1
    vocab_size: int = 8000


@dataclass
class GraphSpec:
    nodes: list[NodeSpec]
    edges: list[EdgeSpec]
    fusion_groups: list[FusionGroup] = field(default_factory=list)
    meta: GraphMeta = field(default_factory=GraphMeta)

    @classmethod
    def from_dict(cls, data: dict) -> GraphSpec:
        nodes = [NodeSpec(**n) for n in data["nodes"]]
        edges = [EdgeSpec(**e) for e in data["edges"]]
        fusion_groups = [FusionGroup(nodes=fg["nodes"]) for fg in data.get("fusion_groups", [])]
        meta = GraphMeta(**data.get("meta", {}))
        return cls(nodes=nodes, edges=edges, fusion_groups=fusion_groups, meta=meta)
