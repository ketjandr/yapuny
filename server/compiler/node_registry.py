from dataclasses import dataclass

try:
    from kernels.fusion.flash_attention import FlashAttention
except ImportError:
    FlashAttention = None
from nodes.attention import (
    AttentionScore,
    CausalMask,
    KVCache,
    OutProjection,
    QKVProjection,
    Softmax,
    ValueWeightedSum,
)
from nodes.dropout import Dropout
from nodes.embeddings import PositionEmbedding, TokenEmbedding
from nodes.head import LMHead
from nodes.mlp import MLPActivation, MLPDown, MLPUp
from nodes.normalization import LayerNorm
from nodes.residual import ResidualAdd


@dataclass
class NodeDef:
    cls: type
    inputs: list[str]
    outputs: list[str]
    build_args: dict[str, str]  # param_name -> expression evaluated against meta


NODE_REGISTRY: dict[str, NodeDef] = {
    "token_embedding": NodeDef(
        cls=TokenEmbedding,
        inputs=["idx"],
        outputs=["out"],
        build_args={"vocab_size": "vocab_size", "n_embd": "n_embd"},
    ),
    "position_embedding": NodeDef(
        cls=PositionEmbedding,
        inputs=["positions"],
        outputs=["out"],
        build_args={"block_size": "block_size", "n_embd": "n_embd"},
    ),
    "qkv_proj": NodeDef(
        cls=QKVProjection,
        inputs=["x"],
        outputs=["q", "k", "v"],
        build_args={"n_embd": "n_embd", "n_head": "n_head"},
    ),
    "kv_cache": NodeDef(
        cls=KVCache,
        inputs=["k", "v"],
        outputs=["k", "v"],
        build_args={},
    ),
    "attention_score": NodeDef(
        cls=AttentionScore,
        inputs=["q", "k"],
        outputs=["out"],
        build_args={"head_dim": "n_embd // n_head"},
    ),
    "causal_mask": NodeDef(
        cls=CausalMask,
        inputs=["x"],
        outputs=["out"],
        build_args={"block_size": "block_size"},
    ),
    "softmax": NodeDef(
        cls=Softmax,
        inputs=["x"],
        outputs=["out"],
        build_args={},
    ),
    "dropout": NodeDef(
        cls=Dropout,
        inputs=["x"],
        outputs=["out"],
        build_args={"p_drop": "dropout"},
    ),
    "value_weighted_sum": NodeDef(
        cls=ValueWeightedSum,
        inputs=["att", "v"],
        outputs=["out"],
        build_args={},
    ),
    "out_proj": NodeDef(
        cls=OutProjection,
        inputs=["x"],
        outputs=["out"],
        build_args={"n_embd": "n_embd"},
    ),
    "residual_add": NodeDef(
        cls=ResidualAdd,
        inputs=["x", "residual"],
        outputs=["out"],
        build_args={},
    ),
    "layernorm": NodeDef(
        cls=LayerNorm,
        inputs=["x"],
        outputs=["out"],
        build_args={"n_embd": "n_embd"},
    ),
    "mlp_up": NodeDef(
        cls=MLPUp,
        inputs=["x"],
        outputs=["out"],
        build_args={"n_embd": "n_embd"},
    ),
    "mlp_activation": NodeDef(
        cls=MLPActivation,
        inputs=["x"],
        outputs=["out"],
        build_args={},
    ),
    "mlp_down": NodeDef(
        cls=MLPDown,
        inputs=["x"],
        outputs=["out"],
        build_args={"n_embd": "n_embd"},
    ),
    "lm_head": NodeDef(
        cls=LMHead,
        inputs=["x"],
        outputs=["out"],
        build_args={"n_embd": "n_embd", "vocab_size": "vocab_size"},
    ),
}

if FlashAttention is not None:
    NODE_REGISTRY["flash_attention"] = NodeDef(
        cls=FlashAttention,
        inputs=["q", "k", "v"],
        outputs=["out"],
        build_args={},
    )


def get_build_kwargs(node_type: str, meta: dict) -> dict:
    """Extract constructor kwargs from graph meta for a given node type."""
    from server.compiler.utils import eval_build_args

    return eval_build_args(NODE_REGISTRY[node_type].build_args, meta)
