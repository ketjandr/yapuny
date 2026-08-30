from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BaseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


# -- Graph --


class NodeSchema(BaseSchema):
    id: str
    type: str
    config: dict = Field(default_factory=dict)
    quantized: str | None = None


class EdgeSchema(BaseSchema):
    from_node: str
    to_node: str
    from_port: str = "out"
    to_port: str = "x"


class FusionGroupSchema(BaseSchema):
    nodes: list[str]


class GraphMetaSchema(BaseSchema):
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    dropout: float = 0.1
    vocab_size: int = 8000


class GraphRequest(BaseSchema):
    nodes: list[NodeSchema]
    edges: list[EdgeSchema]
    fusion_groups: list[FusionGroupSchema] = Field(default_factory=list)
    meta: GraphMetaSchema = Field(default_factory=GraphMetaSchema)


# -- Data --


class PrepareDataRequest(BaseSchema):
    vocab_size: int = 8000
    val_fraction: float = 0.1


# -- Training --


class TrainRequest(BaseSchema):
    max_steps: int = 2000
    batch_size: int = 32
    learning_rate: float = 3e-4
    eval_interval: int = 200
    eval_iters: int = 50
    checkpoint_path: str | None = None
    bench: bool = False


# -- Generate --


class GenerateRequest(BaseSchema):
    prompt_ids: list[int]
    max_new_tokens: int = 50
    temperature: float = 1.0
    top_k: int | None = None
    bench: bool = False


class DecodeRequest(BaseSchema):
    token_ids: list[int]


# -- Benchmark --


class ProfileRequest(BaseSchema):
    mode: str = "decode"
    prompt_tokens: int = 64
    new_tokens: int = 128
    warmup: int = 3


class BenchRunRequest(BaseSchema):
    graphs: list[GraphRequest] = Field(min_length=1, max_length=5)
    prompt_ids: list[int]
    max_new_tokens: int = 50
    temperature: float = 1.0
    top_k: int | None = None
