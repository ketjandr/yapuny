"""Builds the default GPT graph spec."""


def make_block_nodes(block_idx: int) -> list[dict]:
    """Generate node specs for one transformer block."""
    b = f"b{block_idx}"
    return [
        {"id": f"{b}_ln1", "type": "layernorm"},
        {"id": f"{b}_qkv", "type": "qkv_proj"},
        {"id": f"{b}_kvcache", "type": "kv_cache"},
        {"id": f"{b}_score", "type": "attention_score"},
        {"id": f"{b}_mask", "type": "causal_mask"},
        {"id": f"{b}_softmax", "type": "softmax"},
        {"id": f"{b}_attn_drop", "type": "dropout"},
        {"id": f"{b}_vws", "type": "value_weighted_sum"},
        {"id": f"{b}_out_proj", "type": "out_proj"},
        {"id": f"{b}_resid_drop1", "type": "dropout"},
        {"id": f"{b}_res1", "type": "residual_add"},
        {"id": f"{b}_ln2", "type": "layernorm"},
        {"id": f"{b}_mlp_up", "type": "mlp_up"},
        {"id": f"{b}_mlp_act", "type": "mlp_activation"},
        {"id": f"{b}_mlp_down", "type": "mlp_down"},
        {"id": f"{b}_resid_drop2", "type": "dropout"},
        {"id": f"{b}_res2", "type": "residual_add"},
    ]


def make_block_edges(block_idx: int, input_node: str, input_port: str) -> list[dict]:
    """Generate edge specs for one transformer block.

    input_node/input_port: where this block gets its input from
    (either emb_drop or previous block's res2).
    """
    b = f"b{block_idx}"
    return [
        # ln1 takes input
        {"from_node": input_node, "from_port": input_port, "to_node": f"{b}_ln1", "to_port": "x"},
        # qkv takes ln1 output
        {"from_node": f"{b}_ln1", "from_port": "out", "to_node": f"{b}_qkv", "to_port": "x"},
        # kv_cache takes q, k from qkv
        {"from_node": f"{b}_qkv", "from_port": "k", "to_node": f"{b}_kvcache", "to_port": "k"},
        {"from_node": f"{b}_qkv", "from_port": "v", "to_node": f"{b}_kvcache", "to_port": "v"},
        # attention score takes q from qkv, k from kv_cache
        {"from_node": f"{b}_qkv", "from_port": "q", "to_node": f"{b}_score", "to_port": "q"},
        {"from_node": f"{b}_kvcache", "from_port": "k", "to_node": f"{b}_score", "to_port": "k"},
        # mask -> softmax -> dropout
        {"from_node": f"{b}_score", "from_port": "out", "to_node": f"{b}_mask", "to_port": "x"},
        {"from_node": f"{b}_mask", "from_port": "out", "to_node": f"{b}_softmax", "to_port": "x"},
        {
            "from_node": f"{b}_softmax",
            "from_port": "out",
            "to_node": f"{b}_attn_drop",
            "to_port": "x",
        },
        # value weighted sum takes att from dropout, v from kv_cache
        {
            "from_node": f"{b}_attn_drop",
            "from_port": "out",
            "to_node": f"{b}_vws",
            "to_port": "att",
        },
        {"from_node": f"{b}_kvcache", "from_port": "v", "to_node": f"{b}_vws", "to_port": "v"},
        # out projection + residual dropout
        {"from_node": f"{b}_vws", "from_port": "out", "to_node": f"{b}_out_proj", "to_port": "x"},
        {
            "from_node": f"{b}_out_proj",
            "from_port": "out",
            "to_node": f"{b}_resid_drop1",
            "to_port": "x",
        },
        # residual add: attn output + skip from input
        {
            "from_node": f"{b}_resid_drop1",
            "from_port": "out",
            "to_node": f"{b}_res1",
            "to_port": "x",
        },
        {
            "from_node": input_node,
            "from_port": input_port,
            "to_node": f"{b}_res1",
            "to_port": "residual",
        },
        # mlp subgraph
        {"from_node": f"{b}_res1", "from_port": "out", "to_node": f"{b}_ln2", "to_port": "x"},
        {"from_node": f"{b}_ln2", "from_port": "out", "to_node": f"{b}_mlp_up", "to_port": "x"},
        {"from_node": f"{b}_mlp_up", "from_port": "out", "to_node": f"{b}_mlp_act", "to_port": "x"},
        {
            "from_node": f"{b}_mlp_act",
            "from_port": "out",
            "to_node": f"{b}_mlp_down",
            "to_port": "x",
        },
        {
            "from_node": f"{b}_mlp_down",
            "from_port": "out",
            "to_node": f"{b}_resid_drop2",
            "to_port": "x",
        },
        # residual add: mlp output + skip from res1
        {
            "from_node": f"{b}_resid_drop2",
            "from_port": "out",
            "to_node": f"{b}_res2",
            "to_port": "x",
        },
        {
            "from_node": f"{b}_res1",
            "from_port": "out",
            "to_node": f"{b}_res2",
            "to_port": "residual",
        },
    ]


def default_gpt_graph(
    n_layer: int = 6,
    n_head: int = 6,
    n_embd: int = 384,
    block_size: int = 256,
    vocab_size: int = 8000,
    dropout: float = 0.1,
) -> dict:
    nodes = [
        {"id": "tok_emb", "type": "token_embedding"},
        {"id": "pos_emb", "type": "position_embedding"},
        {"id": "emb_add", "type": "residual_add"},
        {"id": "emb_drop", "type": "dropout"},
    ]

    edges = [
        {"from_node": "_input", "from_port": "idx", "to_node": "tok_emb", "to_port": "idx"},
        {
            "from_node": "_input",
            "from_port": "positions",
            "to_node": "pos_emb",
            "to_port": "positions",
        },
        {"from_node": "tok_emb", "from_port": "out", "to_node": "emb_add", "to_port": "x"},
        {"from_node": "pos_emb", "from_port": "out", "to_node": "emb_add", "to_port": "residual"},
        {"from_node": "emb_add", "from_port": "out", "to_node": "emb_drop", "to_port": "x"},
    ]

    # add transformer blocks
    for i in range(n_layer):
        nodes.extend(make_block_nodes(i))
        if i == 0:
            input_node, input_port = "emb_drop", "out"
        else:
            input_node, input_port = f"b{i - 1}_res2", "out"
        edges.extend(make_block_edges(i, input_node, input_port))

    # final layernorm + lm_head
    last_block = f"b{n_layer - 1}_res2"
    nodes.append({"id": "ln_f", "type": "layernorm"})
    nodes.append({"id": "lm_head", "type": "lm_head"})
    edges.append({"from_node": last_block, "from_port": "out", "to_node": "ln_f", "to_port": "x"})
    edges.append({"from_node": "ln_f", "from_port": "out", "to_node": "lm_head", "to_port": "x"})

    return {
        "nodes": nodes,
        "edges": edges,
        "fusion_groups": [],
        "meta": {
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
            "block_size": block_size,
            "vocab_size": vocab_size,
            "dropout": dropout,
        },
    }


def blocked_gpt_graph(n_layer: int = 6, **meta) -> dict:
    """The compact form: one logical block + a block spec the compiler unrolls n_layer
    times (vs default_gpt_graph's explicit N-block stack). Same architecture either way."""
    g = default_gpt_graph(n_layer=1, **meta)
    g["block"] = {"nodes": [n["id"] for n in make_block_nodes(0)]}
    g["meta"]["n_layer"] = n_layer
    return g
