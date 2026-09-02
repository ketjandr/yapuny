// Node catalog: display + port contracts per backend node type (mirror of NODE_REGISTRY).
import type { GraphMetaSchema } from "./types";

export type NodeVariant = "req" | "opt" | "flash" | "io";

// Symbolic tensor axis, batch dropped. T/S are dynamic; the rest resolve against meta.
// Canonical per-type shapes - non-standard placements would need real shape inference.
export type Axis = "T" | "S" | "C" | "4C" | "H" | "hd" | "V";

type ShapeMode = "train" | "inference";

export function resolveAxis(axis: Axis, meta: GraphMetaSchema, mode: ShapeMode): string {
  switch (axis) {
    case "T":
    case "S":
      // train/prefill runs the full window, so T = S = block_size; inference stays symbolic
      return mode === "train" ? String(meta.block_size) : axis;
    case "C":
      return String(meta.n_embd);
    case "4C":
      return String(4 * meta.n_embd);
    case "H":
      return String(meta.n_head);
    case "hd":
      return String(Math.floor(meta.n_embd / meta.n_head));
    case "V":
      return String(meta.vocab_size);
  }
}

// ["H","T","hd"] + default meta -> "(6, T, 64)" inference, "(6, 256, 64)" train
export function formatShape(shape: Axis[], meta: GraphMetaSchema, mode: ShapeMode): string {
  return `(${shape.map((a) => resolveAxis(a, meta, mode)).join(", ")})`;
}

export interface PortDef {
  id: string; // the backend port name (edge from_port / to_port)
  label: string; // short role label shown at the handle
  shape: Axis[]; // symbolic axes, batch dropped; resolved against meta for display
}

export interface NodeDef {
  type: string; // backend node type key
  label: string; // uppercase display name
  subtitle: string; // mono caption under the name
  variant: NodeVariant;
  inputs: PortDef[];
  outputs: PortDef[];
  fusable: boolean; // appears in a fusion pattern -> shows the bottom fusion port
  quantizable: boolean; // has an nn.Linear the backend can quantize -> W8/W4 allowed
  badge?: string; // small corner badge (e.g. "flash")
  width?: number; // card width override (px); default derived below
  trainingNoop?: boolean; // does nothing during training (e.g. kv_cache) -> greyed in train mode
}

const p = (id: string, shape: Axis[], label = id): PortDef => ({ id, label, shape });

// Ordered roughly by the forward pass so the palette reads top-to-bottom like a GPT.
export const NODE_CATALOG: Record<string, NodeDef> = {
  token_embedding: {
    type: "token_embedding",
    label: "Token Emb",
    subtitle: "vocab × embd",
    variant: "req",
    inputs: [p("idx", ["T"])],
    outputs: [p("out", ["T", "C"], "emb")],
    fusable: false,
    quantizable: false,
  },
  position_embedding: {
    type: "position_embedding",
    label: "Position Emb",
    subtitle: "block × embd",
    variant: "req",
    inputs: [p("positions", ["T"], "pos")],
    outputs: [p("out", ["T", "C"], "emb")],
    fusable: false,
    quantizable: false,
  },
  layernorm: {
    type: "layernorm",
    label: "LayerNorm",
    subtitle: "γ, β",
    variant: "req",
    inputs: [p("x", ["T", "C"])],
    outputs: [p("out", ["T", "C"])],
    fusable: true,
    quantizable: false,
  },
  qkv_proj: {
    type: "qkv_proj",
    label: "QKV Proj",
    subtitle: "linear → q k v",
    variant: "req",
    inputs: [p("x", ["T", "C"])],
    outputs: [p("q", ["H", "T", "hd"]), p("k", ["H", "T", "hd"]), p("v", ["H", "T", "hd"])],
    fusable: false,
    quantizable: true,
  },
  kv_cache: {
    type: "kv_cache",
    label: "KV Cache",
    subtitle: "decode reuse",
    variant: "req",
    inputs: [p("k", ["H", "T", "hd"]), p("v", ["H", "T", "hd"])],
    outputs: [p("k", ["H", "S", "hd"]), p("v", ["H", "S", "hd"])],
    fusable: false,
    quantizable: false,
    trainingNoop: true, // pure passthrough during training -> greyed in train mode
  },
  attention_score: {
    type: "attention_score",
    label: "Attn Score",
    subtitle: "QKᵀ / √d",
    variant: "req",
    inputs: [p("q", ["H", "T", "hd"]), p("k", ["H", "S", "hd"])],
    outputs: [p("out", ["H", "T", "S"], "att")],
    fusable: true,
    quantizable: false,
  },
  causal_mask: {
    type: "causal_mask",
    label: "Causal Mask",
    subtitle: "tril −∞",
    variant: "req",
    inputs: [p("x", ["H", "T", "S"], "att")],
    outputs: [p("out", ["H", "T", "S"], "att")],
    fusable: true,
    quantizable: false,
  },
  softmax: {
    type: "softmax",
    label: "Softmax",
    subtitle: "row-wise",
    variant: "req",
    inputs: [p("x", ["H", "T", "S"], "att")],
    outputs: [p("out", ["H", "T", "S"], "att")],
    fusable: true,
    quantizable: false,
  },
  value_weighted_sum: {
    type: "value_weighted_sum",
    label: "Value Sum",
    subtitle: "att · V",
    variant: "req",
    inputs: [p("att", ["H", "T", "S"]), p("v", ["H", "S", "hd"])],
    outputs: [p("out", ["H", "T", "hd"])],
    fusable: false,
    quantizable: false,
  },
  out_proj: {
    type: "out_proj",
    label: "Out Proj",
    subtitle: "merge heads · linear",
    variant: "req",
    inputs: [p("x", ["H", "T", "hd"])],
    outputs: [p("out", ["T", "C"])],
    fusable: false,
    quantizable: true,
  },
  flash_attention: {
    type: "flash_attention",
    label: "Flash Attention",
    subtitle: "QKᵀ · softmax · V",
    variant: "flash",
    inputs: [p("q", ["H", "T", "hd"]), p("k", ["H", "S", "hd"]), p("v", ["H", "S", "hd"])],
    outputs: [p("out", ["H", "T", "hd"])],
    fusable: false,
    quantizable: false,
    badge: "flash",
    width: 200,
  },
  residual_add: {
    type: "residual_add",
    label: "Residual Add",
    subtitle: "skip + add",
    variant: "req",
    inputs: [p("x", ["T", "C"]), p("residual", ["T", "C"], "resid")],
    outputs: [p("out", ["T", "C"])],
    fusable: true,
    quantizable: false,
  },
  dropout: {
    type: "dropout",
    label: "Dropout",
    subtitle: "p_drop",
    variant: "req",
    inputs: [p("x", ["T", "C"])],
    outputs: [p("out", ["T", "C"])],
    fusable: true,
    quantizable: false,
  },
  mlp_up: {
    type: "mlp_up",
    label: "MLP Up",
    subtitle: "embd → 4·embd",
    variant: "req",
    inputs: [p("x", ["T", "C"])],
    outputs: [p("out", ["T", "4C"])],
    fusable: true,
    quantizable: true,
  },
  mlp_activation: {
    type: "mlp_activation",
    label: "GELU",
    subtitle: "activation",
    variant: "req",
    inputs: [p("x", ["T", "4C"])],
    outputs: [p("out", ["T", "4C"])],
    fusable: true,
    quantizable: false,
  },
  mlp_down: {
    type: "mlp_down",
    label: "MLP Down",
    subtitle: "4·embd → embd",
    variant: "req",
    inputs: [p("x", ["T", "4C"])],
    outputs: [p("out", ["T", "C"])],
    fusable: true,
    quantizable: true,
  },
  lm_head: {
    type: "lm_head",
    label: "LM Head",
    subtitle: "linear → vocab",
    variant: "req",
    inputs: [p("x", ["T", "C"])],
    outputs: [p("out", ["T", "V"], "logits")],
    fusable: false,
    quantizable: true,
  },
};

export const CATALOG_ORDER = Object.keys(NODE_CATALOG);

// UI-only pseudo-node for the compiler-seeded graph input; not in NODE_REGISTRY/palette.
export const INPUT_NODE: NodeDef = {
  type: "_input",
  label: "Input",
  subtitle: "idx · positions",
  variant: "io",
  inputs: [],
  outputs: [p("idx", ["T"]), p("positions", ["T"], "pos")],
  fusable: false,
  quantizable: false,
};

export function getNodeDef(type: string): NodeDef | undefined {
  return NODE_CATALOG[type];
}

// Resolve any renderable node def, including the _input pseudo-node.
export function resolveNodeDef(type: string): NodeDef | undefined {
  if (type === "_input") return INPUT_NODE;
  return NODE_CATALOG[type];
}

// --- geometry: skinny cards, height grows with port count ---
export const NODE_HEADER_H = 40; // header + subtitle band
export const PORT_ROW_H = 30; // vertical space per port row
const NODE_MIN_H = 96;
const NODE_W = 132; // skinny fixed width

export function nodeWidth(def: NodeDef): number {
  return def.width ?? NODE_W;
}

export function nodeHeight(def: NodeDef): number {
  const rows = Math.max(def.inputs.length, def.outputs.length, 1);
  return Math.max(NODE_MIN_H, NODE_HEADER_H + rows * PORT_ROW_H);
}

// Vertical center (px from card top) for the i-th port of a side with `count` ports.
export function portTop(def: NodeDef, index: number, count: number): number {
  const h = nodeHeight(def);
  const bandTop = NODE_HEADER_H;
  const band = h - bandTop;
  return bandTop + (band * (index + 0.5)) / count;
}
