// Node catalog: display + port contracts per backend node type (mirror of NODE_REGISTRY).
import type { GraphMetaSchema } from "./types";

export type NodeVariant = "req" | "io";

export type NodeCategory = "embedding" | "attention" | "mlp" | "norm" | "head";

// category -> short badge label + full name + accent color (a token name from tokens.css)
export const CATEGORY: Record<NodeCategory, { label: string; full: string; accent: string }> = {
  embedding: { label: "EMBED", full: "Embedding", accent: "--steel" },
  attention: { label: "ATTN", full: "Attention", accent: "--ice" },
  mlp: { label: "MLP", full: "Multilayer Perceptron", accent: "--amber" },
  norm: { label: "NORM", full: "Normalization", accent: "--green" },
  head: { label: "HEAD", full: "Head", accent: "--violet" },
};

// which family each backend node type belongs to (drives the accent bar + badge)
export const CATEGORY_OF: Record<string, NodeCategory> = {
  token_embedding: "embedding",
  position_embedding: "embedding",
  qkv_proj: "attention",
  kv_cache: "attention",
  attention_score: "attention",
  causal_mask: "attention",
  softmax: "attention",
  value_weighted_sum: "attention",
  out_proj: "attention",
  flash_attention: "attention",
  mlp_up: "mlp",
  mlp_activation: "mlp",
  mlp_down: "mlp",
  layernorm: "norm",
  residual_add: "norm",
  dropout: "norm",
  lm_head: "head",
};

// Symbolic tensor axis (batch dropped): T/S are dynamic, the rest resolve against meta.
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
  subtitle: string | ((meta: GraphMetaSchema) => string); // mono formula under the name
  variant: NodeVariant;
  inputs: PortDef[];
  outputs: PortDef[];
  fusable: boolean; // appears in a fusion pattern -> shows the bottom fusion port
  quantizable: boolean; // has an nn.Linear the backend can quantize -> W8/W4 allowed
  trainingNoop?: boolean; // does nothing during training (e.g. kv_cache) -> greyed in train mode
}

const p = (id: string, shape: Axis[], label = id): PortDef => ({ id, label, shape });

const headDim = (m: GraphMetaSchema) => Math.floor(m.n_embd / m.n_head);

// resolves a node's subtitle (static string, or derived from meta)
export function resolveSubtitle(def: NodeDef, meta: GraphMetaSchema): string {
  return typeof def.subtitle === "function" ? def.subtitle(meta) : def.subtitle;
}

// Ordered roughly by the forward pass so the palette reads top-to-bottom like a GPT.
export const NODE_CATALOG: Record<string, NodeDef> = {
  token_embedding: {
    type: "token_embedding",
    label: "Token Embedding",
    subtitle: "emb = W_tok[idx]",
    variant: "req",
    inputs: [p("idx", ["T"])],
    outputs: [p("out", ["T", "C"], "emb")],
    fusable: false,
    quantizable: false,
  },
  position_embedding: {
    type: "position_embedding",
    label: "Position Embedding",
    subtitle: "emb = W_pos[pos]",
    variant: "req",
    inputs: [p("positions", ["T"], "pos")],
    outputs: [p("out", ["T", "C"], "emb")],
    fusable: false,
    quantizable: false,
  },
  layernorm: {
    type: "layernorm",
    label: "LayerNorm",
    subtitle: "out = γ·norm(x) + β",
    variant: "req",
    inputs: [p("x", ["T", "C"])],
    outputs: [p("out", ["T", "C"])],
    fusable: true,
    quantizable: false,
  },
  qkv_proj: {
    type: "qkv_proj",
    label: "QKV Projection",
    subtitle: "q,k,v = x·W_qkv",
    variant: "req",
    inputs: [p("x", ["T", "C"])],
    outputs: [p("q", ["H", "T", "hd"]), p("k", ["H", "T", "hd"]), p("v", ["H", "T", "hd"])],
    fusable: false,
    quantizable: true,
  },
  kv_cache: {
    type: "kv_cache",
    label: "KV Cache",
    subtitle: "k,v = cat(cache, k,v)",
    variant: "req",
    inputs: [p("k", ["H", "T", "hd"]), p("v", ["H", "T", "hd"])],
    outputs: [p("k", ["H", "S", "hd"]), p("v", ["H", "S", "hd"])],
    fusable: false,
    quantizable: false,
    trainingNoop: true, // pure passthrough during training -> greyed in train mode
  },
  attention_score: {
    type: "attention_score",
    label: "Attention Score",
    subtitle: (m) => `att = q·kᵀ / √${headDim(m)}`,
    variant: "req",
    inputs: [p("q", ["H", "T", "hd"]), p("k", ["H", "S", "hd"])],
    outputs: [p("out", ["H", "T", "S"], "att")],
    fusable: true,
    quantizable: false,
  },
  causal_mask: {
    type: "causal_mask",
    label: "Causal Mask",
    subtitle: "att[j>i] = −∞",
    variant: "req",
    inputs: [p("x", ["H", "T", "S"], "att")],
    outputs: [p("out", ["H", "T", "S"], "att")],
    fusable: true,
    quantizable: false,
  },
  softmax: {
    type: "softmax",
    label: "Softmax",
    subtitle: "att = softmax(att)",
    variant: "req",
    inputs: [p("x", ["H", "T", "S"], "att")],
    outputs: [p("out", ["H", "T", "S"], "att")],
    fusable: true,
    quantizable: false,
  },
  value_weighted_sum: {
    type: "value_weighted_sum",
    label: "Value Sum",
    subtitle: "out = att·v",
    variant: "req",
    inputs: [p("att", ["H", "T", "S"]), p("v", ["H", "S", "hd"])],
    outputs: [p("out", ["H", "T", "hd"])],
    fusable: false,
    quantizable: false,
  },
  out_proj: {
    type: "out_proj",
    label: "Out Projection",
    subtitle: "out = merge(x)·W_o",
    variant: "req",
    inputs: [p("x", ["H", "T", "hd"])],
    outputs: [p("out", ["T", "C"])],
    fusable: false,
    quantizable: true,
  },
  flash_attention: {
    type: "flash_attention",
    label: "Flash Attention",
    subtitle: (m) => `out = softmax(q·kᵀ/√${headDim(m)})·v`,
    variant: "req",
    inputs: [p("q", ["H", "T", "hd"]), p("k", ["H", "S", "hd"]), p("v", ["H", "S", "hd"])],
    outputs: [p("out", ["H", "T", "hd"])],
    fusable: false,
    quantizable: false,
  },
  residual_add: {
    type: "residual_add",
    label: "Residual Add",
    subtitle: "out = x + resid",
    variant: "req",
    inputs: [p("x", ["T", "C"]), p("residual", ["T", "C"], "resid")],
    outputs: [p("out", ["T", "C"])],
    fusable: true,
    quantizable: false,
  },
  dropout: {
    type: "dropout",
    label: "Dropout",
    subtitle: "out = drop(x, p)",
    variant: "req",
    inputs: [p("x", ["T", "C"])],
    outputs: [p("out", ["T", "C"])],
    fusable: true,
    quantizable: false,
  },
  mlp_up: {
    type: "mlp_up",
    label: "MLP Up",
    subtitle: "out = x·W_up",
    variant: "req",
    inputs: [p("x", ["T", "C"])],
    outputs: [p("out", ["T", "4C"])],
    fusable: true,
    quantizable: true,
  },
  mlp_activation: {
    type: "mlp_activation",
    label: "GELU",
    subtitle: "out = gelu(x)",
    variant: "req",
    inputs: [p("x", ["T", "4C"])],
    outputs: [p("out", ["T", "4C"])],
    fusable: true,
    quantizable: false,
  },
  mlp_down: {
    type: "mlp_down",
    label: "MLP Down",
    subtitle: "out = x·W_down",
    variant: "req",
    inputs: [p("x", ["T", "4C"])],
    outputs: [p("out", ["T", "C"])],
    fusable: true,
    quantizable: true,
  },
  lm_head: {
    type: "lm_head",
    label: "LM Head",
    subtitle: "logits = x·W_vocab",
    variant: "req",
    inputs: [p("x", ["T", "C"])],
    outputs: [p("out", ["T", "V"], "logits")],
    fusable: false,
    quantizable: true,
  },
};

export const CATALOG_ORDER = Object.keys(NODE_CATALOG);

// UI-only pseudo-nodes for the graph endpoints; not in NODE_REGISTRY/palette.
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
export const OUTPUT_NODE: NodeDef = {
  type: "_output",
  label: "Output",
  subtitle: "logits",
  variant: "io",
  inputs: [p("logits", ["T", "V"])],
  outputs: [],
  fusable: false,
  quantizable: false,
};

export function getNodeDef(type: string): NodeDef | undefined {
  return NODE_CATALOG[type];
}

// Resolve any renderable node def, including the _input / _output pseudo-nodes.
export function resolveNodeDef(type: string): NodeDef | undefined {
  if (type === "_input") return INPUT_NODE;
  if (type === "_output") return OUTPUT_NODE;
  return NODE_CATALOG[type];
}

// --- geometry: height grows with port count, width fits the resolved labels ---
export const NODE_HEADER_H = 50; // header + subtitle band
export const PORT_ROW_H = 30; // vertical space per port row
const NODE_MIN_W = 120;
const SHAPE_CHAR_W = 4.5; // px per char of the 7px mono shape line
const HEAD_CHAR_W = 8.4; // px per char of the 11px uppercase header (with tracking)
const SUB_CHAR_W = 5.7; // px per char of the 8.5px mono subtitle formula (with tracking)

function widestLabel(ports: PortDef[], meta: GraphMetaSchema): number {
  if (ports.length === 0) return 0;
  // size against train mode - it resolves T/S to numbers, the widest the labels ever get
  return Math.max(...ports.map((p) => formatShape(p.shape, meta, "train").length * SHAPE_CHAR_W));
}

// Content-driven width: fits both label stacks side by side, the header, and the subtitle.
export function nodeWidth(def: NodeDef, meta: GraphMetaSchema): number {
  const labels = 44 + widestLabel(def.inputs, meta) + widestLabel(def.outputs, meta);
  const header = 26 + (def.quantizable ? 30 : 0) + def.label.length * HEAD_CHAR_W;
  const subtitle = 24 + resolveSubtitle(def, meta).length * SUB_CHAR_W;
  return Math.round(Math.max(NODE_MIN_W, labels, header, subtitle));
}

export function nodeHeight(def: NodeDef): number {
  const rows = Math.max(def.inputs.length, def.outputs.length, 1);
  return NODE_HEADER_H + rows * PORT_ROW_H; // exactly fits the port rows (no clamp)
}

// Fixed row grid (px from card top) so ports line up across nodes regardless of height.
export function portTop(index: number): number {
  return NODE_HEADER_H + PORT_ROW_H * (index + 0.5);
}
