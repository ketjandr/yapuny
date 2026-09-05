// Seed graph: one pre-LN transformer block (embeddings -> attention -> MLP -> LM head).
import { nodeWidth, resolveNodeDef } from "./nodeCatalog";
import type { GraphMetaSchema } from "./types";

export interface PlacedNode {
  id: string;
  type: string;
  x: number;
  y: number;
}

export interface SeedEdge {
  from: string;
  fromPort: string;
  to: string;
  toPort: string;
}

export const DEFAULT_META: GraphMetaSchema = {
  n_layer: 6,
  n_head: 6,
  n_embd: 384,
  block_size: 256,
  dropout: 0.1,
  vocab_size: 8000,
};

const COL_GAP = 50; // uniform horizontal gap between stages (x is width-driven, not fixed)
const ROW = 130; // main pipeline row (node top y)

function typeWidth(type: string): number {
  return nodeWidth(resolveNodeDef(type)!, DEFAULT_META);
}

// The main pipeline as ordered stages (columns); x is computed from cumulative widths so
// gaps stay even regardless of each node's content width. tok/pos embed share a column.
const STAGES: { id: string; type: string; y: number }[][] = [
  [
    { id: "tok_emb", type: "token_embedding", y: ROW - 55 },
    { id: "pos_emb", type: "position_embedding", y: ROW + 85 },
  ],
  [{ id: "emb_add", type: "residual_add", y: ROW }],
  [{ id: "emb_drop", type: "dropout", y: ROW }],
  [{ id: "ln1", type: "layernorm", y: ROW }],
  [{ id: "qkv", type: "qkv_proj", y: ROW }],
  [{ id: "attn", type: "attention_score", y: ROW }],
  [{ id: "mask", type: "causal_mask", y: ROW }],
  [{ id: "smax", type: "softmax", y: ROW }],
  [{ id: "vsum", type: "value_weighted_sum", y: ROW }],
  [{ id: "oproj", type: "out_proj", y: ROW }],
  [{ id: "attn_drop", type: "dropout", y: ROW }],
  [{ id: "res1", type: "residual_add", y: ROW }],
  [{ id: "ln2", type: "layernorm", y: ROW }],
  [{ id: "mlp_up", type: "mlp_up", y: ROW }],
  [{ id: "gelu", type: "mlp_activation", y: ROW }],
  [{ id: "mlp_down", type: "mlp_down", y: ROW }],
  [{ id: "mlp_drop", type: "dropout", y: ROW }],
  [{ id: "res2", type: "residual_add", y: ROW }],
  [{ id: "lnf", type: "layernorm", y: ROW }],
  [{ id: "lm_head", type: "lm_head", y: ROW }],
];

function buildLayout(): { nodes: PlacedNode[]; endX: number } {
  const nodes: PlacedNode[] = [];
  const kvW = typeWidth("kv_cache");
  let x = 0;
  for (const stage of STAGES) {
    for (const n of stage) nodes.push({ id: n.id, type: n.type, x, y: n.y });
    let advance = Math.max(...stage.map((n) => typeWidth(n.type))) + COL_GAP;
    if (stage.some((n) => n.id === "qkv")) advance += kvW + COL_GAP; // room for kv below
    x += advance;
  }
  // kv cache: a branch below the main row, centered in the widened qkv->attn gap
  const qkv = nodes.find((n) => n.id === "qkv")!;
  nodes.push({ id: "kv", type: "kv_cache", x: qkv.x + typeWidth("qkv_proj") + COL_GAP, y: ROW + 115 });
  return { nodes, endX: x };
}

const layout = buildLayout();
export const DEFAULT_LAYOUT = layout.nodes;

// pseudo-node positions: just past each end of the pipeline
export const INPUT_POS = { x: -(typeWidth("_input") + COL_GAP), y: ROW };
export const OUTPUT_POS = { x: layout.endX, y: ROW };

export const DEFAULT_EDGES: SeedEdge[] = [
  { from: "_input", fromPort: "idx", to: "tok_emb", toPort: "idx" },
  { from: "_input", fromPort: "positions", to: "pos_emb", toPort: "positions" },
  { from: "tok_emb", fromPort: "out", to: "emb_add", toPort: "x" },
  { from: "pos_emb", fromPort: "out", to: "emb_add", toPort: "residual" },
  { from: "emb_add", fromPort: "out", to: "emb_drop", toPort: "x" },
  { from: "emb_drop", fromPort: "out", to: "ln1", toPort: "x" },
  { from: "ln1", fromPort: "out", to: "qkv", toPort: "x" },
  // Q bypasses the cache; K/V flow through it
  { from: "qkv", fromPort: "q", to: "attn", toPort: "q" },
  { from: "qkv", fromPort: "k", to: "kv", toPort: "k" },
  { from: "qkv", fromPort: "v", to: "kv", toPort: "v" },
  { from: "kv", fromPort: "k", to: "attn", toPort: "k" },
  { from: "kv", fromPort: "v", to: "vsum", toPort: "v" },
  { from: "attn", fromPort: "out", to: "mask", toPort: "x" },
  { from: "mask", fromPort: "out", to: "smax", toPort: "x" },
  { from: "smax", fromPort: "out", to: "vsum", toPort: "att" },
  { from: "vsum", fromPort: "out", to: "oproj", toPort: "x" },
  { from: "oproj", fromPort: "out", to: "attn_drop", toPort: "x" },
  { from: "attn_drop", fromPort: "out", to: "res1", toPort: "x" },
  { from: "emb_drop", fromPort: "out", to: "res1", toPort: "residual" }, // attn skip
  { from: "res1", fromPort: "out", to: "ln2", toPort: "x" },
  { from: "ln2", fromPort: "out", to: "mlp_up", toPort: "x" },
  { from: "mlp_up", fromPort: "out", to: "gelu", toPort: "x" },
  { from: "gelu", fromPort: "out", to: "mlp_down", toPort: "x" },
  { from: "mlp_down", fromPort: "out", to: "mlp_drop", toPort: "x" },
  { from: "mlp_drop", fromPort: "out", to: "res2", toPort: "x" },
  { from: "res1", fromPort: "out", to: "res2", toPort: "residual" }, // mlp skip
  { from: "res2", fromPort: "out", to: "lnf", toPort: "x" },
  { from: "lnf", fromPort: "out", to: "lm_head", toPort: "x" },
  { from: "lm_head", fromPort: "out", to: "_output", toPort: "logits" },
];
