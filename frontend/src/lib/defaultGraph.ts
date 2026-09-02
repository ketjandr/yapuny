// Seed graph: one pre-LN transformer block (embeddings -> attention -> MLP -> LM head).
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

const COL = 194; // horizontal spacing between stages (fits the widest content-sized node)
const ROW = 120; // main pipeline row (node top y)

export const DEFAULT_LAYOUT: PlacedNode[] = [
  // Embedding (tok/pos branch in, merge into the add)
  { id: "tok_emb", type: "token_embedding", x: 0 * COL, y: ROW - 80 },
  { id: "pos_emb", type: "position_embedding", x: 0 * COL, y: ROW + 90 },
  { id: "emb_add", type: "residual_add", x: 1 * COL, y: ROW },
  { id: "emb_drop", type: "dropout", x: 2 * COL, y: ROW },
  // Attention (KV cache dips below the main row)
  { id: "ln1", type: "layernorm", x: 3 * COL, y: ROW },
  { id: "qkv", type: "qkv_proj", x: 4 * COL, y: ROW },
  { id: "kv", type: "kv_cache", x: 5 * COL, y: ROW + 150 },
  { id: "attn", type: "attention_score", x: 6 * COL, y: ROW },
  { id: "mask", type: "causal_mask", x: 7 * COL, y: ROW },
  { id: "smax", type: "softmax", x: 8 * COL, y: ROW },
  { id: "vsum", type: "value_weighted_sum", x: 9 * COL, y: ROW },
  { id: "oproj", type: "out_proj", x: 10 * COL, y: ROW },
  { id: "attn_drop", type: "dropout", x: 11 * COL, y: ROW },
  { id: "res1", type: "residual_add", x: 12 * COL, y: ROW },
  // MLP
  { id: "ln2", type: "layernorm", x: 13 * COL, y: ROW },
  { id: "mlp_up", type: "mlp_up", x: 14 * COL, y: ROW },
  { id: "gelu", type: "mlp_activation", x: 15 * COL, y: ROW },
  { id: "mlp_down", type: "mlp_down", x: 16 * COL, y: ROW },
  { id: "mlp_drop", type: "dropout", x: 17 * COL, y: ROW },
  { id: "res2", type: "residual_add", x: 18 * COL, y: ROW },
  // Output
  { id: "lnf", type: "layernorm", x: 19 * COL, y: ROW },
  { id: "lm_head", type: "lm_head", x: 20 * COL, y: ROW },
];

// "_input" is the implicit graph input the compiler seeds (idx + positions).
export const DEFAULT_EDGES: SeedEdge[] = [
  { from: "_input", fromPort: "idx", to: "tok_emb", toPort: "idx" },
  { from: "_input", fromPort: "positions", to: "pos_emb", toPort: "positions" },
  { from: "tok_emb", fromPort: "out", to: "emb_add", toPort: "x" },
  { from: "pos_emb", fromPort: "out", to: "emb_add", toPort: "residual" },
  { from: "emb_add", fromPort: "out", to: "emb_drop", toPort: "x" },
  { from: "emb_drop", fromPort: "out", to: "ln1", toPort: "x" },
  { from: "ln1", fromPort: "out", to: "qkv", toPort: "x" },
  // Q bypasses the cache; K/V flow through it.
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
];

export const DEFAULT_META: GraphMetaSchema = {
  n_layer: 1,
  n_head: 6,
  n_embd: 384,
  block_size: 256,
  dropout: 0.1,
  vocab_size: 8000,
};
