// Seed graph: one pre-LN transformer block, built in variants - absolute (position_embedding) vs
// rotary (RoPE) positions, with/without a kv cache, and unfused (score/mask/softmax/vsum) vs flash
// attention. templateCanvas (lib/projects.ts) picks a variant per starter template.
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

// what a seed's attention path looks like
export interface SeedVariant {
  rope: boolean; // rotary positions (replaces position_embedding)
  cache: boolean; // kv cache branch below the attention
  flash: boolean; // fused flash attention (replaces score/mask/softmax/value-sum)
}
export const ABS_VARIANT: SeedVariant = { rope: false, cache: false, flash: false };
export const OPT_VARIANT: SeedVariant = { rope: true, cache: true, flash: true };

const COL_GAP = 50; // uniform horizontal gap between stages (x is width-driven, not fixed)
const ROW = 130; // main pipeline row (node top y)

function typeWidth(type: string): number {
  return nodeWidth(resolveNodeDef(type)!, DEFAULT_META);
}

// _input sits just left of the pipeline; seed graphs compute their own _output x (buildSeed.endX),
// blank/reconstructed graphs place _output relative to their own content
export const INPUT_POS = { x: -(typeWidth("_input") + COL_GAP), y: ROW };

// the main pipeline as ordered stages (columns) for a variant; embeddings may stack tok/pos
function stagesFor(v: SeedVariant): { id: string; type: string; y: number }[][] {
  const stages: { id: string; type: string; y: number }[][] = [];
  if (v.rope) stages.push([{ id: "tok_emb", type: "token_embedding", y: ROW }]);
  else {
    stages.push([
      { id: "tok_emb", type: "token_embedding", y: ROW - 55 },
      { id: "pos_emb", type: "position_embedding", y: ROW + 85 },
    ]);
    stages.push([{ id: "emb_add", type: "residual_add", y: ROW }]);
  }
  stages.push([{ id: "emb_drop", type: "dropout", y: ROW }]);
  stages.push([{ id: "ln1", type: "layernorm", y: ROW }]);
  stages.push([{ id: "qkv", type: "qkv_proj", y: ROW }]);
  if (v.rope) stages.push([{ id: "rope", type: "rope", y: ROW }]);
  if (v.flash) stages.push([{ id: "flash", type: "flash_attention", y: ROW }]);
  else {
    stages.push([{ id: "attn", type: "attention_score", y: ROW }]);
    stages.push([{ id: "mask", type: "causal_mask", y: ROW }]);
    stages.push([{ id: "smax", type: "softmax", y: ROW }]);
    stages.push([{ id: "vsum", type: "value_weighted_sum", y: ROW }]);
  }
  stages.push([{ id: "oproj", type: "out_proj", y: ROW }]);
  stages.push([{ id: "attn_drop", type: "dropout", y: ROW }]);
  stages.push([{ id: "res1", type: "residual_add", y: ROW }]);
  stages.push([{ id: "ln2", type: "layernorm", y: ROW }]);
  stages.push([{ id: "mlp_up", type: "mlp_up", y: ROW }]);
  stages.push([{ id: "gelu", type: "mlp_activation", y: ROW }]);
  stages.push([{ id: "mlp_down", type: "mlp_down", y: ROW }]);
  stages.push([{ id: "mlp_drop", type: "dropout", y: ROW }]);
  stages.push([{ id: "res2", type: "residual_add", y: ROW }]);
  stages.push([{ id: "lnf", type: "layernorm", y: ROW }]);
  stages.push([{ id: "lm_head", type: "lm_head", y: ROW }]);
  return stages;
}

function seedEdges(v: SeedVariant): SeedEdge[] {
  const e: SeedEdge[] = [];
  const add = (from: string, fromPort: string, to: string, toPort: string) =>
    e.push({ from, fromPort, to, toPort });

  add("_input", "idx", "tok_emb", "idx");
  if (v.rope) add("tok_emb", "out", "emb_drop", "x");
  else {
    add("_input", "positions", "pos_emb", "positions");
    add("tok_emb", "out", "emb_add", "x");
    add("pos_emb", "out", "emb_add", "residual");
    add("emb_add", "out", "emb_drop", "x");
  }
  add("emb_drop", "out", "ln1", "x");
  add("ln1", "out", "qkv", "x");

  // q/k optionally rotated by rope before attention
  let q: [string, string] = ["qkv", "q"];
  let k: [string, string] = ["qkv", "k"];
  if (v.rope) {
    add("qkv", "q", "rope", "q");
    add("qkv", "k", "rope", "k");
    add("_input", "positions", "rope", "positions");
    q = ["rope", "q"];
    k = ["rope", "k"];
  }
  // k/v into attention, optionally routed through the cache
  let kv: [string, string] = k;
  let vv: [string, string] = ["qkv", "v"];
  if (v.cache) {
    add(k[0], k[1], "kv", "k");
    add("qkv", "v", "kv", "v");
    kv = ["kv", "k"];
    vv = ["kv", "v"];
  }
  if (v.flash) {
    add(q[0], q[1], "flash", "q");
    add(kv[0], kv[1], "flash", "k");
    add(vv[0], vv[1], "flash", "v");
    add("flash", "out", "oproj", "x");
  } else {
    add(q[0], q[1], "attn", "q");
    add(kv[0], kv[1], "attn", "k");
    add("attn", "out", "mask", "x");
    add("mask", "out", "smax", "x");
    add("smax", "out", "vsum", "att");
    add(vv[0], vv[1], "vsum", "v");
    add("vsum", "out", "oproj", "x");
  }

  add("oproj", "out", "attn_drop", "x");
  add("attn_drop", "out", "res1", "x");
  add("emb_drop", "out", "res1", "residual"); // attn skip
  add("res1", "out", "ln2", "x");
  add("ln2", "out", "mlp_up", "x");
  add("mlp_up", "out", "gelu", "x");
  add("gelu", "out", "mlp_down", "x");
  add("mlp_down", "out", "mlp_drop", "x");
  add("mlp_drop", "out", "res2", "x");
  add("res1", "out", "res2", "residual"); // mlp skip
  add("res2", "out", "lnf", "x");
  add("lnf", "out", "lm_head", "x");
  add("lm_head", "out", "_output", "logits");
  return e;
}

// build a variant's placed nodes + wired edges; endX is where the _output pseudo-node sits
export function buildSeed(v: SeedVariant): { nodes: PlacedNode[]; edges: SeedEdge[]; endX: number } {
  const stages = stagesFor(v);
  const nodes: PlacedNode[] = [];
  const kvW = typeWidth("kv_cache");
  const under = v.rope ? "rope" : "qkv"; // the kv branch hangs below this stage

  let x = 0;
  for (const stage of stages) {
    for (const n of stage) nodes.push({ id: n.id, type: n.type, x, y: n.y });
    let advance = Math.max(...stage.map((n) => typeWidth(n.type))) + COL_GAP;
    if (v.cache && stage.some((n) => n.id === under)) advance += kvW + COL_GAP; // room for kv below
    x += advance;
  }
  if (v.cache) {
    const u = nodes.find((n) => n.id === under)!;
    nodes.push({ id: "kv", type: "kv_cache", x: u.x + typeWidth(u.type) + COL_GAP, y: ROW + 115 });
  }
  return { nodes, edges: seedEdges(v), endX: x };
}
