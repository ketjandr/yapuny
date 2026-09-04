// Derive a readable, structure-based display id for each node, e.g. dropout_0, dropout_1
// (per-type counter in pipeline order). This is presentation only - the node's real identity
// stays its stable id. Consumed by the properties panel + benchmark; recompute on demand.
import type { Edge, Node } from "@xyflow/react";
import type { YNodeData } from "./graph";

const nodeType = (n: Node) => (n.data as YNodeData).type;

// Topological (pipeline) order, tie-broken by x then id so branches stay deterministic.
function topoOrder(nodes: Node[], edges: Edge[]): Node[] {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const indeg = new Map(nodes.map((n) => [n.id, 0]));
  const adj = new Map<string, string[]>();
  for (const e of edges) {
    if (!byId.has(e.source) || !byId.has(e.target)) continue;
    if (!adj.has(e.source)) adj.set(e.source, []);
    adj.get(e.source)!.push(e.target);
    indeg.set(e.target, indeg.get(e.target)! + 1);
  }

  const earlier = (a: Node, b: Node) => a.position.x - b.position.x || a.id.localeCompare(b.id);
  const ready = nodes.filter((n) => indeg.get(n.id) === 0).sort(earlier);
  const order: Node[] = [];
  while (ready.length) {
    const n = ready.shift()!;
    order.push(n);
    for (const t of adj.get(n.id) ?? []) {
      indeg.set(t, indeg.get(t)! - 1);
      if (indeg.get(t) === 0) ready.push(byId.get(t)!);
    }
    ready.sort(earlier);
  }

  // cycle fallback: append anything left so every node still gets an id
  if (order.length < nodes.length) {
    const placed = new Set(order.map((n) => n.id));
    for (const n of nodes) if (!placed.has(n.id)) order.push(n);
  }
  return order;
}

// map: node.id -> structural id. A type with one instance keeps its bare name (lm_head);
// repeated types are indexed in pipeline order (dropout_0/1/2). Pseudo-nodes are skipped.
export function structuralIds(nodes: Node[], edges: Edge[]): Map<string, string> {
  const order = topoOrder(nodes, edges).filter(
    (n) => nodeType(n) !== "_input" && nodeType(n) !== "_output",
  );

  const total: Record<string, number> = {};
  for (const n of order) total[nodeType(n)] = (total[nodeType(n)] ?? 0) + 1;

  const seen: Record<string, number> = {};
  const out = new Map<string, string>();
  for (const n of order) {
    const type = nodeType(n);
    if (total[type] === 1) {
      out.set(n.id, type);
    } else {
      const i = seen[type] ?? 0;
      seen[type] = i + 1;
      out.set(n.id, `${type}_${i}`);
    }
  }
  return out;
}

const LAYER_PREFIX = /l\d+_/g; // the l{layer}_ prefix the backend adds when unrolling a block
const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

// Rewrite a backend validation message to reference structural ids: strip block-unroll prefixes
// (so l2_res1 -> res1, the canvas node), then swap each raw node id for its structural id.
export function humanizeMessage(msg: string, ids: Map<string, string>): string {
  const logical = msg.replace(LAYER_PREFIX, "");
  if (ids.size === 0) return logical;
  // longest id first so a shorter id can't shadow a longer one that contains it
  const keys = [...ids.keys()].sort((a, b) => b.length - a.length);
  const re = new RegExp(`\\b(?:${keys.map(escapeRe).join("|")})\\b`, "g");
  return logical.replace(re, (m) => ids.get(m) ?? m);
}
