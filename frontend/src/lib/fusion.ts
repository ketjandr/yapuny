// Fusion is a separate connection type: the user drags between the bottom diamond ports of
// fusable nodes. These edges are visual-only (green energy stream) and are NOT sent to the
// backend - they never touch the data graph, block reachability, or the structure hash.
import type { Edge, Node } from "@xyflow/react";
import type { CanvasMode } from "@/store/canvasStore";
import type { YNodeData } from "./graph";

export const FUSE_PORT = "fuse"; // handle id of the bottom fusion diamond (source + target)

// Whether fusion is shown for a given canvas mode. Fusion is inference-only for now (the fused
// kernels can't train yet).
export function fusionVisible(mode: CanvasMode): boolean {
  return mode === "inference";
}

export function isFusionEdge(e: Edge): boolean {
  return e.type === "fusion";
}

// nodes that participate in any fusion stream (for the green border aura)
export function fusedNodeIds(edges: Edge[]): Set<string> {
  const ids = new Set<string>();
  for (const e of edges) {
    if (isFusionEdge(e)) {
      ids.add(e.source);
      ids.add(e.target);
    }
  }
  return ids;
}

// Every fusion group's internal data edges must form a straight chain.
export function fusionChainError(edges: Edge[]): string | null {
  const dataEdges = edges.filter((e) => !isFusionEdge(e));
  for (const group of deriveFusionGroups(edges)) {
    const set = new Set(group);
    const outDeg = new Map<string, number>();
    const inDeg = new Map<string, number>();
    for (const e of dataEdges) {
      if (set.has(e.source) && set.has(e.target)) {
        outDeg.set(e.source, (outDeg.get(e.source) ?? 0) + 1);
        inDeg.set(e.target, (inDeg.get(e.target) ?? 0) + 1);
      }
    }
    for (const n of group) {
      if ((outDeg.get(n) ?? 0) > 1) return "A node in the group can't feed two fused nodes!";
      if ((inDeg.get(n) ?? 0) > 1) return "A node in the group can't be fed by two fused nodes!";
      if ((outDeg.get(n) ?? 0) >= 1 && dataEdges.some((e) => e.source === n && !set.has(e.target)))
        return "A non-final node in the group can't feed an external node!";
    }
  }
  return null;
}

// Connect-time validation for a new fusion edge: the pair must be directly data-connected, and
// the resulting groups must still be straight chains.
export function validateFusionConnect(source: string, target: string, edges: Edge[]): string | null {
  const hasData = (a: string, b: string) =>
    edges.some((e) => !isFusionEdge(e) && e.source === a && e.target === b);
  if (!hasData(source, target) && !hasData(target, source)) {
    return "Fusion needs a direct connection between the two nodes!";
  }
  const probe = { id: "__probe__", type: "fusion", source, target } as Edge;
  return fusionChainError([...edges, probe]);
}

// -- validation (against the backend fusion catalog, cached client-side) --

// GET /fusion/available: whether fusion kernels are built, and the valid node-type patterns
export interface FusionCatalog {
  available: boolean;
  patterns: { nodes: string[]; kernel: string }[];
}

export interface FusionValidation {
  badNodes: Set<string>;
  badEdges: Set<string>;
}

// fusion groups = connected components of the fusion edges (union-find)
export function deriveFusionGroups(edges: Edge[]): string[][] {
  const parent = new Map<string, string>();
  const find = (x: string): string => {
    let r = x;
    while ((parent.get(r) ?? r) !== r) r = parent.get(r) ?? r;
    parent.set(x, r);
    return r;
  };
  for (const e of edges) {
    if (!isFusionEdge(e)) continue;
    if (!parent.has(e.source)) parent.set(e.source, e.source);
    if (!parent.has(e.target)) parent.set(e.target, e.target);
    const ra = find(e.source);
    const rb = find(e.target);
    if (ra !== rb) parent.set(ra, rb);
  }
  const groups = new Map<string, string[]>();
  for (const id of parent.keys()) {
    const r = find(id);
    const list = groups.get(r);
    if (list) list.push(id);
    else groups.set(r, [id]);
  }
  return [...groups.values()];
}

// The kernel a fused group forms (its registry class name), or null if it matches none / the
// catalog isn't available. Compact, meaningful identity for the group - unlike listing members.
export function matchFusionKernel(
  group: string[],
  typeOf: Map<string, string>,
  dataEdges: Edge[],
  catalog: FusionCatalog | undefined,
): string | null {
  if (!catalog?.available) return null;
  const ordered = orderByDataFlow(group, dataEdges);
  if (!isDataChain(ordered, dataEdges)) return null;
  const types = ordered.map((id) => typeOf.get(id));
  const match = catalog.patterns.find(
    (p) => p.nodes.length === types.length && p.nodes.every((t, i) => t === types[i]),
  );
  return match?.kernel ?? null;
}

// order a group's nodes by their data flow (Kahn); falls back to input order if not a clean chain
function orderByDataFlow(group: string[], dataEdges: Edge[]): string[] {
  const set = new Set(group);
  const indeg = new Map(group.map((id) => [id, 0]));
  const adj = new Map<string, string[]>();
  for (const e of dataEdges) {
    if (set.has(e.source) && set.has(e.target)) {
      (adj.get(e.source) ?? adj.set(e.source, []).get(e.source)!).push(e.target);
      indeg.set(e.target, (indeg.get(e.target) ?? 0) + 1);
    }
  }
  const queue = group.filter((id) => indeg.get(id) === 0);
  const out: string[] = [];
  while (queue.length) {
    const n = queue.shift()!;
    out.push(n);
    for (const m of adj.get(n) ?? []) {
      indeg.set(m, indeg.get(m)! - 1);
      if (indeg.get(m) === 0) queue.push(m);
    }
  }
  return out.length === group.length ? out : group;
}

// The fusion groups to send in a GraphRequest: each connected component of fusion edges, its nodes
// ordered by data flow so the backend can match the node-type pattern to a kernel.
export function fusionGroupsForRequest(edges: Edge[]): { nodes: string[] }[] {
  const dataEdges = edges.filter((e) => !isFusionEdge(e));
  return deriveFusionGroups(edges).map((group) => ({ nodes: orderByDataFlow(group, dataEdges) }));
}

// each consecutive pair in the ordered group must be joined by a data edge (a contiguous chain)
function isDataChain(ordered: string[], dataEdges: Edge[]): boolean {
  for (let i = 0; i < ordered.length - 1; i++) {
    if (!dataEdges.some((e) => e.source === ordered[i] && e.target === ordered[i + 1])) return false;
  }
  return true;
}

// Client-side, UX-only: flag fused nodes/edges whose group is unavailable or doesn't match a
// kernel pattern. The backend validator stays the authority at compile. Undefined catalog (not
// yet fetched / offline) means "don't flag" - optimistic.
export function validateFusion(
  nodes: Node[],
  edges: Edge[],
  catalog: FusionCatalog | undefined,
): FusionValidation {
  const badNodes = new Set<string>();
  const badEdges = new Set<string>();
  const fusionEdges = edges.filter(isFusionEdge);
  if (fusionEdges.length === 0 || !catalog) return { badNodes, badEdges };

  const typeOf = new Map(nodes.map((n) => [n.id, (n.data as YNodeData).type]));
  const dataEdges = edges.filter((e) => !isFusionEdge(e));

  for (const group of deriveFusionGroups(edges)) {
    let ok = catalog.available;
    if (ok) {
      const ordered = orderByDataFlow(group, dataEdges);
      const types = ordered.map((id) => typeOf.get(id));
      ok =
        isDataChain(ordered, dataEdges) &&
        catalog.patterns.some(
          (p) => p.nodes.length === types.length && p.nodes.every((t, i) => t === types[i]),
        );
    }
    if (!ok) {
      for (const id of group) badNodes.add(id);
      for (const e of fusionEdges) {
        if (group.includes(e.source) && group.includes(e.target)) badEdges.add(e.id);
      }
    }
  }
  return { badNodes, badEdges };
}
