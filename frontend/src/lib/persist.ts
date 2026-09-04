// Canvas persistence: serialize the graph + view state to localStorage so the canvas survives a
// reload. Pure helpers only (no store import, to avoid a cycle); the autosave subscription that
// calls writePersisted lives in the store. Transient React Flow flags (selection / drag) are
// normalized out so selecting a node doesn't churn the save or pollute the persisted blob.
import type { Edge, Node, Viewport } from "@xyflow/react";
import type { GraphMetaSchema } from "./types";

export const STORAGE_KEY = "yapuny.canvas.v1";

// the graph state captured at compile time; the target of "revert to compiled"
export interface CompiledSnapshot {
  nodes: Node[];
  edges: Edge[];
  meta: GraphMetaSchema;
  blockStart: string | null;
  blockEnd: string | null;
}

export interface PersistedCanvas extends CompiledSnapshot {
  mode: "train" | "inference";
  needsCompile: boolean;
  trained: boolean;
  lastCompiled: CompiledSnapshot | null;
  viewport: Viewport | null;
}

// reset transient flags so selection/drag state neither triggers a save nor persists
export function cleanNodes(nodes: Node[]): Node[] {
  return nodes.map((n) => ({ ...n, selected: false, dragging: false }));
}
export function cleanEdges(edges: Edge[]): Edge[] {
  return edges.map((e) => ({ ...e, selected: false }));
}

export function loadPersisted(): PersistedCanvas | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as PersistedCanvas;
  } catch {
    return null; // unavailable / corrupt -> fall back to the seed graph
  }
}

export function writePersisted(data: PersistedCanvas): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  } catch {
    /* quota exceeded / storage disabled -> silently skip this save */
  }
}
