// Per-project canvas persistence. Each project's graph + view state is stored under its own key
// (yapuny.canvas.<id>) so the Models page can hold many independent canvases; the projects index
// (titles / timestamps) lives in lib/projects.ts. Pure helpers only (no store import). Transient
// React Flow flags (selection / drag) are normalized out so selecting a node doesn't churn saves.
import type { Edge, Node, Viewport } from "@xyflow/react";
import type { GraphMetaSchema } from "./types";

const CANVAS_PREFIX = "yapuny.canvas.";
const canvasKey = (id: string) => `${CANVAS_PREFIX}${id}`;

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

export function loadCanvas(id: string): PersistedCanvas | null {
  try {
    const raw = localStorage.getItem(canvasKey(id));
    if (!raw) return null;
    return JSON.parse(raw) as PersistedCanvas;
  } catch {
    return null; // unavailable / corrupt
  }
}

export function writeCanvas(id: string, data: PersistedCanvas): void {
  try {
    localStorage.setItem(canvasKey(id), JSON.stringify(data));
  } catch {
    /* quota exceeded / storage disabled -> skip */
  }
}

export function deleteCanvas(id: string): void {
  try {
    localStorage.removeItem(canvasKey(id));
  } catch {
    /* ignore */
  }
}
