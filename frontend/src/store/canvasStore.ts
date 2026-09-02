// Store: nodes/edges/meta/mode + needsCompile (structural edits set it, cosmetic don't).
import {
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  type Connection,
  type Edge,
  type EdgeChange,
  type Node,
  type NodeChange,
  reconnectEdge,
} from "@xyflow/react";
import { create } from "zustand";
import { DEFAULT_META } from "@/lib/defaultGraph";
import { canvasToGraph, makeNode, seedToCanvas } from "@/lib/graph";
import type { GraphMetaSchema, GraphRequest } from "@/lib/types";
import { toast } from "@/store/toastStore";

const seed = seedToCanvas();

// View mode; the hook point for mode-specific rendering (grey kv_cache, T/S, fusion/quant).
export type CanvasMode = "train" | "inference";

interface CanvasState {
  nodes: Node[];
  edges: Edge[];
  meta: GraphMetaSchema;
  mode: CanvasMode;
  selectedId: string | null;
  needsCompile: boolean;

  onNodesChange: (changes: NodeChange[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (conn: Connection) => void;
  onReconnect: (oldEdge: Edge, newConnection: Connection) => void;

  setSelected: (id: string | null) => void;
  addNode: (type: string, position: { x: number; y: number }) => void;
  markCompiled: () => void;
  setMode: (mode: CanvasMode) => void;

  toGraph: () => GraphRequest;
}

// is a target port already the destination of an edge (ignoring `exceptId`)?
function portOccupied(
  edges: Edge[],
  target: string | null,
  targetHandle: string | null | undefined,
  exceptId?: string,
): boolean {
  return edges.some(
    (e) => e.id !== exceptId && e.target === target && e.targetHandle === targetHandle,
  );
}

// guard shared by connect/reconnect: toasts and returns true if the input is taken
function rejectIfOccupied(edges: Edge[], conn: Connection, exceptId?: string): boolean {
  if (portOccupied(edges, conn.target, conn.targetHandle, exceptId)) {
    toast.error("Only one input per port, input was already connected!");
    return true;
  }
  return false;
}

// structural changes (vs cosmetic move/select)
function nodesAreMeaningful(changes: NodeChange[]): boolean {
  return changes.some((c) => c.type === "add" || c.type === "remove" || c.type === "replace");
}
function edgesAreMeaningful(changes: EdgeChange[]): boolean {
  return changes.some((c) => c.type === "add" || c.type === "remove" || c.type === "replace");
}

export const useCanvasStore = create<CanvasState>((set, get) => ({
  nodes: seed.nodes,
  edges: seed.edges,
  meta: DEFAULT_META,
  mode: "train", // editing/training is the default view
  selectedId: null,
  needsCompile: true, // a fresh (uncompiled) graph needs a compile

  onNodesChange: (changes) =>
    set((s) => ({
      nodes: applyNodeChanges(changes, s.nodes),
      needsCompile: s.needsCompile || nodesAreMeaningful(changes),
    })),

  onEdgesChange: (changes) =>
    set((s) => ({
      edges: applyEdgeChanges(changes, s.edges),
      needsCompile: s.needsCompile || edgesAreMeaningful(changes),
    })),

  // each input (target) port accepts at most one edge; output ports may fan out
  onConnect: (conn) => {
    const { edges } = get();
    if (rejectIfOccupied(edges, conn)) return;
    set({ edges: addEdge({ ...conn }, edges), needsCompile: true });
  },

  onReconnect: (oldEdge, newConnection) => {
    const { edges } = get();
    if (rejectIfOccupied(edges, newConnection, oldEdge.id)) return;
    set({ edges: reconnectEdge(oldEdge, newConnection, edges), needsCompile: true });
  },

  setSelected: (id) => set({ selectedId: id }),

  addNode: (type, position) =>
    set((s) => ({ nodes: [...s.nodes, makeNode(type, position)], needsCompile: true })),

  markCompiled: () => set({ needsCompile: false }),
  setMode: (mode) => set({ mode }), // not a graph edit -> no needsCompile

  toGraph: () => {
    const s = get();
    return canvasToGraph(s.nodes, s.edges, s.meta);
  },
}));
