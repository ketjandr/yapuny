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
import { analyzeBlock } from "@/lib/block";
import { DEFAULT_META } from "@/lib/defaultGraph";
import { FUSE_PORT, fusionChainError, isFusionEdge, validateFusionConnect } from "@/lib/fusion";
import { canvasToGraph, makeEdge, makeFusionEdge, makeNode, seedToCanvas, type YNodeData } from "@/lib/graph";
import type { GraphMetaSchema, GraphRequest } from "@/lib/types";
import { toast } from "@/store/toastStore";

const seed = seedToCanvas();

export const N_LAYER_MIN = 1;
export const N_LAYER_MAX = 16; // block loop count bound; shared by the config slider + block stepper

// View mode; the hook point for mode-specific rendering (grey kv_cache, T/S, fusion/quant).
export type CanvasMode = "train" | "inference";

interface Clipboard {
  nodes: { id: string; type: string; quantized: string | null; x: number; y: number }[];
  edges: { from: string; fromPort: string | null; to: string; toPort: string | null }[];
}

const isPseudo = (type: string) => type === "_input" || type === "_output";

const PASTE_STEP = 28; // flow px offset per keyboard paste (cascades so copies don't stack)
let pasteCount = 0; // reset on copy; drives the cascading keyboard-paste offset

// build fresh nodes/edges from the clipboard, shifted by (ox, oy), all selected
function placeClipboard(clip: Clipboard, ox: number, oy: number) {
  const idMap = new Map<string, string>();
  const newNodes = clip.nodes.map((n) => {
    const node = makeNode(n.type, { x: n.x + ox, y: n.y + oy }, n.quantized);
    idMap.set(n.id, node.id);
    return { ...node, selected: true };
  });
  const newEdges = clip.edges.map((e) => {
    // fusion edges (both ports are the fuse port) must be recreated as fusion, not data, edges
    const edge =
      e.fromPort === FUSE_PORT
        ? makeFusionEdge(idMap.get(e.from)!, idMap.get(e.to)!)
        : makeEdge(idMap.get(e.from)!, e.fromPort, idMap.get(e.to)!, e.toPort);
    return { ...edge, selected: true };
  });
  return { newNodes, newEdges };
}

// deselect everything, append the pasted (selected) subgraph
function withPaste(s: CanvasState, newNodes: Node[], newEdges: Edge[]) {
  return {
    nodes: [...s.nodes.map((n) => ({ ...n, selected: false })), ...newNodes],
    edges: [...s.edges.map((e) => ({ ...e, selected: false })), ...newEdges],
    needsCompile: true,
  };
}

interface CanvasState {
  nodes: Node[];
  edges: Edge[];
  meta: GraphMetaSchema;
  mode: CanvasMode;
  selectedId: string | null;
  needsCompile: boolean;
  blockStart: string | null; // block boundary markers; the repeated slice is derived from them
  blockEnd: string | null;
  clipboard: Clipboard | null;

  onNodesChange: (changes: NodeChange[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (conn: Connection) => void;
  onReconnect: (oldEdge: Edge, newConnection: Connection) => void;

  setSelected: (id: string | null) => void;
  addNode: (type: string, position: { x: number; y: number }) => void;
  removeNodes: (ids: string[]) => void;
  removeEdges: (ids: string[]) => void;
  copyNodes: (nodeIds: string[], edgeIds: string[]) => void;
  paste: (position: { x: number; y: number }) => void;
  pasteAtOffset: () => void;
  setBlockStart: (id: string | null) => void;
  setBlockEnd: (id: string | null) => void;
  clearBlock: () => void;
  setNLayer: (n: number) => void;
  setMeta: (patch: Partial<GraphMetaSchema>) => void;
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
    toast.error("Only one input tensor per port, it was already connected!");
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
  blockStart: null,
  blockEnd: null,
  clipboard: null,

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

  // each input (target) port accepts at most one edge; output ports may fan out.
  // a connection touching a fuse port is a (visual-only) fusion stream, handled separately.
  onConnect: (conn) => {
    const { edges } = get();
    const srcFuse = conn.sourceHandle === FUSE_PORT;
    const tgtFuse = conn.targetHandle === FUSE_PORT;
    if (srcFuse || tgtFuse) {
      // fusion must join two distinct fuse ports; ignore duplicates (either direction)
      if (!srcFuse || !tgtFuse || conn.source === conn.target) return;
      const dupe = edges.some(
        (e) =>
          isFusionEdge(e) &&
          ((e.source === conn.source && e.target === conn.target) ||
            (e.source === conn.target && e.target === conn.source)),
      );
      if (dupe) return;
      const err = validateFusionConnect(conn.source!, conn.target!, edges);
      if (err) {
        toast.error(err);
        return;
      }
      set({ edges: [...edges, makeFusionEdge(conn.source!, conn.target!)] }); // no needsCompile
      return;
    }
    if (rejectIfOccupied(edges, conn)) return;
    // a data edge must not break an existing fusion chain (fuse-first-then-connect)
    const next = addEdge({ ...conn }, edges);
    const err = fusionChainError(next);
    if (err) {
      toast.error(err);
      return;
    }
    set({ edges: next, needsCompile: true });
  },

  onReconnect: (oldEdge, newConnection) => {
    const { edges } = get();
    if (rejectIfOccupied(edges, newConnection, oldEdge.id)) return;
    const next = reconnectEdge(oldEdge, newConnection, edges);
    const err = fusionChainError(next);
    if (err) {
      toast.error(err);
      return;
    }
    set({ edges: next, needsCompile: true });
  },

  setSelected: (id) => set({ selectedId: id }),

  addNode: (type, position) =>
    set((s) => ({ nodes: [...s.nodes, makeNode(type, position)], needsCompile: true })),

  // delete nodes (and their connected edges); pseudo-nodes (_input/_output) are protected
  removeNodes: (ids) => {
    const { nodes } = get();
    const del = new Set(
      ids.filter((id) => {
        const t = (nodes.find((n) => n.id === id)?.data as YNodeData | undefined)?.type;
        return t !== undefined && !isPseudo(t);
      }),
    );
    if (del.size === 0) return;
    set((s) => ({
      nodes: s.nodes.filter((n) => !del.has(n.id)),
      edges: s.edges.filter((e) => !del.has(e.source) && !del.has(e.target)),
      blockStart: s.blockStart && del.has(s.blockStart) ? null : s.blockStart,
      blockEnd: s.blockEnd && del.has(s.blockEnd) ? null : s.blockEnd,
      selectedId: s.selectedId && del.has(s.selectedId) ? null : s.selectedId,
      needsCompile: true,
    }));
  },

  removeEdges: (ids) => {
    const del = new Set(ids);
    set((s) => ({ edges: s.edges.filter((e) => !del.has(e.id)), needsCompile: true }));
  },

  // copy the nodes (pseudo-nodes excluded); an edge is included only if it was selected too
  // (in edgeIds) and both its endpoints are among the copied nodes
  copyNodes: (nodeIds, edgeIds) => {
    const { nodes, edges } = get();
    const wanted = new Set(nodeIds);
    const picked = nodes.filter(
      (n) => wanted.has(n.id) && !isPseudo((n.data as YNodeData).type),
    );
    const pickedIds = new Set(picked.map((n) => n.id));
    const wantEdges = new Set(edgeIds);
    const clip: Clipboard = {
      nodes: picked.map((n) => ({
        id: n.id,
        type: (n.data as YNodeData).type,
        quantized: (n.data as YNodeData).quantized,
        x: n.position.x,
        y: n.position.y,
      })),
      edges: edges
        .filter((e) => wantEdges.has(e.id) && pickedIds.has(e.source) && pickedIds.has(e.target))
        .map((e) => ({
          from: e.source,
          fromPort: e.sourceHandle ?? null,
          to: e.target,
          toPort: e.targetHandle ?? null,
        })),
    };
    pasteCount = 0; // fresh copy restarts the cascade
    set({ clipboard: clip.nodes.length ? clip : null });
  },

  // paste with the group's top-left at `position` (right-click paste at the cursor)
  paste: (position) => {
    const { clipboard } = get();
    if (!clipboard?.nodes.length) return;
    const ax = Math.min(...clipboard.nodes.map((n) => n.x));
    const ay = Math.min(...clipboard.nodes.map((n) => n.y));
    const { newNodes, newEdges } = placeClipboard(clipboard, position.x - ax, position.y - ay);
    set((s) => withPaste(s, newNodes, newEdges));
  },

  // keyboard paste: a deterministic cascading offset from the original (no cursor dependency)
  pasteAtOffset: () => {
    const { clipboard } = get();
    if (!clipboard?.nodes.length) return;
    pasteCount += 1;
    const d = PASTE_STEP * pasteCount;
    const { newNodes, newEdges } = placeClipboard(clipboard, d, d);
    set((s) => withPaste(s, newNodes, newEdges));
  },

  // block boundaries change what gets unrolled -> structural (weights invalidate on recompile)
  setBlockStart: (id) => set({ blockStart: id, needsCompile: true }),
  setBlockEnd: (id) => set({ blockEnd: id, needsCompile: true }),
  clearBlock: () => set({ blockStart: null, blockEnd: null, needsCompile: true }),
  // loop count; also structural (changes the unrolled depth)
  setNLayer: (n) =>
    set((s) => ({
      meta: { ...s.meta, n_layer: Math.min(N_LAYER_MAX, Math.max(N_LAYER_MIN, n)) },
      needsCompile: true,
    })),

  // merge a config patch; smart constraint: n_embd is snapped to a multiple of n_head so the
  // per-head dim stays an integer. Config changes are structural -> needsCompile.
  setMeta: (patch) =>
    set((s) => {
      const m = { ...s.meta, ...patch };
      const head = Math.max(1, Math.round(m.n_head));
      const embd = Math.max(head, Math.round(m.n_embd / head) * head);
      return { meta: { ...m, n_head: head, n_embd: embd }, needsCompile: true };
    }),

  markCompiled: () => set({ needsCompile: false }),
  setMode: (mode) => set({ mode }), // not a graph edit -> no needsCompile

  // emit the block only when it is a valid single-in/single-out shape-preserving slice
  toGraph: () => {
    const s = get();
    const block = analyzeBlock(s.nodes, s.edges, s.blockStart, s.blockEnd);
    return canvasToGraph(s.nodes, s.edges, s.meta, block.valid ? [...block.nodeIds] : undefined);
  },
}));
