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
  type Viewport,
} from "@xyflow/react";
import { create } from "zustand";
import { analyzeBlock } from "@/lib/block";
import { DEFAULT_META } from "@/lib/defaultGraph";
import { FUSE_PORT, fusionChainError, isFusionEdge, validateFusionConnect } from "@/lib/fusion";
import { canvasToGraph, makeEdge, makeFusionEdge, makeNode, seedToCanvas, type YNodeData } from "@/lib/graph";
import {
  cleanEdges,
  cleanNodes,
  type CompiledSnapshot,
  loadPersisted,
  type PersistedCanvas,
  writePersisted,
} from "@/lib/persist";
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
  };
}

interface CanvasState {
  nodes: Node[];
  edges: Edge[];
  meta: GraphMetaSchema;
  mode: CanvasMode;
  selectedId: string | null;
  modelId: string; // stable, frontend-minted id: the key into the backend model cache + weight locker
  lastCompiled: CompiledSnapshot | null; // snapshot taken at compile; target of revertToCompiled
  viewport: Viewport | null; // persisted pan/zoom, restored on reload
  saveStatus: "saving" | "saved"; // autosave indicator
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
  revertToCompiled: () => void;
  setMode: (mode: CanvasMode) => void;
  setViewport: (vp: Viewport) => void;
  setSaveStatus: (status: "saving" | "saved") => void;

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

// restore the last session from localStorage, else start from the seed graph
const persisted = loadPersisted();

// stable per-canvas id, minted once and persisted; the backend keys its model cache + locker by it
function mintModelId(): string {
  try {
    return crypto.randomUUID();
  } catch {
    return `m_${Date.now()}_${Math.random().toString(36).slice(2)}`;
  }
}

export const useCanvasStore = create<CanvasState>((set, get) => ({
  nodes: persisted?.nodes ?? seed.nodes,
  edges: persisted?.edges ?? seed.edges,
  meta: persisted?.meta ?? DEFAULT_META,
  mode: persisted?.mode ?? "train", // editing/training is the default view
  selectedId: null,
  modelId: persisted?.modelId ?? mintModelId(),
  lastCompiled: persisted?.lastCompiled ?? null,
  viewport: persisted?.viewport ?? null,
  saveStatus: "saved",
  blockStart: persisted?.blockStart ?? null,
  blockEnd: persisted?.blockEnd ?? null,
  clipboard: null,

  // needsCompile/trained are NOT tracked here — the backend (worker model cache + weight locker) is
  // the source of truth, surfaced via compileStore. Edits just mutate the graph.
  onNodesChange: (changes) => set((s) => ({ nodes: applyNodeChanges(changes, s.nodes) })),

  onEdgesChange: (changes) => set((s) => ({ edges: applyEdgeChanges(changes, s.edges) })),

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
    set({ edges: next });
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
    set({ edges: next });
  },

  setSelected: (id) => set({ selectedId: id }),

  addNode: (type, position) => set((s) => ({ nodes: [...s.nodes, makeNode(type, position)] })),

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
    }));
  },

  removeEdges: (ids) => {
    const del = new Set(ids);
    set((s) => ({ edges: s.edges.filter((e) => !del.has(e.id)) }));
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

  // block boundaries change what gets unrolled -> structural (changes the compiled signature)
  setBlockStart: (id) => set({ blockStart: id }),
  setBlockEnd: (id) => set({ blockEnd: id }),
  clearBlock: () => set({ blockStart: null, blockEnd: null }),
  // loop count; also structural (changes the unrolled depth)
  setNLayer: (n) =>
    set((s) => ({
      meta: { ...s.meta, n_layer: Math.min(N_LAYER_MAX, Math.max(N_LAYER_MIN, n)) },
    })),

  // merge a config patch; smart constraint: n_embd is snapped to a multiple of n_head so the
  // per-head dim stays an integer.
  setMeta: (patch) =>
    set((s) => {
      const m = { ...s.meta, ...patch };
      const head = Math.max(1, Math.round(m.n_head));
      const embd = Math.max(head, Math.round(m.n_embd / head) * head);
      return { meta: { ...m, n_head: head, n_embd: embd } };
    }),

  // snapshot the graph locally so "revert to compiled" can restore it (positions and all). The
  // compiled/trained STATUS is owned by the backend (compileStore), not recorded here.
  markCompiled: () =>
    set((s) => ({
      lastCompiled: {
        nodes: cleanNodes(s.nodes),
        edges: cleanEdges(s.edges),
        meta: s.meta,
        blockStart: s.blockStart,
        blockEnd: s.blockEnd,
      },
    })),
  // restore the graph to the last compiled snapshot (discards uncompiled edits). The backend status
  // (compileStore) will re-resolve to "compiled" once model_status re-checks the restored graph.
  revertToCompiled: () =>
    set((s) => {
      const c = s.lastCompiled;
      if (!c) return {};
      return {
        nodes: c.nodes.map((n) => ({ ...n })),
        edges: c.edges.map((e) => ({ ...e })),
        meta: c.meta,
        blockStart: c.blockStart,
        blockEnd: c.blockEnd,
        selectedId: null,
      };
    }),
  setMode: (mode) => set({ mode }), // not a graph edit -> no needsCompile
  setViewport: (viewport) => set({ viewport }), // pan/zoom is cosmetic -> no needsCompile
  setSaveStatus: (saveStatus) => set({ saveStatus }),

  // emit the block only when it is a valid single-in/single-out shape-preserving slice
  toGraph: () => {
    const s = get();
    const block = analyzeBlock(s.nodes, s.edges, s.blockStart, s.blockEnd);
    return canvasToGraph(s.nodes, s.edges, s.meta, block.valid ? [...block.nodeIds] : undefined);
  },
}));

// ---- autosave: debounce-persist the graph/view state to localStorage on any meaningful change ----
// The signature excludes transient flags (selection/drag are normalized out) and non-persisted
// state (selectedId, clipboard, saveStatus), so selecting a node or flipping the indicator won't
// trigger a save loop.
const SAVE_DEBOUNCE_MS = 500;

function persistableOf(s: CanvasState): PersistedCanvas {
  return {
    nodes: cleanNodes(s.nodes),
    edges: cleanEdges(s.edges),
    meta: s.meta,
    mode: s.mode,
    modelId: s.modelId,
    lastCompiled: s.lastCompiled,
    viewport: s.viewport,
    blockStart: s.blockStart,
    blockEnd: s.blockEnd,
  };
}

let lastSig = JSON.stringify(persistableOf(useCanvasStore.getState()));
let saveTimer: ReturnType<typeof setTimeout> | null = null;

useCanvasStore.subscribe((s) => {
  const data = persistableOf(s);
  const sig = JSON.stringify(data);
  if (sig === lastSig) return; // nothing persistable changed (selection / saveStatus / clipboard)
  lastSig = sig;
  if (s.saveStatus !== "saving") useCanvasStore.setState({ saveStatus: "saving" });
  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    writePersisted(data);
    useCanvasStore.setState({ saveStatus: "saved" });
  }, SAVE_DEBOUNCE_MS);
});
