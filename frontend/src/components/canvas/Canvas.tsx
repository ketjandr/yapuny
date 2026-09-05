// Canvas: renders the graph store via React Flow; palette drop adds nodes.
import { useCallback, useMemo, useState } from "react";
import {
  Background,
  BackgroundVariant,
  type Connection,
  Controls,
  type Edge,
  MiniMap,
  type Node,
  ReactFlow,
  ReactFlowProvider,
  SelectionMode,
  useReactFlow,
} from "@xyflow/react";
import { GraphNode } from "./GraphNode";
import { SmoothEdge } from "./SmoothEdge";
import { FusionEdge } from "./FusionEdge";
import { ConnectionLine } from "./ConnectionLine";
import { BlockWrapper } from "./BlockWrapper";
import { ModeToggle } from "./ModeToggle";
import { ContextMenu, type CanvasMenu } from "./ContextMenu";
import { useCanvasShortcuts } from "./useCanvasShortcuts";
import { ValidationOverlay } from "./ValidationOverlay";
import { SaveBar, CompileBar, RevertTool } from "./CanvasStatus";
import { useTooltip } from "@/components/tooltipContext";
import { useQuery } from "@tanstack/react-query";
import { useCanvasStore } from "@/store/canvasStore";
import { analyzeBlock } from "@/lib/block";
import { api } from "@/lib/api";
import { FUSE_PORT, type FusionCatalog, fusionVisible, validateFusion } from "@/lib/fusion";
import type { YNodeData } from "@/lib/graph";
import { PALETTE_MIME } from "@/components/panes/NodePalette";

const nodeTypes = { graph: GraphNode };
const edgeTypes = { default: SmoothEdge, fusion: FusionEdge };

function CanvasInner() {
  const nodes = useCanvasStore((s) => s.nodes);
  const edges = useCanvasStore((s) => s.edges);
  const onNodesChange = useCanvasStore((s) => s.onNodesChange);
  const onEdgesChange = useCanvasStore((s) => s.onEdgesChange);
  const onConnect = useCanvasStore((s) => s.onConnect);
  const onReconnect = useCanvasStore((s) => s.onReconnect);
  const setSelected = useCanvasStore((s) => s.setSelected);
  const addNode = useCanvasStore((s) => s.addNode);
  const blockStart = useCanvasStore((s) => s.blockStart);
  const blockEnd = useCanvasStore((s) => s.blockEnd);
  const mode = useCanvasStore((s) => s.mode); // only drives the data-fusion CSS marker below
  const setViewport = useCanvasStore((s) => s.setViewport);
  // read the persisted viewport once at mount: restore it via defaultViewport, else fitView
  const initialViewport = useMemo(() => useCanvasStore.getState().viewport, []);

  // cached-per-session fusion catalog; validate fusions locally against it (backend re-checks at
  // compile). Undefined (offline / not fetched) means no flags - optimistic.
  const { data: fusionCatalog } = useQuery<FusionCatalog>({
    queryKey: ["fusion-available"],
    queryFn: api.fusionAvailable,
    staleTime: Infinity,
    retry: false,
  });
  const fusion = useMemo(
    () => validateFusion(nodes, edges, fusionCatalog),
    [nodes, edges, fusionCatalog],
  );

  // same pattern for quantization: if the worker can't quantize, flag any quantized node red
  // (matches the backend "quantization is not supported on this worker" validation error)
  const { data: quantCatalog } = useQuery<{ available: boolean }>({
    queryKey: ["quantization-available"],
    queryFn: api.quantizationAvailable,
    staleTime: Infinity,
    retry: false,
  });
  const quantBad = useMemo(() => {
    const bad = new Set<string>();
    // quantization is inference-only, so only flag it (red) while in inference mode
    if (quantCatalog && !quantCatalog.available && fusionVisible(mode)) {
      for (const n of nodes) if ((n.data as YNodeData).quantized) bad.add(n.id);
    }
    return bad;
  }, [nodes, quantCatalog, mode]);

  // mark the edges the block error blames (multi input/output tensors) + invalid fusion beams;
  // fusion edges already carry their own type and render as the beam
  const displayEdges = useMemo(() => {
    const blockBad = new Set(analyzeBlock(nodes, edges, blockStart, blockEnd).problemEdgeIds ?? []);
    if (blockBad.size === 0 && fusion.badEdges.size === 0) return edges;
    return edges.map((e) => {
      const extra = [
        blockBad.has(e.id) ? "edge-error" : "",
        fusion.badEdges.has(e.id) ? "fuse-bad" : "",
      ].filter(Boolean);
      return extra.length ? { ...e, className: [e.className, ...extra].filter(Boolean).join(" ") } : e;
    });
  }, [nodes, edges, blockStart, blockEnd, fusion]);

  // flag red: nodes in an invalid fusion group, and quantized nodes the worker can't quantize
  const displayNodes = useMemo(() => {
    if (fusion.badNodes.size === 0 && quantBad.size === 0) return nodes;
    return nodes.map((n) => {
      const extra = [
        fusion.badNodes.has(n.id) ? "fuse-bad" : "",
        quantBad.has(n.id) ? "quant-bad" : "",
      ].filter(Boolean);
      return extra.length ? { ...n, className: [n.className, ...extra].filter(Boolean).join(" ") } : n;
    });
  }, [nodes, fusion, quantBad]);

  const { screenToFlowPosition } = useReactFlow();
  useCanvasShortcuts();

  // a fusion (diamond) port only connects to another fusion port; data ports only to data ports
  const isValidConnection = useCallback((c: Connection | Edge) => {
    return (c.sourceHandle === FUSE_PORT) === (c.targetHandle === FUSE_PORT);
  }, []);

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const type = e.dataTransfer.getData(PALETTE_MIME);
      if (!type) return;
      const position = screenToFlowPosition({ x: e.clientX, y: e.clientY });
      addNode(type, position);
    },
    [screenToFlowPosition, addNode],
  );

  const [menu, setMenu] = useState<CanvasMenu | null>(null);
  const closeMenu = useCallback(() => setMenu(null), []);
  const [boxSelect, setBoxSelect] = useState(false);
  const boxTip = useTooltip("Multi-select nodes and edges");

  const onNodeContextMenu = useCallback((e: React.MouseEvent, n: Node) => {
    e.preventDefault();
    if ((n.data as YNodeData).type.startsWith("_")) return; // no actions on _input/_output
    setMenu({ x: e.clientX, y: e.clientY, kind: "node", id: n.id });
  }, []);

  const onEdgeContextMenu = useCallback((e: React.MouseEvent, ed: Edge) => {
    e.preventDefault();
    setMenu({ x: e.clientX, y: e.clientY, kind: "edge", id: ed.id });
  }, []);

  const onPaneContextMenu = useCallback((e: MouseEvent | React.MouseEvent) => {
    e.preventDefault();
    setMenu({ x: e.clientX, y: e.clientY, kind: "pane" });
  }, []);

  return (
    <div className="cv" data-fusion={fusionVisible(mode) ? "on" : "off"} onDrop={onDrop} onDragOver={onDragOver}>
      {/* shared filter: static turbulence displacement gives the fused-node border its wispy,
          crackling edge (computed once - motion elsewhere is cheap CSS, not an animated filter) */}
      <svg className="fusion-defs" aria-hidden="true">
        <defs>
          <filter id="fusion-turb" x="-60%" y="-120%" width="220%" height="340%">
            <feTurbulence type="fractalNoise" baseFrequency="0.014 0.022" numOctaves="2" seed="7" result="n" />
            <feDisplacementMap in="SourceGraphic" in2="n" scale="8" xChannelSelector="R" yChannelSelector="G" />
          </filter>
          {/* displace the flowing energy into irregular random shapes (noise is static/cached) */}
          <filter id="fusion-flow" x="-80%" y="-200%" width="260%" height="500%">
            <feTurbulence type="fractalNoise" baseFrequency="0.016 0.03" numOctaves="2" seed="5" result="n" />
            <feDisplacementMap in="SourceGraphic" in2="n" scale="16" xChannelSelector="R" yChannelSelector="G" />
          </filter>
          <filter id="fusion-flow2" x="-80%" y="-200%" width="260%" height="500%">
            <feTurbulence type="fractalNoise" baseFrequency="0.024 0.042" numOctaves="2" seed="14" result="n" />
            <feDisplacementMap in="SourceGraphic" in2="n" scale="13" xChannelSelector="R" yChannelSelector="G" />
          </filter>
          {/* shared flowing-energy gradients (green + red), defined once so their SMIL clock never
              restarts - a per-edge gradient re-created each drag frame was the freeze bug. A fixed
              horizontal repeat tiles all of user space, so every beam samples it wherever it sits. */}
          <linearGradient id="fbg-shared" gradientUnits="userSpaceOnUse" x1={0} y1={0} x2={78} y2={0} spreadMethod="repeat">
            <stop offset="0" stopColor="#5be39a" stopOpacity="0" />
            <stop offset="0.18" stopColor="#8ff2cd" stopOpacity="0.55" />
            <stop offset="0.4" stopColor="#5be39a" stopOpacity="0" />
            <stop offset="1" stopColor="#5be39a" stopOpacity="0" />
            <animateTransform attributeName="gradientTransform" type="translate" from="0 0" to="78 0" dur="2.8s" repeatCount="indefinite" />
          </linearGradient>
          <linearGradient id="fbg-shared-bad" gradientUnits="userSpaceOnUse" x1={0} y1={0} x2={78} y2={0} spreadMethod="repeat">
            <stop offset="0" stopColor="#d9738a" stopOpacity="0" />
            <stop offset="0.18" stopColor="#ffc2cc" stopOpacity="0.6" />
            <stop offset="0.4" stopColor="#d9738a" stopOpacity="0" />
            <stop offset="1" stopColor="#d9738a" stopOpacity="0" />
            <animateTransform attributeName="gradientTransform" type="translate" from="0 0" to="78 0" dur="2.8s" repeatCount="indefinite" />
          </linearGradient>
        </defs>
      </svg>
      <ReactFlow
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        nodes={displayNodes}
        edges={displayEdges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onReconnect={onReconnect}
        isValidConnection={isValidConnection}
        connectionLineComponent={ConnectionLine}
        onNodeClick={(_, n) => setSelected(n.id)}
        onPaneClick={() => setSelected(null)}
        onNodeContextMenu={onNodeContextMenu}
        onEdgeContextMenu={onEdgeContextMenu}
        onPaneContextMenu={onPaneContextMenu}
        multiSelectionKeyCode={["Meta", "Control"]}
        selectionOnDrag={boxSelect}
        selectionMode={SelectionMode.Partial}
        panOnDrag={boxSelect ? [1, 2] : true}
        onSelectionEnd={() => setBoxSelect(false)}
        onMoveEnd={(_, vp) => setViewport(vp)}
        connectionRadius={34}
        defaultViewport={initialViewport ?? undefined}
        fitView={!initialViewport}
        minZoom={0.2}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1.5} color="#2b3340" />
        <BlockWrapper />
        <Controls showInteractive={false} />
        <MiniMap pannable zoomable />
      </ReactFlow>
      <button
        type="button"
        className={`box-select${boxSelect ? " a" : ""}`}
        aria-label="Multi-select nodes and edges"
        aria-pressed={boxSelect}
        onClick={() => setBoxSelect((v) => !v)}
        {...boxTip}
      >
        <svg
          viewBox="0 0 24 24"
          width="16"
          height="16"
          fill="none"
          stroke="currentColor"
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <rect x="3" y="3" width="18" height="18" rx="2" strokeDasharray="4 3" />
        </svg>
      </button>
      <RevertTool />
      <ModeToggle />
      <ValidationOverlay />
      <SaveBar />
      <CompileBar />
      {menu && <ContextMenu menu={menu} onClose={closeMenu} />}
    </div>
  );
}

export function Canvas() {
  return (
    <ReactFlowProvider>
      <CanvasInner />
    </ReactFlowProvider>
  );
}
