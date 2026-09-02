// Canvas: renders the graph store via React Flow; palette drop adds nodes.
import { useCallback, useState } from "react";
import {
  Background,
  BackgroundVariant,
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
import { ModeToggle } from "./ModeToggle";
import { ContextMenu, type CanvasMenu } from "./ContextMenu";
import { useCanvasShortcuts } from "./useCanvasShortcuts";
import { ValidationOverlay } from "./ValidationOverlay";
import { useCanvasStore } from "@/store/canvasStore";
import type { YNodeData } from "@/lib/graph";
import { PALETTE_MIME } from "@/components/sidebar/NodePalette";

const nodeTypes = { graph: GraphNode };
const edgeTypes = { default: SmoothEdge };

function CanvasInner() {
  const nodes = useCanvasStore((s) => s.nodes);
  const edges = useCanvasStore((s) => s.edges);
  const onNodesChange = useCanvasStore((s) => s.onNodesChange);
  const onEdgesChange = useCanvasStore((s) => s.onEdgesChange);
  const onConnect = useCanvasStore((s) => s.onConnect);
  const onReconnect = useCanvasStore((s) => s.onReconnect);
  const setSelected = useCanvasStore((s) => s.setSelected);
  const addNode = useCanvasStore((s) => s.addNode);

  const { screenToFlowPosition } = useReactFlow();
  useCanvasShortcuts();

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
    <div className="cv" onDrop={onDrop} onDragOver={onDragOver}>
      <ReactFlow
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onReconnect={onReconnect}
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
        connectionRadius={34}
        fitView
        minZoom={0.2}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1.5} color="#2b3340" />
        <Controls showInteractive={false} />
        <MiniMap pannable zoomable />
      </ReactFlow>
      <button
        type="button"
        className={`box-select${boxSelect ? " a" : ""}`}
        title="Box select — drag to select nodes and edges"
        aria-pressed={boxSelect}
        onClick={() => setBoxSelect((v) => !v)}
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
      <ModeToggle />
      <ValidationOverlay />
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
