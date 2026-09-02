// Canvas: renders the graph store via React Flow; palette drop adds nodes.
import { useCallback } from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
} from "@xyflow/react";
import { GraphNode } from "./GraphNode";
import { SmoothEdge } from "./SmoothEdge";
import { ModeToggle } from "./ModeToggle";
import { ValidationOverlay } from "./ValidationOverlay";
import { useCanvasStore } from "@/store/canvasStore";
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
        connectionRadius={34}
        fitView
        minZoom={0.2}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1.5} color="#2b3340" />
        <Controls showInteractive={false} />
        <MiniMap pannable zoomable />
      </ReactFlow>
      <ModeToggle />
      <ValidationOverlay />
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
