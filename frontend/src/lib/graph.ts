// Canvas nodes/edges <-> backend GraphRequest. Handle ids are the backend port names.
import type { Edge, Node } from "@xyflow/react";
import { DEFAULT_EDGES, DEFAULT_LAYOUT, INPUT_POS, OUTPUT_POS, type SeedEdge } from "./defaultGraph";
import { FUSE_PORT, isFusionEdge } from "./fusion";
import type { EdgeSchema, GraphMetaSchema, GraphRequest, NodeSchema } from "./types";

export const RF_NODE_TYPE = "graph";

export interface YNodeData extends Record<string, unknown> {
  type: string; // backend node type key (or _input / _output)
  quantized: string | null; // "w8" | "w4" | null
}

const isPseudo = (type: string) => type === "_input" || type === "_output";

function rfNode(id: string, type: string, x: number, y: number, quantized: string | null = null): Node {
  return {
    id,
    type: RF_NODE_TYPE,
    position: { x, y },
    data: { type, quantized } satisfies YNodeData,
    deletable: !isPseudo(type), // the _input / _output pseudo-nodes are not user-deletable
  };
}

function seedEdgeToRf(e: SeedEdge): Edge {
  return {
    id: `${e.from}.${e.fromPort}->${e.to}.${e.toPort}`,
    source: e.from,
    sourceHandle: e.fromPort,
    target: e.to,
    targetHandle: e.toPort,
  };
}

// fresh canvas: the _input / _output pseudo-nodes + the wired seed layout
export function seedToCanvas(): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = [
    rfNode("_input", "_input", INPUT_POS.x, INPUT_POS.y),
    ...DEFAULT_LAYOUT.map((n) => rfNode(n.id, n.type, n.x, n.y)),
    rfNode("_output", "_output", OUTPUT_POS.x, OUTPUT_POS.y),
  ];
  const edges: Edge[] = DEFAULT_EDGES.map(seedEdgeToRf);
  return { nodes, edges };
}

// canvas -> GraphRequest. The _input / _output pseudo-nodes are dropped as nodes, but their edges
// are kept: _input seeds the graph, _output anchors the sink so validation requires a complete
// input -> output chain. The compiler treats both as pseudo-endpoints (never built or executed).
export function canvasToGraph(
  nodes: Node[],
  edges: Edge[],
  meta: GraphMetaSchema,
  blockNodeIds?: string[],
): GraphRequest {
  const schemaNodes: NodeSchema[] = nodes
    .filter((n) => !isPseudo((n.data as YNodeData).type))
    .map((n) => {
      const d = n.data as YNodeData;
      return { id: n.id, type: d.type, config: {}, quantized: d.quantized ?? null };
    });

  const schemaEdges: EdgeSchema[] = edges
    .filter((e) => !isFusionEdge(e)) // fusion is visual-only; the _output edge is the sink anchor
    .map((e) => ({
      from_node: e.source,
      to_node: e.target,
      from_port: e.sourceHandle ?? "out",
      to_port: e.targetHandle ?? "x",
    }));

  const graph: GraphRequest = { nodes: schemaNodes, edges: schemaEdges, fusion_groups: [], meta };
  if (blockNodeIds?.length) graph.block = { nodes: blockNodeIds };
  return graph;
}

// GraphRequest -> canvas, grid-laid out (no positions in the schema); for loading saved models
export function graphToCanvas(graph: GraphRequest): { nodes: Node[]; edges: Edge[] } {
  const COLS = 6;
  const nodes: Node[] = graph.nodes.map((n, i) =>
    rfNode(n.id, n.type, (i % COLS) * 212, Math.floor(i / COLS) * 170, n.quantized ?? null),
  );
  // _input / _output are pseudo-endpoints: not in graph.nodes, present only as edge endpoints
  if (graph.edges.some((e) => e.from_node === "_input")) {
    nodes.unshift(rfNode("_input", "_input", INPUT_POS.x, INPUT_POS.y));
  }
  if (graph.edges.some((e) => e.to_node === "_output")) {
    nodes.push(rfNode("_output", "_output", OUTPUT_POS.x, OUTPUT_POS.y));
  }
  const edges: Edge[] = graph.edges.map((e) => ({
    id: `${e.from_node}.${e.from_port ?? "out"}->${e.to_node}.${e.to_port ?? "x"}`,
    source: e.from_node,
    sourceHandle: e.from_port ?? "out",
    target: e.to_node,
    targetHandle: e.to_port ?? "x",
  }));
  return { nodes, edges };
}

export function makeNode(
  type: string,
  position: { x: number; y: number },
  quantized: string | null = null,
): Node {
  const id = `${type}_${Math.random().toString(36).slice(2, 7)}`;
  return rfNode(id, type, position.x, position.y, quantized);
}

export function makeEdge(
  source: string,
  sourceHandle: string | null | undefined,
  target: string,
  targetHandle: string | null | undefined,
): Edge {
  return {
    id: `${source}.${sourceHandle}->${target}.${targetHandle}`,
    source,
    sourceHandle: sourceHandle ?? undefined,
    target,
    targetHandle: targetHandle ?? undefined,
  };
}

// a fusion stream between two nodes' bottom diamond ports (visual-only)
export function makeFusionEdge(source: string, target: string): Edge {
  return {
    id: `fuse:${source}->${target}`,
    type: "fusion",
    source,
    sourceHandle: FUSE_PORT,
    target,
    targetHandle: FUSE_PORT,
  };
}
