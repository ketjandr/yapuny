// Canvas nodes/edges <-> backend GraphRequest. Handle ids are the backend port names.
import type { Edge, Node } from "@xyflow/react";
import { DEFAULT_EDGES, DEFAULT_LAYOUT, type SeedEdge } from "./defaultGraph";
import type { EdgeSchema, GraphMetaSchema, GraphRequest, NodeSchema } from "./types";

export const RF_NODE_TYPE = "graph";

export interface YNodeData extends Record<string, unknown> {
  type: string; // backend node type key (or "_input")
  quantized: string | null; // "w8" | "w4" | null
}

const INPUT_POS = { x: -176, y: 120 }; // left of the pipeline (feeds tok/pos emb)

function rfNode(id: string, type: string, x: number, y: number, quantized: string | null = null): Node {
  return {
    id,
    type: RF_NODE_TYPE,
    position: { x, y },
    data: { type, quantized } satisfies YNodeData,
    deletable: type !== "_input", // the _input pseudo-node is not user-deletable
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

// fresh canvas: the _input node + the wired seed layout
export function seedToCanvas(): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = [
    rfNode("_input", "_input", INPUT_POS.x, INPUT_POS.y),
    ...DEFAULT_LAYOUT.map((n) => rfNode(n.id, n.type, n.x, n.y)),
  ];
  const edges: Edge[] = DEFAULT_EDGES.map(seedEdgeToRf);
  return { nodes, edges };
}

// canvas -> GraphRequest; _input is dropped from nodes (compiler seeds it), its edges kept
export function canvasToGraph(nodes: Node[], edges: Edge[], meta: GraphMetaSchema): GraphRequest {
  const schemaNodes: NodeSchema[] = nodes
    .filter((n) => (n.data as YNodeData).type !== "_input")
    .map((n) => {
      const d = n.data as YNodeData;
      return { id: n.id, type: d.type, config: {}, quantized: d.quantized ?? null };
    });

  const schemaEdges: EdgeSchema[] = edges.map((e) => ({
    from_node: e.source,
    to_node: e.target,
    from_port: e.sourceHandle ?? "out",
    to_port: e.targetHandle ?? "x",
  }));

  return { nodes: schemaNodes, edges: schemaEdges, fusion_groups: [], meta };
}

// GraphRequest -> canvas, grid-laid out (no positions in the schema); for loading saved models
export function graphToCanvas(graph: GraphRequest): { nodes: Node[]; edges: Edge[] } {
  const COLS = 6;
  const nodes: Node[] = graph.nodes.map((n, i) =>
    rfNode(n.id, n.type, (i % COLS) * 212, Math.floor(i / COLS) * 170, n.quantized ?? null),
  );
  if (graph.edges.some((e) => e.from_node === "_input")) {
    nodes.unshift(rfNode("_input", "_input", INPUT_POS.x, INPUT_POS.y));
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

export function makeNode(type: string, position: { x: number; y: number }): Node {
  const id = `${type}_${Math.random().toString(36).slice(2, 7)}`;
  return rfNode(id, type, position.x, position.y);
}
