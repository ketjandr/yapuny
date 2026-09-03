// Derive the block node set from its start/end markers and validate it can loop:
// a single-input / single-output slice whose output shape matches its input (shape-preserving).
import type { Edge, Node } from "@xyflow/react";
import type { YNodeData } from "./graph";
import { type Axis, resolveNodeDef } from "./nodeCatalog";

export interface BlockAnalysis {
  nodeIds: Set<string>;
  valid: boolean;
  error?: string;
  problemEdgeIds?: string[]; // edges to highlight for the current error (multi in/out)
}

function adjacency(edges: Edge[], reverse: boolean): Map<string, string[]> {
  const adj = new Map<string, string[]>();
  for (const e of edges) {
    const [from, to] = reverse ? [e.target, e.source] : [e.source, e.target];
    if (!adj.has(from)) adj.set(from, []);
    adj.get(from)!.push(to);
  }
  return adj;
}

function reachable(adj: Map<string, string[]>, start: string): Set<string> {
  const seen = new Set<string>([start]);
  const stack = [start];
  while (stack.length) {
    for (const next of adj.get(stack.pop()!) ?? []) {
      if (!seen.has(next)) {
        seen.add(next);
        stack.push(next);
      }
    }
  }
  return seen;
}

// block = nodes reachable from start AND able to reach end (i.e. on a path start -> end)
export function deriveBlockNodes(edges: Edge[], startId: string, endId: string): Set<string> {
  const forward = reachable(adjacency(edges, false), startId);
  const backward = reachable(adjacency(edges, true), endId);
  return new Set([...forward].filter((id) => backward.has(id)));
}

function portShape(node: Node | undefined, port: string | null | undefined, output: boolean) {
  const def = resolveNodeDef((node?.data as YNodeData | undefined)?.type ?? "");
  const ports = output ? def?.outputs : def?.inputs;
  return ports?.find((p) => p.id === port)?.shape;
}

const sameShape = (a?: Axis[], b?: Axis[]) =>
  !!a && !!b && a.length === b.length && a.every((x, i) => x === b[i]);

export function analyzeBlock(
  nodes: Node[],
  edges: Edge[],
  startId: string | null,
  endId: string | null,
): BlockAnalysis {
  if (!startId || !endId) return { nodeIds: new Set(), valid: false, error: "mark a block start and end" };

  const nodeIds = deriveBlockNodes(edges, startId, endId);
  if (!nodeIds.has(startId) || !nodeIds.has(endId)) {
    return { nodeIds, valid: false, error: "the block end must be downstream of the block start" };
  }

  // A loopable block reads exactly ONE external tensor and emits exactly ONE. That single
  // input tensor may fan out to several internal target ports (e.g. the pre-norm residual
  // stream feeds both ln.x and res.residual); on unroll the single output fans back to all
  // of them, so the wiring stays unambiguous. Zero or multiple distinct tensors break that.
  const inEdges = edges.filter((e) => nodeIds.has(e.target) && !nodeIds.has(e.source));
  const outEdges = edges.filter((e) => nodeIds.has(e.source) && !nodeIds.has(e.target));
  const inTensors = new Set(inEdges.map((e) => `${e.source}.${e.sourceHandle}`));
  const outTensors = new Set(outEdges.map((e) => `${e.source}.${e.sourceHandle}`));

  if (inTensors.size === 0)
    return { nodeIds, valid: false, error: "block has no input tensors" };
  if (inTensors.size > 1)
    return {
      nodeIds,
      valid: false,
      error: `block has ${inTensors.size} input tensors, it must have exactly one`,
      problemEdgeIds: inEdges.map((e) => e.id),
    };

  if (outTensors.size === 0)
    return { nodeIds, valid: false, error: "block has no output tensors" };
  if (outTensors.size > 1)
    return {
      nodeIds,
      valid: false,
      error: `block emits ${outTensors.size} output tensors, it must emit exactly one`,
      problemEdgeIds: outEdges.map((e) => e.id),
    };

  // loop-back: block output must be shape-compatible with each input target, or it can't stack
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const exit = outEdges[0];
  const exitShape = portShape(byId.get(exit.source), exit.sourceHandle, true);
  const mismatched = inEdges.filter(
    (e) => !sameShape(exitShape, portShape(byId.get(e.target), e.targetHandle, false)),
  );
  if (mismatched.length > 0) {
    return {
      nodeIds,
      valid: false,
      error: "block output shape must match its input shape",
      // the exit tensor's edge + each input edge it can't feed back into
      problemEdgeIds: [...outEdges.map((e) => e.id), ...mismatched.map((e) => e.id)],
    };
  }
  return { nodeIds, valid: true };
}
