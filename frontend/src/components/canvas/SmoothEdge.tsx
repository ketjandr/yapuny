// Custom edge: plain bezier when clear, one smooth arch when it must clear a node row.
import { BaseEdge, getBezierPath, useNodes, type EdgeProps, type Node } from "@xyflow/react";
import { nodeHeight, resolveNodeDef } from "@/lib/nodeCatalog";
import type { YNodeData } from "@/lib/graph";
import { useCanvasStore } from "@/store/canvasStore";

const BASE_CLEAR = 38; // min lift over the obstacle row
const LIFT_PER_PX = 0.17; // longer skips arch higher
const MAX_LIFT = 280; // cap for cross-canvas skips
const MAX_HOFF = 110; // fixed control offset -> steep takeoff clears adjacent nodes
const PAD = 8; // arch a touch before edges would touch

interface Rect {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

function rectOf(n: Node): Rect {
  const def = resolveNodeDef((n.data as YNodeData)?.type);
  const w = n.measured?.width ?? 140; // rough fallback before React Flow measures the node
  const h = n.measured?.height ?? (def ? nodeHeight(def) : 0);
  return { left: n.position.x, right: n.position.x + w, top: n.position.y, bottom: n.position.y + h };
}

// rects the straight source->target segment passes through
function collisions(sx: number, sy: number, tx: number, ty: number, rects: Rect[]): Rect[] {
  const loX = Math.min(sx, tx);
  const hiX = Math.max(sx, tx);
  const dx = tx - sx;
  const yAt = (x: number) => (dx === 0 ? sy : sy + ((ty - sy) * (x - sx)) / dx);
  const hit: Rect[] = [];
  for (const r of rects) {
    const a = Math.max(r.left, loX);
    const b = Math.min(r.right, hiX);
    if (a > b) continue; // no horizontal overlap
    const ya = yAt(a);
    const yb = yAt(b);
    if (Math.max(ya, yb) >= r.top - PAD && Math.min(ya, yb) <= r.bottom + PAD) hit.push(r);
  }
  return hit;
}

export function SmoothEdge(props: EdgeProps) {
  const {
    id,
    source,
    target,
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    markerEnd,
    style,
  } = props;
  const nodes = useNodes();
  const mode = useCanvasStore((s) => s.mode);

  // grey edges touching a node that's inactive in this mode (e.g. kv_cache while training)
  const dimmed =
    mode === "train" &&
    nodes.some(
      (n) =>
        (n.id === source || n.id === target) &&
        resolveNodeDef((n.data as YNodeData)?.type)?.trainingNoop,
    );

  let path: string;
  const rects = nodes.filter((n) => n.id !== source && n.id !== target).map(rectOf);
  const hits = collisions(sourceX, sourceY, targetX, targetY, rects);

  if (hits.length === 0) {
    [path] = getBezierPath({ sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition });
  } else {
    const top = Math.min(...hits.map((r) => r.top));
    const bottom = Math.max(...hits.map((r) => r.bottom));
    const dx = targetX - sourceX;
    const span = Math.abs(dx);
    const dir = Math.sign(dx) || 1;

    const lift = Math.min(BASE_CLEAR + span * LIFT_PER_PX, MAX_LIFT);
    const midY = (sourceY + targetY) / 2;
    const goUp = midY - (top - lift) <= bottom + lift - midY; // whichever bends less
    const controlY = goUp ? top - lift : bottom + lift;

    const hoff = Math.min(span * 0.5, MAX_HOFF);
    const c1x = sourceX + dir * hoff;
    const c2x = targetX - dir * hoff;
    path = `M${sourceX},${sourceY} C${c1x},${controlY} ${c2x},${controlY} ${targetX},${targetY}`;
  }

  return (
    <BaseEdge
      id={id}
      path={path}
      markerEnd={markerEnd}
      style={dimmed ? { ...style, opacity: 0.28 } : style}
    />
  );
}
