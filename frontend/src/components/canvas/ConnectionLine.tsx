// The line shown while dragging a connection. From a fusion (diamond) port it previews the
// fusion beam; otherwise it falls back to the normal bezier connection line.
import { type ConnectionLineComponentProps, getBezierPath } from "@xyflow/react";
import { FUSE_PORT } from "@/lib/fusion";
import { FusionBeam } from "./FusionBeam";

export function ConnectionLine({
  fromX,
  fromY,
  toX,
  toY,
  fromPosition,
  toPosition,
  fromHandle,
}: ConnectionLineComponentProps) {
  if (fromHandle?.id === FUSE_PORT) {
    return <FusionBeam sourceX={fromX} sourceY={fromY} targetX={toX} targetY={toY} />;
  }
  const [path] = getBezierPath({
    sourceX: fromX,
    sourceY: fromY,
    sourcePosition: fromPosition,
    targetX: toX,
    targetY: toY,
    targetPosition: toPosition,
  });
  return <path className="react-flow__connection-path" d={path} fill="none" />;
}
