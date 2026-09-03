// Fusion tether edge: renders the shared fusion beam between two nodes' bottom diamond ports.
import type { EdgeProps } from "@xyflow/react";
import { FusionBeam } from "./FusionBeam";

export function FusionEdge({ sourceX, sourceY, targetX, targetY }: EdgeProps) {
  return <FusionBeam sourceX={sourceX} sourceY={sourceY} targetX={targetX} targetY={targetY} interactive />;
}
