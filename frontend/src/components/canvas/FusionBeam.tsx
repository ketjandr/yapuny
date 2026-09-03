// The fusion beam visual, shared by the committed edge (FusionEdge) and the drag preview
// (ConnectionLine): a steady green beam (glow + core) with irregular translucent energy strands
// flowing source -> target. The flowing gradient is shared (defined once in Canvas defs) and the
// energy stroke references it via CSS - so nothing per-instance is created and the SMIL flow can
// never restart (the "frozen for a second on drag" bug). `interactive` adds a hit path.
function fusionBeamPath(sx: number, sy: number, tx: number, ty: number): string {
  const dx = tx - sx;
  const bow = Math.min(Math.abs(dx) * 0.4 + 34, 120); // how far the energy dips below the ports
  const c1x = sx + dx * 0.25;
  const c2x = tx - dx * 0.25;
  return `M${sx},${sy} C${c1x},${sy + bow} ${c2x},${ty + bow} ${tx},${ty}`;
}

interface Props {
  sourceX: number;
  sourceY: number;
  targetX: number;
  targetY: number;
  interactive?: boolean;
}

export function FusionBeam({ sourceX, sourceY, targetX, targetY, interactive = false }: Props) {
  const d = fusionBeamPath(sourceX, sourceY, targetX, targetY);
  return (
    <g className="fusion-beam">
      {interactive && <path className="fb-hit" d={d} />}
      <path className="fb-haze" d={d} />
      <path className="fb-base" d={d} />
      <path className="fb-energy" d={d} />
      <path className="fb-energy2" d={d} />
      {/* white overlay shown only when the edge is selected (via .react-flow__edge.selected) */}
      <path className="fb-select" d={d} />
    </g>
  );
}
