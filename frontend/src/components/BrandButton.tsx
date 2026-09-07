// The clickable Yapuny brand mark; returns to the models home. Shared by the app navbar and the
// models page so the logo behaves identically everywhere.
import { useNavigate } from "react-router-dom";

// Logo: three upright rounded square panels fanned in a row, like standing dominoes = a transformer's
// stacked blocks. Faces stand straight (no skew); a darker rim behind each shows the slab thickness.
// Front panel brightest -> back darkest for depth (minimalist white/grey, opaque not glassy). Drawn
// back->front so the front panel sits over the others.
function BrandMark() {
  const panel = (tx: number, ty: number, face: string, edge: string) => (
    <g key={tx} transform={`translate(${tx} ${ty})`}>
      <rect x="-1.8" y="1.8" width="15" height="15" rx="3" fill={edge} />
      <rect x="0" y="0" width="15" height="15" rx="3" fill={face} />
    </g>
  );
  return (
    <svg className="mark" viewBox="0 0 34 27" fill="none" aria-hidden="true">
      {panel(2, 2, "#8b93a3", "#5f6675")}
      {panel(10, 5, "var(--txt2)", "#7a8291")}
      {panel(18, 8, "var(--txt)", "#aab2be")}
    </svg>
  );
}

export function BrandButton() {
  const navigate = useNavigate();
  return (
    <button className="brand-btn" type="button" onClick={() => navigate("/")} title="Back to models">
      <BrandMark />
      <span className="brand-name">Yapuny</span>
    </button>
  );
}
