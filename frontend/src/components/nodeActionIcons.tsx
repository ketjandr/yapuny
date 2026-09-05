// Small glyphs shared by the node action controls (Inspector toolbar + right-click menu):
// quantize (a "compress" mark) and the block start/end brackets.

export function QuantIcon() {
  // "compress" glyph (arrows pulled inward) -> reduce weight precision
  return (
    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polyline points="15 4 15 9 20 9" />
      <polyline points="9 20 9 15 4 15" />
      <line x1="15" y1="9" x2="21" y2="3" />
      <line x1="3" y1="21" x2="9" y2="15" />
    </svg>
  );
}

export function BracketIcon({ side }: { side: "start" | "end" }) {
  const d = side === "start" ? "M16 4 H8 V20 H16" : "M8 4 H16 V20 H8";
  return (
    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={d} />
    </svg>
  );
}
