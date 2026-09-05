// Static SVG schematic of a project's graph, computed from its persisted canvas (node positions +
// sizes + category colors + edges). This is far cheaper than mounting a React Flow instance per
// card and needs no image snapshot/extra storage: it just reads the last-saved canvas and draws
// rounded rects + edge curves, scaled to fit. Reflects the last-edited state.
import { useMemo } from "react";
import { deriveBlockNodes } from "@/lib/block";
import type { YNodeData } from "@/lib/graph";
import { CATEGORY, CATEGORY_OF, nodeHeight, nodeWidth, resolveNodeDef } from "@/lib/nodeCatalog";
import { loadCanvas } from "@/lib/persist";

interface Rect {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  fill: string;
  quantized: boolean; // draw a silver border (optimistic: never flagged as unsupported)
}
interface Geo {
  rects: Rect[];
  paths: string[];
  fusionPaths: string[]; // bottom-to-bottom beams between fused nodes (always drawn green)
  block: { x: number; y: number; w: number; h: number } | null; // block region (always drawn blue)
  viewBox: string;
}

function build(id: string): Geo | null {
  const c = loadCanvas(id);
  if (!c || c.nodes.length === 0) return null;

  const rects: Rect[] = [];
  const byId = new Map<string, Rect>();
  for (const n of c.nodes) {
    const type = (n.data as YNodeData).type;
    const def = resolveNodeDef(type);
    if (!def) continue;
    const cat = CATEGORY_OF[type];
    const r: Rect = {
      id: n.id,
      x: n.position.x,
      y: n.position.y,
      w: nodeWidth(def, c.meta),
      h: nodeHeight(def),
      fill: cat ? `var(${CATEGORY[cat].accent})` : "var(--muted)",
      quantized: !!(n.data as YNodeData).quantized,
    };
    rects.push(r);
    byId.set(n.id, r);
  }
  if (rects.length === 0) return null;

  const dataEdges = c.edges.filter((e) => e.type !== "fusion");

  // data edges: side-to-side curves (data flows left->right)
  const paths: string[] = [];
  for (const e of dataEdges) {
    const s = byId.get(e.source);
    const t = byId.get(e.target);
    if (!s || !t) continue;
    const x1 = s.x + s.w;
    const y1 = s.y + s.h / 2;
    const x2 = t.x;
    const y2 = t.y + t.h / 2;
    const dx = Math.max(24, Math.abs(x2 - x1) * 0.4);
    paths.push(`M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`);
  }

  // fusion beams: bottom-diamond to bottom-diamond, mirroring the canvas beam
  const fusionPaths: string[] = [];
  for (const e of c.edges) {
    if (e.type !== "fusion") continue;
    const s = byId.get(e.source);
    const t = byId.get(e.target);
    if (!s || !t) continue;
    const x1 = s.x + s.w / 2;
    const y1 = s.y + s.h;
    const x2 = t.x + t.w / 2;
    const y2 = t.y + t.h;
    const dip = 26;
    fusionPaths.push(`M${x1},${y1} C${x1},${y1 + dip} ${x2},${y2 + dip} ${x2},${y2}`);
  }

  // block region: bbox of the derived slice (fusion edges excluded from the reachability)
  let block: Geo["block"] = null;
  if (c.blockStart && c.blockEnd) {
    const ids = deriveBlockNodes(dataEdges, c.blockStart, c.blockEnd);
    const members = rects.filter((r) => ids.has(r.id));
    if (members.length > 0) {
      const bx = Math.min(...members.map((r) => r.x));
      const by = Math.min(...members.map((r) => r.y));
      const bmaxX = Math.max(...members.map((r) => r.x + r.w));
      const bmaxY = Math.max(...members.map((r) => r.y + r.h));
      const bp = 22;
      block = { x: bx - bp, y: by - bp, w: bmaxX - bx + bp * 2, h: bmaxY - by + bp * 2 };
    }
  }

  const minX = Math.min(...rects.map((r) => r.x));
  const minY = Math.min(...rects.map((r) => r.y));
  const maxX = Math.max(...rects.map((r) => r.x + r.w));
  const maxY = Math.max(...rects.map((r) => r.y + r.h));
  const pad = 48;
  const viewBox = `${minX - pad} ${minY - pad} ${maxX - minX + pad * 2} ${maxY - minY + pad * 2}`;
  return { rects, paths, fusionPaths, block, viewBox };
}

// `rev` (the project's updatedAt) busts the memo so the thumbnail refreshes after edits
export function GraphThumbnail({ id, rev }: { id: string; rev: number }) {
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const geo = useMemo(() => build(id), [id, rev]);
  if (!geo) return <span className="mcard-glyph" />;
  return (
    <svg className="mcard-thumb" viewBox={geo.viewBox} preserveAspectRatio="xMidYMid meet" aria-hidden="true">
      {geo.block && (
        <rect
          className="mt-block"
          x={geo.block.x}
          y={geo.block.y}
          width={geo.block.w}
          height={geo.block.h}
          rx={12}
        />
      )}
      {geo.paths.map((d, i) => (
        <path key={i} className="mt-edge" d={d} />
      ))}
      {geo.fusionPaths.map((d, i) => (
        <path key={`f${i}`} className="mt-fuse" d={d} />
      ))}
      {geo.rects.map((r) => (
        <rect
          key={r.id}
          className={`mt-node${r.quantized ? " q" : ""}`}
          x={r.x}
          y={r.y}
          width={r.w}
          height={r.h}
          rx={5}
          style={{ fill: r.fill }}
        />
      ))}
    </svg>
  );
}
