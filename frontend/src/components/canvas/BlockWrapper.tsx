// The block region: a labeled box drawn behind the repeated slice, with an editable xN loop
// count. Lives in the viewport layer so it pans/zooms with the nodes.
import { useCallback } from "react";
import { ViewportPortal } from "@xyflow/react";
import { analyzeBlock } from "@/lib/block";
import type { YNodeData } from "@/lib/graph";
import { nodeHeight, nodeWidth, resolveNodeDef } from "@/lib/nodeCatalog";
import { N_LAYER_MAX, N_LAYER_MIN, useCanvasStore } from "@/store/canvasStore";

const PAD = 26; // flow px of breathing room between the nodes and the box

// stop mousedown/dblclick from reaching React Flow's pane (its zoom/pan use native d3
// listeners, so a React onDoubleClick would fire too late to prevent the zoom)
function stopGestures(el: HTMLDivElement | null) {
  if (!el) return;
  const stop = (e: Event) => e.stopPropagation();
  el.addEventListener("mousedown", stop);
  el.addEventListener("dblclick", stop);
}

export function BlockWrapper() {
  const nodes = useCanvasStore((s) => s.nodes);
  const edges = useCanvasStore((s) => s.edges);
  const meta = useCanvasStore((s) => s.meta);
  const blockStart = useCanvasStore((s) => s.blockStart);
  const blockEnd = useCanvasStore((s) => s.blockEnd);
  const setNLayer = useCanvasStore((s) => s.setNLayer);
  const clearBlock = useCanvasStore((s) => s.clearBlock);
  const headRef = useCallback(stopGestures, []);

  if (!blockStart || !blockEnd) return null;

  const block = analyzeBlock(nodes, edges, blockStart, blockEnd);
  const members = nodes.filter((n) => block.nodeIds.has(n.id));
  if (members.length === 0) return null;

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const n of members) {
    const def = resolveNodeDef((n.data as YNodeData).type);
    const w = def ? nodeWidth(def, meta) : 140;
    const h = def ? nodeHeight(def) : 90;
    minX = Math.min(minX, n.position.x);
    minY = Math.min(minY, n.position.y);
    maxX = Math.max(maxX, n.position.x + w);
    maxY = Math.max(maxY, n.position.y + h);
  }

  const style = {
    transform: `translate(${minX - PAD}px, ${minY - PAD}px)`,
    width: maxX - minX + PAD * 2,
    height: maxY - minY + PAD * 2,
  };

  const invalid = block.valid ? "" : " invalid";
  return (
    <ViewportPortal>
      {/* fill sinks behind nodes + edges */}
      <div className={`block-fill${invalid}`} style={style} />
      {/* frame (border) + header + error float in front */}
      <div className={`block-frame${invalid}`} style={style}>
        <div className="block-head" ref={headRef}>
          <span className="block-title">Block</span>
          <span className="block-x">
            <button
              type="button"
              aria-label="Fewer layers"
              disabled={meta.n_layer <= N_LAYER_MIN}
              onClick={() => setNLayer(meta.n_layer - 1)}
            >
              &minus;
            </button>
            <span className="block-count">&times;{meta.n_layer}</span>
            <button
              type="button"
              aria-label="More layers"
              disabled={meta.n_layer >= N_LAYER_MAX}
              onClick={() => setNLayer(meta.n_layer + 1)}
            >
              +
            </button>
          </span>
          <button type="button" className="block-clear" aria-label="Dissolve block" onClick={clearBlock}>
            &times;
          </button>
        </div>
        {!block.valid && <div className="block-err">{block.error}</div>}
      </div>
    </ViewportPortal>
  );
}
