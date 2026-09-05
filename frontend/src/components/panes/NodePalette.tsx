// NodePalette: the Node Registry. Drag any node type onto the canvas, grouped by category.
import type { CSSProperties } from "react";
import { DEFAULT_META } from "@/lib/defaultGraph";
import {
  CATALOG_ORDER,
  CATEGORY,
  CATEGORY_OF,
  NODE_CATALOG,
  type NodeCategory,
  resolveSubtitle,
} from "@/lib/nodeCatalog";

export const PALETTE_MIME = "application/yapuny-node";

const ORDER: NodeCategory[] = ["embedding", "attention", "mlp", "norm", "head"];

// the node type being dragged, shared out-of-band: getData() is blocked during dragover, so the
// canvas reads this to render its live preview. Cleared on dragend.
let dragging: string | null = null;
export const draggingType = () => dragging;

// a preloaded transparent 1x1 gif; used as the drag image to hide the browser's default snapshot
// of the chip, so the on-canvas preview node is the only thing that follows the cursor
const EMPTY_IMG = typeof Image !== "undefined" ? new Image() : null;
if (EMPTY_IMG) EMPTY_IMG.src =
  "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7";

function onDragStart(e: React.DragEvent, type: string) {
  dragging = type;
  e.dataTransfer.setData(PALETTE_MIME, type);
  e.dataTransfer.effectAllowed = "move";
  if (EMPTY_IMG?.complete) e.dataTransfer.setDragImage(EMPTY_IMG, 0, 0);
}

function onDragEnd() {
  dragging = null;
}

export function NodePalette() {
  return (
    <>
      <div className="reg-head">
        <h3 className="side-eyebrow">Node Registry</h3>
      </div>
      {ORDER.map((c) => {
        const types = CATALOG_ORDER.filter((t) => CATEGORY_OF[t] === c);
        if (types.length === 0) return null;
        const style = { "--accent": `var(${CATEGORY[c].accent})` } as CSSProperties;
        return (
          <section className="grp" key={c} style={style}>
            <h3>
              {CATEGORY[c].full}
              <span className="tag">{types.length}</span>
            </h3>
            <div className="pal">
              {types.map((t) => {
                const def = NODE_CATALOG[t];
                return (
                  <div
                    key={t}
                    className="pchip"
                    draggable
                    onDragStart={(e) => onDragStart(e, t)}
                    onDragEnd={onDragEnd}
                    title={resolveSubtitle(def, DEFAULT_META)}
                  >
                    {def.label}
                  </div>
                );
              })}
            </div>
          </section>
        );
      })}
    </>
  );
}
