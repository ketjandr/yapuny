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

function onDragStart(e: React.DragEvent, type: string) {
  e.dataTransfer.setData(PALETTE_MIME, type);
  e.dataTransfer.effectAllowed = "move";
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
