// NodePalette: drag any node type onto the canvas, grouped by category.
import type { CSSProperties } from "react";
import {
  CATALOG_ORDER,
  CATEGORY,
  CATEGORY_OF,
  NODE_CATALOG,
  type NodeCategory,
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
      {ORDER.map((c) => {
        const types = CATALOG_ORDER.filter((t) => CATEGORY_OF[t] === c);
        if (types.length === 0) return null;
        const style: CSSProperties = {};
        (style as Record<string, string>)["--accent"] = `var(${CATEGORY[c].accent})`;
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
                    title={def.subtitle}
                  >
                    <span className="m" style={{ color: "var(--accent)" }}>
                      ▪
                    </span>
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
