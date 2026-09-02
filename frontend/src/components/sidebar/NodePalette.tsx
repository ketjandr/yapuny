// NodePalette: drag any node type onto the canvas (sampling params aren't nodes).
import { CATALOG_ORDER, NODE_CATALOG, type NodeVariant } from "@/lib/nodeCatalog";

export const PALETTE_MIME = "application/yapuny-node";

const GROUPS: { title: string; variants: NodeVariant[] }[] = [
  { title: "Core", variants: ["req"] },
  { title: "Optional", variants: ["opt", "flash"] },
];

function onDragStart(e: React.DragEvent, type: string) {
  e.dataTransfer.setData(PALETTE_MIME, type);
  e.dataTransfer.effectAllowed = "move";
}

export function NodePalette() {
  return (
    <>
      {GROUPS.map((group) => {
        const types = CATALOG_ORDER.filter((t) => group.variants.includes(NODE_CATALOG[t].variant));
        if (types.length === 0) return null;
        return (
          <section className="grp" key={group.title}>
            <h3>
              {group.title}
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
                    <span className="m">{def.variant === "flash" ? "⚡" : "▪"}</span>
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
