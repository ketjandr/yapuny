// Right-click menu for the canvas: node (copy/delete/block markers), edge (delete), pane (paste).
import { useReactFlow } from "@xyflow/react";
import { useCanvasStore } from "@/store/canvasStore";

export interface CanvasMenu {
  x: number;
  y: number;
  kind: "node" | "edge" | "pane";
  id?: string;
}

interface Item {
  label: string;
  onClick: () => void;
  disabled?: boolean;
}

export function ContextMenu({ menu, onClose }: { menu: CanvasMenu; onClose: () => void }) {
  const { screenToFlowPosition } = useReactFlow();
  const nodes = useCanvasStore((s) => s.nodes);
  const edges = useCanvasStore((s) => s.edges);
  const clipboard = useCanvasStore((s) => s.clipboard);
  const copyNodes = useCanvasStore((s) => s.copyNodes);
  const removeNodes = useCanvasStore((s) => s.removeNodes);
  const removeEdges = useCanvasStore((s) => s.removeEdges);
  const paste = useCanvasStore((s) => s.paste);
  const setBlockStart = useCanvasStore((s) => s.setBlockStart);
  const setBlockEnd = useCanvasStore((s) => s.setBlockEnd);

  const run = (fn: () => void) => {
    fn();
    onClose();
  };

  // act on the whole selection when the right-clicked element is part of it, else just it
  const targets = (id: string, selected: string[]) => (selected.includes(id) ? selected : [id]);

  let items: Item[] = [];
  if (menu.kind === "node" && menu.id) {
    const ids = targets(
      menu.id,
      nodes.filter((n) => n.selected).map((n) => n.id),
    );
    // edges come along only if the user also selected them (opt-in via selection)
    const selectedEdges = edges.filter((e) => e.selected).map((e) => e.id);
    const multi = ids.length > 1;
    const noun = multi ? `${ids.length} nodes` : "node";
    items = [
      { label: `Copy ${noun}`, onClick: () => run(() => copyNodes(ids, selectedEdges)) },
      { label: `Delete ${noun}`, onClick: () => run(() => removeNodes(ids)) },
      // a block boundary is a single node, so disable when multiple are selected
      { label: "Set as block start", disabled: multi, onClick: () => run(() => setBlockStart(menu.id!)) },
      { label: "Set as block end", disabled: multi, onClick: () => run(() => setBlockEnd(menu.id!)) },
    ];
  } else if (menu.kind === "edge" && menu.id) {
    const ids = targets(
      menu.id,
      edges.filter((e) => e.selected).map((e) => e.id),
    );
    items = [
      { label: ids.length > 1 ? `Delete ${ids.length} edges` : "Delete edge", onClick: () => run(() => removeEdges(ids)) },
    ];
  } else {
    const count = clipboard?.nodes.length ?? 0;
    items = [
      {
        label: count > 0 ? `Paste ${count > 1 ? `${count} nodes` : "node"}` : "Paste",
        disabled: count === 0,
        onClick: () => run(() => paste(screenToFlowPosition({ x: menu.x, y: menu.y }))),
      },
    ];
  }

  return (
    <>
      <div
        className="ctx-backdrop"
        onClick={onClose}
        onContextMenu={(e) => {
          e.preventDefault();
          onClose();
        }}
      />
      <div className="ctx-menu" style={{ left: menu.x, top: menu.y }}>
        {items.map((it) => (
          <button
            key={it.label}
            type="button"
            className="ctx-item"
            disabled={it.disabled}
            onClick={it.onClick}
          >
            {it.label}
          </button>
        ))}
      </div>
    </>
  );
}
