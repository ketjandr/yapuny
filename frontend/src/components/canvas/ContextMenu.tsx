// Right-click menu for the canvas: node (copy/delete + quant/block action row), edge (delete),
// pane (paste).
import { useReactFlow } from "@xyflow/react";
import { BracketIcon, QuantIcon } from "@/components/nodeActionIcons";
import { useTooltip } from "@/components/tooltipContext";
import type { YNodeData } from "@/lib/graph";
import { resolveNodeDef } from "@/lib/nodeCatalog";
import { nextQuant } from "@/lib/quant";
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
        {menu.kind === "node" && menu.id && (
          <NodeActions
            nodeId={menu.id}
            multi={targets(menu.id, nodes.filter((n) => n.selected).map((n) => n.id)).length > 1}
          />
        )}
      </div>
    </>
  );
}

// quant + block-start/block-end toggles for a node, laid out as a 3-part icon row. Toggling leaves
// the menu open (so several can be set at once); a block boundary is single-node, disabled on multi.
function NodeActions({ nodeId, multi }: { nodeId: string; multi: boolean }) {
  const nodes = useCanvasStore((s) => s.nodes);
  const mode = useCanvasStore((s) => s.mode);
  const blockStart = useCanvasStore((s) => s.blockStart);
  const blockEnd = useCanvasStore((s) => s.blockEnd);
  const setQuantized = useCanvasStore((s) => s.setQuantized);
  const setBlockStart = useCanvasStore((s) => s.setBlockStart);
  const setBlockEnd = useCanvasStore((s) => s.setBlockEnd);

  const d = nodes.find((n) => n.id === nodeId)?.data as YNodeData | undefined;
  const def = resolveNodeDef(d?.type ?? "");
  const quantizable = (def?.quantizable ?? false) && mode === "inference";
  const isStart = nodeId === blockStart;
  const isEnd = nodeId === blockEnd;

  const quantTip = useTooltip("Quantize weights (W8 / W4)");
  const startTip = useTooltip(isStart ? "Clear block start" : "Set as block start");
  const endTip = useTooltip(isEnd ? "Clear block end" : "Set as block end");

  return (
    <div className="ctx-icons">
      {quantizable && (
        <button
          type="button"
          className={`ctx-icon${d?.quantized ? " on" : ""}`}
          onClick={() => setQuantized(nodeId, nextQuant(d?.quantized ?? null))}
          aria-label="Quantize weights"
          {...quantTip}
        >
          {d?.quantized ? <span className="ctx-icon-mode">{d.quantized.toUpperCase()}</span> : <QuantIcon />}
        </button>
      )}
      <button
        type="button"
        className={`ctx-icon${isStart ? " on" : ""}`}
        disabled={multi}
        onClick={() => setBlockStart(isStart ? null : nodeId)}
        aria-label="Set as block start"
        {...startTip}
      >
        <BracketIcon side="start" />
      </button>
      <button
        type="button"
        className={`ctx-icon${isEnd ? " on" : ""}`}
        disabled={multi}
        onClick={() => setBlockEnd(isEnd ? null : nodeId)}
        aria-label="Set as block end"
        {...endTip}
      >
        <BracketIcon side="end" />
      </button>
    </div>
  );
}
