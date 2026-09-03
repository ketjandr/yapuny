// Node card: catalog-driven geometry, inputs on the left edge, outputs on the right.
import type { CSSProperties } from "react";
import { Handle, type NodeProps, Position } from "@xyflow/react";
import {
  CATEGORY,
  CATEGORY_OF,
  formatShape,
  type NodeVariant,
  nodeHeight,
  nodeWidth,
  portTop,
  resolveNodeDef,
  resolveSubtitle,
} from "@/lib/nodeCatalog";
import type { YNodeData } from "@/lib/graph";
import { FUSE_PORT, fusionVisible } from "@/lib/fusion";
import { useCanvasStore } from "@/store/canvasStore";

const VARIANT_CLASS: Record<NodeVariant, string> = {
  req: "",
  io: "io",
};

export function GraphNode({ id, data, selected }: NodeProps) {
  const { type, quantized } = data as YNodeData;
  const meta = useCanvasStore((s) => s.meta);
  const mode = useCanvasStore((s) => s.mode);
  const blockStart = useCanvasStore((s) => s.blockStart);
  const blockEnd = useCanvasStore((s) => s.blockEnd);
  const fused = useCanvasStore((s) =>
    s.edges.some((e) => e.type === "fusion" && (e.source === id || e.target === id)),
  );
  const def = resolveNodeDef(type);
  if (!def) {
    return <div className="yn" style={{ padding: 8 }}>?{type}</div>;
  }

  // block boundary marker for this node (a node can be both in a single-node block)
  const blockRole =
    id === blockStart && id === blockEnd
      ? "block"
      : id === blockStart
        ? "start"
        : id === blockEnd
          ? "end"
          : null;

  const w = nodeWidth(def, meta);
  const h = nodeHeight(def);
  const dimmed = mode === "train" && !!def.trainingNoop; // inactive while training
  const cat = CATEGORY_OF[type] ? CATEGORY[CATEGORY_OF[type]] : null; // family color + badge
  const cls = [
    "yn",
    VARIANT_CLASS[def.variant],
    selected ? "sel" : "",
    quantized ? "quantized" : "",
    dimmed ? "dimmed" : "",
    // fused aura is inference-only for now; this is a class on the inner div only, so it never
    // affects the RF node/handle model (the fuse-port Handles below always render).
    fused && fusionVisible(mode) ? "fused" : "",
  ]
    .filter(Boolean)
    .join(" ");

  const style: CSSProperties = { width: w, height: h };
  if (cat) (style as Record<string, string>)["--accent"] = `var(${cat.accent})`;

  return (
    <div className={cls} style={style}>
      <div className="st" />
      {(blockRole === "start" || blockRole === "block") && (
        <span className="yn-bracket start" title="Block start" />
      )}
      {(blockRole === "end" || blockRole === "block") && (
        <span className="yn-bracket end" title="Block end" />
      )}
      {cat && <span className="yn-badge">{cat.label}</span>}

      <div className="yn-head">
        <span>{def.label}</span>
        {quantized && <span className="qtag">{quantized.toUpperCase()}</span>}
      </div>
      <div className="yn-sub">{resolveSubtitle(def, meta)}</div>

      {def.inputs.map((port, i) => (
        <Handle
          key={`in-${port.id}`}
          id={port.id}
          type="target"
          position={Position.Left}
          style={{ top: portTop(i) }}
        />
      ))}
      {def.inputs.map((port, i) => (
        <span
          key={`inl-${port.id}`}
          className="yn-plabel in"
          style={{ top: portTop(i) }}
        >
          <span className="role">{port.label}</span>
          <span className="shp">{formatShape(port.shape, meta, mode)}</span>
        </span>
      ))}

      {def.outputs.map((port, i) => (
        <Handle
          key={`out-${port.id}`}
          id={port.id}
          type="source"
          position={Position.Right}
          style={{ top: portTop(i) }}
        />
      ))}
      {def.outputs.map((port, i) => (
        <span
          key={`outl-${port.id}`}
          className="yn-plabel out"
          style={{ top: portTop(i) }}
        >
          <span className="role">{port.label}</span>
          <span className="shp">{formatShape(port.shape, meta, mode)}</span>
        </span>
      ))}

      {/* fusion port: the bottom diamond. Overlapping target + source handles (source on top,
          so a drag starts from it) let a stream be dragged between any two fusable nodes. */}
      {def.fusable && (
        <>
          <Handle id={FUSE_PORT} type="target" position={Position.Bottom} className={`yn-fuse${fused ? " on" : ""}`} />
          <Handle id={FUSE_PORT} type="source" position={Position.Bottom} className={`yn-fuse${fused ? " on" : ""}`} />
        </>
      )}
    </div>
  );
}
