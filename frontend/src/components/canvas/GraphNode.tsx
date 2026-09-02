// Node card: catalog-driven geometry, inputs on the left edge, outputs on the right.
import { Handle, type NodeProps, Position } from "@xyflow/react";
import {
  formatShape,
  type NodeVariant,
  nodeHeight,
  nodeWidth,
  portTop,
  resolveNodeDef,
} from "@/lib/nodeCatalog";
import type { YNodeData } from "@/lib/graph";
import { useCanvasStore } from "@/store/canvasStore";

const VARIANT_CLASS: Record<NodeVariant, string> = {
  req: "",
  opt: "opt",
  flash: "flash",
  io: "io",
};

export function GraphNode({ data, selected }: NodeProps) {
  const { type, quantized } = data as YNodeData;
  const meta = useCanvasStore((s) => s.meta);
  const mode = useCanvasStore((s) => s.mode);
  const def = resolveNodeDef(type);
  if (!def) {
    return <div className="yn" style={{ padding: 8 }}>?{type}</div>;
  }

  const w = nodeWidth(def);
  const h = nodeHeight(def);
  const dimmed = mode === "train" && !!def.trainingNoop; // inactive while training
  const cls = [
    "yn",
    VARIANT_CLASS[def.variant],
    selected ? "sel" : "",
    quantized ? "quantized" : "",
    dimmed ? "dimmed" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={cls} style={{ width: w, height: h }}>
      <div className="st" />
      {def.badge && <span className="yn-badge">{def.badge}</span>}

      <div className="yn-head">
        <span>{def.label}</span>
        {quantized && <span className="qtag">{quantized.toUpperCase()}</span>}
      </div>
      <div className="yn-sub">{def.subtitle}</div>

      {def.inputs.map((port, i) => (
        <Handle
          key={`in-${port.id}`}
          id={port.id}
          type="target"
          position={Position.Left}
          style={{ top: portTop(def, i, def.inputs.length) }}
        />
      ))}
      {def.inputs.map((port, i) => (
        <span
          key={`inl-${port.id}`}
          className="yn-plabel in"
          style={{ top: portTop(def, i, def.inputs.length) }}
        >
          <span className="role">{port.label}</span>
          <span className="shp">{formatShape(port.shape, meta)}</span>
        </span>
      ))}

      {def.outputs.map((port, i) => (
        <Handle
          key={`out-${port.id}`}
          id={port.id}
          type="source"
          position={Position.Right}
          style={{ top: portTop(def, i, def.outputs.length) }}
        />
      ))}
      {def.outputs.map((port, i) => (
        <span
          key={`outl-${port.id}`}
          className="yn-plabel out"
          style={{ top: portTop(def, i, def.outputs.length) }}
        >
          <span className="role">{port.label}</span>
          <span className="shp">{formatShape(port.shape, meta)}</span>
        </span>
      ))}

      {def.fusable && <span className="yn-fport" />}
    </div>
  );
}
