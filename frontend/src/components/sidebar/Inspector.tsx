// Inspector: contextual properties for the current selection. Single node -> node properties,
// single edge -> edge properties, multi/empty -> a greyed placeholder. Read-only for now.
import { type CSSProperties, type ReactNode, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import type { Edge, Node } from "@xyflow/react";
import { api } from "@/lib/api";
import { deriveFusionGroups, type FusionCatalog, matchFusionKernel } from "@/lib/fusion";
import type { YNodeData } from "@/lib/graph";
import { CATEGORY, CATEGORY_OF, formatShape, resolveNodeDef } from "@/lib/nodeCatalog";
import { structuralIds } from "@/lib/structuralId";
import { useCanvasStore } from "@/store/canvasStore";
import type { GraphMetaSchema } from "@/lib/types";
import type { CanvasMode } from "@/store/canvasStore";

function Row({ label, children, mono }: { label: string; children: ReactNode; mono?: boolean }) {
  return (
    <div className="insp-row">
      <span className="insp-k">{label}</span>
      <span className={`insp-v${mono ? " mono" : ""}`}>{children}</span>
    </div>
  );
}

const EMPTY_FIELDS = ["ID", "Type", "Family", "Block", "Fusion", "Quant"];

type Friendly = (id: string) => string;

export function Inspector() {
  const nodes = useCanvasStore((s) => s.nodes);
  const edges = useCanvasStore((s) => s.edges);
  const meta = useCanvasStore((s) => s.meta);
  const mode = useCanvasStore((s) => s.mode);
  const blockStart = useCanvasStore((s) => s.blockStart);
  const blockEnd = useCanvasStore((s) => s.blockEnd);

  const ids = useMemo(() => structuralIds(nodes, edges), [nodes, edges]);
  const typeOf = useMemo(
    () => new Map(nodes.map((n) => [n.id, (n.data as YNodeData).type])),
    [nodes],
  );
  // structural id (dropout_0), pseudo-nodes read as input/output, else the raw id
  const friendly: Friendly = (id) => {
    const s = ids.get(id);
    if (s) return s;
    const t = typeOf.get(id);
    return t === "_input" ? "input" : t === "_output" ? "output" : id;
  };

  const selNodes = nodes.filter((n) => n.selected);
  const selEdges = edges.filter((e) => e.selected);

  if (selEdges.length === 1 && selNodes.length === 0) {
    return <EdgeProps edge={selEdges[0]} friendly={friendly} typeOf={typeOf} meta={meta} mode={mode} />;
  }
  if (selNodes.length === 1 && selEdges.length === 0) {
    return (
      <NodeProps
        node={selNodes[0]}
        edges={edges}
        ids={ids}
        typeOf={typeOf}
        friendly={friendly}
        blockStart={blockStart}
        blockEnd={blockEnd}
      />
    );
  }

  // multi-select or nothing selected: greyed placeholder with nulled fields
  const count = selNodes.length + selEdges.length;
  return (
    <section className="insp-sec">
      <h3 className="side-eyebrow">Node Properties</h3>
      <div className="insp disabled">
        {EMPTY_FIELDS.map((f) => (
          <Row key={f} label={f}>
            —
          </Row>
        ))}
        <p className="insp-note">{count > 1 ? `${count} items selected` : "Select a node or edge"}</p>
      </div>
    </section>
  );
}

function NodeProps({
  node,
  edges,
  ids,
  typeOf,
  friendly,
  blockStart,
  blockEnd,
}: {
  node: Node;
  edges: Edge[];
  ids: Map<string, string>;
  typeOf: Map<string, string>;
  friendly: Friendly;
  blockStart: string | null;
  blockEnd: string | null;
}) {
  const d = node.data as YNodeData;
  const def = resolveNodeDef(d.type);
  const category = CATEGORY_OF[d.type];
  const cat = category ? CATEGORY[category] : null;

  const { data: catalog } = useQuery<FusionCatalog>({
    queryKey: ["fusion-available"],
    queryFn: api.fusionAvailable,
    staleTime: Infinity,
    retry: false,
  });

  const group = useMemo(
    () => deriveFusionGroups(edges).find((g) => g.includes(node.id)),
    [edges, node.id],
  );
  // fusion is shown by kernel + this node's slot (constant length), not by listing every member
  const order = useMemo(() => new Map([...ids.keys()].map((id, i) => [id, i])), [ids]);
  const fusion = useMemo(() => {
    if (!group) return "none";
    const n = group.length;
    const dataEdges = edges.filter((e) => e.type !== "fusion");
    const kernel = matchFusionKernel(group, typeOf, dataEdges, catalog);
    if (kernel) {
      const chain = [...group].sort((a, b) => (order.get(a) ?? 0) - (order.get(b) ?? 0));
      return `${kernel} · ${chain.indexOf(node.id) + 1}/${n}`;
    }
    return catalog?.available ? `${n} nodes · invalid` : `${n} nodes`;
  }, [group, edges, typeOf, catalog, order, node.id]);

  const blockRole =
    node.id === blockStart && node.id === blockEnd
      ? "start + end"
      : node.id === blockStart
        ? "start"
        : node.id === blockEnd
          ? "end"
          : "—";

  const style = cat ? ({ "--accent": `var(${cat.accent})` } as CSSProperties) : undefined;

  return (
    <section className="insp-sec">
      <h3 className="side-eyebrow">Node Properties</h3>
      <div className="insp" style={style}>
        <Row label="ID" mono>
          {friendly(node.id)}
        </Row>
        <Row label="Type">{def?.label ?? d.type}</Row>
        {cat && (
          <Row label="Family">
            <span className="insp-fam">{cat.full}</span>
          </Row>
        )}
        <Row label="Block">{blockRole}</Row>
        <Row label="Fusion" mono>
          {fusion}
        </Row>
        <Row label="Quant" mono>
          {d.quantized ? d.quantized.toUpperCase() : "none"}
        </Row>
      </div>
    </section>
  );
}

function EdgeProps({
  edge,
  friendly,
  typeOf,
  meta,
  mode,
}: {
  edge: Edge;
  friendly: Friendly;
  typeOf: Map<string, string>;
  meta: GraphMetaSchema;
  mode: CanvasMode;
}) {
  const fusion = edge.type === "fusion";
  const srcDef = resolveNodeDef(typeOf.get(edge.source) ?? "");
  const tgtDef = resolveNodeDef(typeOf.get(edge.target) ?? "");
  const srcPort = srcDef?.outputs.find((p) => p.id === (edge.sourceHandle ?? "out"));
  const tgtPort = tgtDef?.inputs.find((p) => p.id === (edge.targetHandle ?? "x"));
  // show the port's tensor-role label (what the canvas shows), not the raw backend handle id
  const fromPort = srcPort?.label ?? edge.sourceHandle ?? "out";
  const toPort = tgtPort?.label ?? edge.targetHandle ?? "x";
  const shape = srcPort ? formatShape(srcPort.shape, meta, mode) : null;

  return (
    <section className="insp-sec">
      <h3 className="side-eyebrow">Edge Properties</h3>
      <div className="insp">
        <Row label="Kind">{fusion ? "Fusion stream" : "Tensor edge"}</Row>
        <Row label="From" mono>
          {friendly(edge.source)}
          {fusion ? "" : ` · ${fromPort}`}
        </Row>
        <Row label="To" mono>
          {friendly(edge.target)}
          {fusion ? "" : ` · ${toPort}`}
        </Row>
        {!fusion && shape && (
          <Row label="Shape" mono>
            {shape}
          </Row>
        )}
      </div>
    </section>
  );
}
