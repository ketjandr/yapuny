// Debounced live graph validation; shows "offline" if the worker is unreachable.
import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import type { YNodeData } from "@/lib/graph";
import { humanizeMessage, structuralIds } from "@/lib/structuralId";
import { useCanvasStore } from "@/store/canvasStore";

interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

type State =
  | { kind: "idle" }
  | { kind: "checking" }
  | { kind: "ok"; result: ValidationResult }
  | { kind: "offline" };

const DEBOUNCE_MS = 400;

export function ValidationOverlay() {
  const nodes = useCanvasStore((s) => s.nodes);
  const edges = useCanvasStore((s) => s.edges);
  const meta = useCanvasStore((s) => s.meta);
  const blockStart = useCanvasStore((s) => s.blockStart);
  const blockEnd = useCanvasStore((s) => s.blockEnd);
  const toGraph = useCanvasStore((s) => s.toGraph);

  const [state, setState] = useState<State>({ kind: "idle" });

  // signature of only validation-relevant graph content: node id/type/quant, edge endpoints +
  // type (incl. fusion), meta, block markers. Excludes positions/selection, so dragging nodes
  // around (or selecting) never re-validates - only a real structural change bumps it.
  const sig = useMemo(
    () =>
      JSON.stringify({
        n: nodes.map((n) => [n.id, (n.data as YNodeData).type, (n.data as YNodeData).quantized]),
        e: edges.map((e) => [e.source, e.sourceHandle, e.target, e.targetHandle, e.type]),
        meta,
        block: [blockStart, blockEnd],
      }),
    [nodes, edges, meta, blockStart, blockEnd],
  );

  useEffect(() => {
    let cancelled = false;
    // Show "validating" from the moment of the edit and hold it through the debounce AND the
    // in-flight request, until the response lands - a continuous pending state (like saving…).
    setState({ kind: "checking" });

    const t = setTimeout(async () => {
      try {
        const result = (await api.validate(toGraph())) as ValidationResult;
        if (!cancelled) setState({ kind: "ok", result });
      } catch {
        if (!cancelled) setState({ kind: "offline" });
      }
    }, DEBOUNCE_MS);

    return () => {
      cancelled = true;
      clearTimeout(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sig]);

  // structural (readable) id per node, so backend messages read in canvas terms
  const ids = useMemo(() => structuralIds(nodes, edges), [nodes, edges]);

  return (
    <div className="vpanel">
      {state.kind === "offline" && <div className="vrow vmuted">validation offline — worker unreachable</div>}
      {state.kind === "checking" && <div className="vrow vmuted">validating…</div>}
      {state.kind === "ok" && <ValidationBody result={state.result} ids={ids} />}
    </div>
  );
}

// humanize each message and drop duplicates (a block-internal error repeats per unrolled layer,
// all collapsing to the same canvas node), preserving order
function humanizeUnique(msgs: string[], ids: Map<string, string>): string[] {
  const out: string[] = [];
  for (const m of msgs) {
    const h = humanizeMessage(m, ids);
    if (!out.includes(h)) out.push(h);
  }
  return out;
}

// when there is no input -> output path, every other error (missing required nodes, unconnected
// inputs, ...) is downstream noise from the same root cause - surface just this one
const NO_PATH = "no complete path from graph input to output";

function ValidationBody({ result, ids }: { result: ValidationResult; ids: Map<string, string> }) {
  if (result.valid && result.warnings.length === 0) {
    return <div className="vrow vok">✓ model valid</div>;
  }
  if (result.errors.includes(NO_PATH)) {
    return <div className="vrow verr">● {NO_PATH}</div>;
  }
  return (
    <>
      {humanizeUnique(result.errors, ids).map((e, i) => (
        <div key={`e${i}`} className="vrow verr">
          ● {e}
        </div>
      ))}
      {humanizeUnique(result.warnings, ids).map((w, i) => (
        <div key={`w${i}`} className="vrow vwarn">
          ▲ {w}
        </div>
      ))}
    </>
  );
}
