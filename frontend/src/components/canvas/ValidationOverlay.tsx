// Debounced live graph validation; shows "offline" if the worker is unreachable. Result lives in
// validationStore so CompileBar can gate the Compile button on validity + in-flight checks.
import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import type { YNodeData } from "@/lib/graph";
import { humanizeMessage, structuralIds } from "@/lib/structuralId";
import { useCanvasStore } from "@/store/canvasStore";
import { type ValidationResult, useValidationStore } from "@/store/validationStore";

const DEBOUNCE_MS = 400;

// when there is no input -> output path, every other error (missing required nodes, unconnected
// inputs, ...) is downstream noise from the same root cause - surface just this one
const NO_PATH = "no complete path from graph input to output";

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

export function ValidationOverlay() {
  const nodes = useCanvasStore((s) => s.nodes);
  const edges = useCanvasStore((s) => s.edges);
  const meta = useCanvasStore((s) => s.meta);
  const toGraph = useCanvasStore((s) => s.toGraph);

  const blockStart = useCanvasStore((s) => s.blockStart);
  const blockEnd = useCanvasStore((s) => s.blockEnd);

  const view = useValidationStore((s) => s.view);
  const setView = useValidationStore((s) => s.setView);
  const setValidating = useValidationStore((s) => s.setValidating);

  // collapse state kept in the parent so it survives ValidationBody unmount/remount across checks
  const [open, setOpen] = useState(true);

  // signature of only validation-relevant graph content: node id/type/quant, edge endpoints + type
  // (incl. fusion), meta, block markers. Excludes positions/selection, so dragging a node around
  // (or selecting one) never re-validates - only a real structural change bumps it.
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
    // keep the last "ok" result visible while re-checking (no message flicker); the Compile button
    // uses the separate `validating` flag, which stays true through the debounce + request
    if (useValidationStore.getState().view.kind !== "ok") setView({ kind: "checking" });
    setValidating(true);

    const t = setTimeout(async () => {
      try {
        const result = (await api.validate(toGraph())) as ValidationResult;
        if (!cancelled) setView({ kind: "ok", result });
      } catch {
        if (!cancelled) setView({ kind: "offline" });
      } finally {
        if (!cancelled) setValidating(false); // superseded checks leave it to the newest effect
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
      {view.kind === "offline" && <div className="vrow vmuted">validation offline — worker unreachable</div>}
      {view.kind === "checking" && <div className="vrow vmuted">validating…</div>}
      {view.kind === "ok" && (
        <ValidationBody result={view.result} ids={ids} open={open} onToggle={() => setOpen((o) => !o)} />
      )}
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

function ValidationBody({
  result,
  ids,
  open,
  onToggle,
}: {
  result: ValidationResult;
  ids: Map<string, string>;
  open: boolean;
  onToggle: () => void;
}) {
  const noPath = result.errors.includes(NO_PATH);
  const errors = noPath ? [NO_PATH] : humanizeUnique(result.errors, ids);
  const warnings = noPath ? [] : humanizeUnique(result.warnings, ids);

  if (errors.length === 0 && warnings.length === 0) {
    return <div className="vrow vok">✓ graph valid</div>;
  }

  // fixed-size summary header (counts never grow); the detail list scrolls within a capped height,
  // so any number of errors/warnings stays viewable without the panel growing off-screen
  return (
    <>
      <button type="button" className="vhead" onClick={onToggle} aria-expanded={open}>
        {errors.length > 0 && <span className="vcount verr">● {plural(errors.length, "error")}</span>}
        {warnings.length > 0 && (
          <span className="vcount vwarn">▲ {plural(warnings.length, "warning")}</span>
        )}
        <span className={`vchev${open ? " o" : ""}`} aria-hidden>
          ▸
        </span>
      </button>
      {open && (
        <div className="vlist">
          {errors.map((e, i) => (
            <div key={`e${i}`} className="vrow verr">
              ● {e}
            </div>
          ))}
          {warnings.map((w, i) => (
            <div key={`w${i}`} className="vrow vwarn">
              ▲ {w}
            </div>
          ))}
        </div>
      )}
    </>
  );
}
