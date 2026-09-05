// Debounced live graph validation; shows "offline" if the worker is unreachable. Result lives in
// validationStore so CompileBar can gate the Compile button on validity + in-flight checks.
import { useEffect } from "react";
import { api } from "@/lib/api";
import { useCanvasStore } from "@/store/canvasStore";
import { type ValidationResult, useValidationStore } from "@/store/validationStore";

const DEBOUNCE_MS = 400;

export function ValidationOverlay() {
  const nodes = useCanvasStore((s) => s.nodes);
  const edges = useCanvasStore((s) => s.edges);
  const meta = useCanvasStore((s) => s.meta);
  const toGraph = useCanvasStore((s) => s.toGraph);

  const view = useValidationStore((s) => s.view);
  const setView = useValidationStore((s) => s.setView);
  const setValidating = useValidationStore((s) => s.setValidating);

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
  }, [nodes, edges, meta]);

  return (
    <div className="vpanel">
      {view.kind === "offline" && <div className="vrow vmuted">validation offline — worker unreachable</div>}
      {view.kind === "checking" && <div className="vrow vmuted">validating…</div>}
      {view.kind === "ok" && <ValidationBody result={view.result} />}
    </div>
  );
}

function ValidationBody({ result }: { result: ValidationResult }) {
  if (result.valid && result.warnings.length === 0) {
    return <div className="vrow vok">✓ graph valid</div>;
  }
  return (
    <>
      {result.errors.map((e, i) => (
        <div key={`e${i}`} className="vrow verr">
          ● {e}
        </div>
      ))}
      {result.warnings.map((w, i) => (
        <div key={`w${i}`} className="vrow vwarn">
          ▲ {w}
        </div>
      ))}
    </>
  );
}
