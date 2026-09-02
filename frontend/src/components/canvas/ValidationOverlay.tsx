// Debounced live graph validation; shows "offline" if the worker is unreachable.
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
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
  const toGraph = useCanvasStore((s) => s.toGraph);

  const [state, setState] = useState<State>({ kind: "idle" });

  useEffect(() => {
    let cancelled = false;
    setState((prev) => (prev.kind === "ok" ? prev : { kind: "checking" }));

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
  }, [nodes, edges, meta]);

  return (
    <div className="vpanel">
      {state.kind === "offline" && <div className="vrow vmuted">validation offline — worker unreachable</div>}
      {state.kind === "checking" && <div className="vrow vmuted">validating…</div>}
      {state.kind === "ok" && <ValidationBody result={state.result} />}
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
