// Canvas status overlays, driven by the worker (source of truth):
//  - SaveBar (bottom-left, by the validator): autosave "Saving…/Saved" chip + revert-to-compiled.
//  - CompileBar (bottom-center): Compile button (calls the compile API, shows "compiling…") +
//    "needs compile/compiled" and "needs train/trained" indicators with glowing dots.
// model_status is polled (debounced) on every graph change so the indicators reflect the worker's
// in-memory model cache; compile / status also report `trained` when weights load from the locker.
import { useEffect } from "react";
import { api } from "@/lib/api";
import { useCanvasStore } from "@/store/canvasStore";
import { useCompileStore } from "@/store/compileStore";
import { useValidationStore } from "@/store/validationStore";
import { toast } from "@/store/toastStore";

const STATUS_DEBOUNCE_MS = 400;

interface StatusResponse {
  status: "ready" | "needs_compile";
  trained?: boolean;
}
interface CompileResponse {
  status: string;
  weights?: string;
  trained?: boolean;
}

// Poll the worker for whether the current graph is compiled (in its cache) and trained.
function useModelStatusPoll() {
  const nodes = useCanvasStore((s) => s.nodes);
  const edges = useCanvasStore((s) => s.edges);
  const meta = useCanvasStore((s) => s.meta);
  const blockStart = useCanvasStore((s) => s.blockStart);
  const blockEnd = useCanvasStore((s) => s.blockEnd);
  const modelId = useCanvasStore((s) => s.modelId);
  const toGraph = useCanvasStore((s) => s.toGraph);
  const compiling = useCompileStore((s) => s.compiling);
  const setResult = useCompileStore((s) => s.setResult);
  const setChecking = useCompileStore((s) => s.setChecking);
  const setPolling = useCompileStore((s) => s.setPolling);

  useEffect(() => {
    if (compiling) return; // don't poll mid-compile; the effect re-runs when compiling clears
    let cancelled = false;
    setChecking();
    setPolling(true); // a check is pending (through the debounce + request) -> Compile disabled
    const t = setTimeout(async () => {
      try {
        const res = (await api.modelStatus({ id: modelId, graph: toGraph() })) as StatusResponse;
        if (cancelled) return;
        const ready = res.status === "ready";
        setResult(ready ? "ready" : "needs_compile", ready && !!res.trained);
      } catch {
        if (!cancelled) setResult("offline", false);
      } finally {
        if (!cancelled) setPolling(false); // superseded checks leave it to the newest effect
      }
    }, STATUS_DEBOUNCE_MS);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes, edges, meta, blockStart, blockEnd, modelId, compiling]);
}

export function SaveBar() {
  const saveStatus = useCanvasStore((s) => s.saveStatus);
  const hasCompiled = useCanvasStore((s) => s.lastCompiled != null);
  const revertToCompiled = useCanvasStore((s) => s.revertToCompiled);
  const status = useCompileStore((s) => s.status);
  const saving = saveStatus === "saving";
  // something to revert to, and the worker says the current graph isn't the compiled one
  const canRevert = hasCompiled && status === "needs_compile";

  return (
    <div className="savebar">
      {/* mirrors the validation rows above it: same .vrow styling + check mark */}
      <div className={`vrow ${saving ? "vmuted" : "vok"}`}>{saving ? "saving…" : "✓ saved"}</div>
      <button
        type="button"
        className="vrow revert-btn"
        disabled={!canRevert}
        onClick={revertToCompiled}
        title="Revert the canvas to the last compiled graph"
      >
        ⟲ revert to compiled
      </button>
    </div>
  );
}

export function CompileBar() {
  useModelStatusPoll();
  const status = useCompileStore((s) => s.status);
  const trained = useCompileStore((s) => s.trained);
  const compiling = useCompileStore((s) => s.compiling);
  const polling = useCompileStore((s) => s.polling);
  const setResult = useCompileStore((s) => s.setResult);
  const setCompiling = useCompileStore((s) => s.setCompiling);
  const validating = useValidationStore((s) => s.validating);
  const invalid = useValidationStore((s) => s.view.kind === "ok" && !s.view.result.valid);
  const modelId = useCanvasStore((s) => s.modelId);
  const toGraph = useCanvasStore((s) => s.toGraph);
  const markCompiled = useCanvasStore((s) => s.markCompiled);

  const compiled = status === "ready";
  const trainedOk = compiled && trained;
  // can't compile while a check is in flight (validation or model/status), if the graph is invalid,
  // if already compiled, or mid-compile
  const disabled = compiling || polling || validating || invalid || compiled;
  const title = invalid
    ? "Fix the graph errors before compiling"
    : compiled
      ? "Graph is already compiled"
      : "Compile the current graph";

  const onCompile = async () => {
    if (compiling) return;
    setCompiling(true);
    try {
      const res = (await api.compile({ id: modelId, graph: toGraph() })) as CompileResponse;
      setResult("ready", !!res.trained);
      markCompiled(); // snapshot the graph so "revert to compiled" has a target
    } catch (e) {
      toast.error(`Compile failed: ${(e as Error).message}`);
    } finally {
      setCompiling(false);
    }
  };

  return (
    <div className="compilebar">
      <button
        type="button"
        className={`compile-btn${compiling ? " compiling" : ""}`}
        disabled={disabled}
        onClick={onCompile}
        title={title}
      >
        {compiling ? "compiling…" : "Compile"}
      </button>
      <Indicator ok={compiled} okLabel="model compiled" needLabel="needs compile" />
      <Indicator ok={trainedOk} okLabel="model trained" needLabel="needs train" />
    </div>
  );
}

function Indicator({ ok, okLabel, needLabel }: { ok: boolean; okLabel: string; needLabel: string }) {
  // reserve the widest label's width (monospace => 1ch per char) so the wrapper width and the
  // Compile button position stay fixed when the text toggles (e.g. "compiled" <-> "needs compile")
  const reserve = Math.max(okLabel.length, needLabel.length);
  return (
    <div className="ci">
      <span className={`ci-dot ${ok ? "ok" : "warn"}`} />
      <span className="ci-label" style={{ minWidth: `${reserve}ch` }}>
        {ok ? okLabel : needLabel}
      </span>
    </div>
  );
}
