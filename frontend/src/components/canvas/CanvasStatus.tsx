// Canvas status overlays:
//  - SaveBar (bottom-left, by the validator): autosave "Saving…/Saved" chip + revert-to-compiled.
//  - CompileBar (bottom-center): Compile button + "needs compile/compiled" and "needs train/
//    trained" indicators, each with a glowing dot (amber = needs, green = good).
import { useCanvasStore } from "@/store/canvasStore";

export function SaveBar() {
  const saveStatus = useCanvasStore((s) => s.saveStatus);
  const canRevert = useCanvasStore((s) => s.lastCompiled != null && s.needsCompile);
  const revertToCompiled = useCanvasStore((s) => s.revertToCompiled);
  const saving = saveStatus === "saving";

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
  const needsCompile = useCanvasStore((s) => s.needsCompile);
  const trained = useCanvasStore((s) => s.trained);
  const markCompiled = useCanvasStore((s) => s.markCompiled);
  // "trained" only reads green once the graph is both compiled and trained
  const trainedOk = !needsCompile && trained;

  return (
    <div className="compilebar">
      <button
        type="button"
        className="compile-btn"
        disabled={!needsCompile}
        onClick={markCompiled}
        title={needsCompile ? "Compile the current graph" : "Graph is already compiled"}
      >
        Compile
      </button>
      <Indicator ok={!needsCompile} okLabel="compiled" needLabel="needs compile" />
      <Indicator ok={trainedOk} okLabel="trained" needLabel="needs train" />
    </div>
  );
}

function Indicator({ ok, okLabel, needLabel }: { ok: boolean; okLabel: string; needLabel: string }) {
  return (
    <div className="ci">
      <span className={`ci-dot ${ok ? "ok" : "warn"}`} />
      {ok ? okLabel : needLabel}
    </div>
  );
}
