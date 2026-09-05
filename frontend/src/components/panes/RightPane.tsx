// Right pane: mode-aware controls. Train -> Config + Train (live loss curve, stats, hyperparams,
// stop); Inference -> Generate (placeholder). A Benchmark toggle reveals the benchmark section
// (per-node profiler for the active model + a head-to-head compare of trained models). Wired to the
// worker over SSE: training via lib/trainStore, benchmarks via lib/benchStore. Collapsible + full-
// view expandable via PaneShell; `expanded` lets sections widen their layout in full view.
import { useEffect, useMemo, useState } from "react";
import { useTooltip } from "@/components/tooltipContext";
import { api } from "@/lib/api";
import { graphForProject } from "@/lib/graph";
import { useBenchStore } from "@/store/benchStore";
import { type CanvasMode, N_LAYER_MAX, N_LAYER_MIN, useCanvasStore } from "@/store/canvasStore";
import { useCompileStore } from "@/store/compileStore";
import { useProjectsStore } from "@/store/projectsStore";
import { useTrainStore } from "@/store/trainStore";
import { PaneShell } from "./PaneShell";

export function RightPane() {
  const mode = useCanvasStore((s) => s.mode);
  const [expanded, setExpanded] = useState(false);
  // benchmark open-state is decoupled per mode: train and inference each remember their own
  const [bench, setBench] = useState<Record<CanvasMode, boolean>>({ train: false, inference: false });
  const title = mode === "train" ? "Training" : "Inference";

  return (
    <PaneShell side="right" title={title} expanded={expanded} onToggleExpand={() => setExpanded((v) => !v)}>
      <div className="pane-body">
        {mode === "train" ? (
          <>
            <ConfigSection expanded={expanded} />
            <TrainSection expanded={expanded} />
          </>
        ) : (
          <GenerateSection expanded={expanded} />
        )}
        <BenchmarkSection
          expanded={expanded}
          open={bench[mode]}
          onToggle={(v) => setBench((b) => ({ ...b, [mode]: v }))}
        />
      </div>
    </PaneShell>
  );
}

// --- config ---

function ConfigSection({ expanded }: { expanded: boolean }) {
  const meta = useCanvasStore((s) => s.meta);
  const setMeta = useCanvasStore((s) => s.setMeta);
  // n_embd snaps to a multiple of n_head (also enforced in the store); head_dim shown as a hint
  const snapEmbd = (n: number) => Math.max(meta.n_head, Math.round(n / meta.n_head) * meta.n_head);
  return (
    <section className="grp">
      <h3>Config</h3>
      {/* full view: lay the 6 sliders out in a 2-column, 3-row grid */}
      <div className={`cfg${expanded ? " grid2" : ""}`}>
        <CfgSlider label="n_layer" value={meta.n_layer} min={N_LAYER_MIN} max={N_LAYER_MAX} step={1} onChange={(v) => setMeta({ n_layer: v })} />
        <CfgSlider label="n_head" value={meta.n_head} min={1} max={16} step={1} onChange={(v) => setMeta({ n_head: v })} />
        <CfgSlider
          label="n_embd"
          hint={`head_dim ${Math.floor(meta.n_embd / meta.n_head)}`}
          value={meta.n_embd}
          min={meta.n_head}
          max={1024}
          step={meta.n_head}
          snap={snapEmbd}
          onChange={(v) => setMeta({ n_embd: v })}
        />
        <CfgSlider label="block_size" value={meta.block_size} min={16} max={1024} step={16} onChange={(v) => setMeta({ block_size: v })} />
        <CfgSlider label="dropout" value={meta.dropout} min={0} max={0.5} step={0.05} float onChange={(v) => setMeta({ dropout: v })} />
        <CfgSlider label="vocab_size" value={meta.vocab_size} min={256} max={50000} step={256} onChange={(v) => setMeta({ vocab_size: v })} />
      </div>
    </section>
  );
}

function CfgSlider({
  label,
  hint,
  value,
  min,
  max,
  step,
  float,
  snap,
  onChange,
}: {
  label: string;
  hint?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  float?: boolean;
  snap?: (n: number) => number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="cfg-slider">
      <div className="cfg-row">
        <span className="cfg-k">
          {label}
          {hint && <span className="cfg-hint">{hint}</span>}
        </span>
        <NumField value={value} min={min} max={max} float={float} snap={snap} onCommit={onChange} />
      </div>
      <input
        className="cfg-range"
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

// typed input: free typing, validated + clamped (+ snapped) to range on blur / Enter
function NumField({
  value,
  min,
  max,
  float,
  snap,
  onCommit,
}: {
  value: number;
  min: number;
  max: number;
  float?: boolean;
  snap?: (n: number) => number;
  onCommit: (v: number) => void;
}) {
  const [text, setText] = useState(String(value));
  useEffect(() => setText(String(value)), [value]);
  const commit = () => {
    let n = Number(text);
    if (!Number.isFinite(n)) {
      setText(String(value));
      return;
    }
    n = Math.min(max, Math.max(min, n));
    if (!float) n = Math.round(n);
    if (snap) n = Math.min(max, Math.max(min, snap(n)));
    onCommit(n);
    setText(String(n));
  };
  return (
    <input
      className="cfg-num mono"
      type="text"
      inputMode={float ? "decimal" : "numeric"}
      value={text}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") e.currentTarget.blur();
      }}
    />
  );
}

// --- train ---

function TrainSection({ expanded }: { expanded: boolean }) {
  const modelId = useCanvasStore((s) => s.modelId);
  const compiled = useCompileStore((s) => s.status === "ready");
  const projects = useProjectsStore((s) => s.projects);
  const run = useTrainStore();
  // stats/curve belong to a run; only show them when the active model owns the current run
  const mine = run.modelId === modelId;
  const status = mine ? run.status : "idle";
  const running = status === "running" || status === "stopping";

  // the worker trains one model at a time; if a DIFFERENT model is mid-run, block Train here
  const otherTraining = !mine && (run.status === "running" || run.status === "stopping");
  const otherName = otherTraining
    ? (projects.find((p) => p.id === run.modelId)?.title ?? "another model")
    : "";

  const [steps, setSteps] = useState("2000");
  const [batch, setBatch] = useState("16");
  const [lr, setLr] = useState("3e-4");

  // on mount / model switch, reattach to whatever run the worker is doing (survives a page reload).
  // We follow the actual training model - even if it isn't the one open - so the store always
  // mirrors the single global run and other projects can disable their Train button. Guarded so we
  // don't re-follow a run this session is already streaming.
  useEffect(() => {
    if (useTrainStore.getState().status === "running" || useTrainStore.getState().status === "stopping") return;
    let cancelled = false;
    api
      .trainStatus()
      .then((s) => {
        if (!cancelled && s?.status === "running" && s.training_id) {
          useTrainStore.getState().follow(s.training_id);
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [modelId]);

  const onTrain = () => {
    if (running) {
      run.stop();
      return;
    }
    run.start({
      id: modelId,
      max_steps: Math.max(1, Math.round(Number(steps)) || 2000),
      batch_size: Math.max(1, Math.round(Number(batch)) || 16),
      learning_rate: Number(lr) || 3e-4,
      bench: true,
    });
  };

  const gate = otherTraining
    ? `Training "${otherName}" is in progress`
    : !compiled
      ? "Compile the model before training"
      : "";
  const btnLabel = status === "stopping" ? "stopping…" : running ? "Stop" : "Train";
  const loss = mine ? run.loss : null;
  const sps = mine ? run.stepsPerSec : null;
  const tip = useTooltip(gate); // on a wrapper span so the hint shows even while the button is disabled

  return (
    <section className="grp">
      <h3>Train</h3>
      <div className="pan-body">
        <LossGraph curve={mine ? run.curve : []} maxSteps={mine ? run.maxSteps : 0} tall={expanded} />
        <div className="stat-row">
          <Stat label="step" value={mine && (running || status !== "idle") ? `${run.step}/${run.maxSteps}` : "—"} />
          <Stat label="loss" value={loss != null ? loss.toFixed(3) : "—"} />
          <Stat label="steps/s" value={sps != null ? sps.toFixed(1) : "—"} />
        </div>
        <div className="hp-row">
          <HPField label="steps" value={steps} onChange={setSteps} disabled={running} />
          <HPField label="batch" value={batch} onChange={setBatch} disabled={running} />
          <HPField label="lr" value={lr} onChange={setLr} disabled={running} />
        </div>
        {/* wrapper carries the tooltip so the hint shows even while the button is disabled */}
        <span className="btn-wrap" {...tip}>
          <button
            type="button"
            className={`btn ${running ? "danger" : "primary"}`}
            disabled={!running && (otherTraining || !compiled)}
            onClick={onTrain}
          >
            {btnLabel}
          </button>
        </span>
        <RunNote status={status} error={mine ? run.error : null} saved={status === "completed"} />
      </div>
    </section>
  );
}

// short status line under the Train button (done / stopped / error), mirroring the run state
function RunNote({ status, error, saved }: { status: string; error: string | null; saved: boolean }) {
  if (status === "error") return <div className="run-note bad">{error ?? "training failed"}</div>;
  if (status === "stopped") return <div className="run-note">stopped — not saved (last trained weights kept)</div>;
  if (saved) return <div className="run-note ok">✓ trained &amp; saved</div>;
  return null;
}

function GenerateSection({ expanded }: { expanded: boolean }) {
  return (
    <section className="grp">
      <h3>Generate</h3>
      <div className="pan-body">
        <textarea className="gen-prompt" placeholder="Enter a prompt…" rows={expanded ? 4 : 3} />
        <div className="hp-row">
          <HP label="temp" def="0.8" />
          <HP label="top_k" def="200" />
          <HP label="tokens" def="256" />
        </div>
        <button type="button" className="btn primary">
          Generate
        </button>
        <div className="gen-output mono">Output will stream here…</div>
      </div>
    </section>
  );
}

// --- benchmark ---

function BenchmarkSection({
  expanded,
  open,
  onToggle,
}: {
  expanded: boolean;
  open: boolean;
  onToggle: (v: boolean) => void;
}) {
  return (
    <section className="grp">
      {/* subheading is always visible; the toggle shows/hides the section body. No bottom margin
          when closed so the collapsed section stays vertically balanced. */}
      <h3 style={open ? undefined : { marginBottom: 0 }}>
        Benchmark
        <button
          type="button"
          role="switch"
          className={`sw${open ? " on" : ""}`}
          aria-checked={open}
          aria-label={open ? "Hide benchmark" : "Show benchmark"}
          onClick={() => onToggle(!open)}
        >
          <span className="sw-knob" />
        </button>
      </h3>
      {open && (
        <div className="pan-body">
          <ProfilePanel />
          <ComparePanel expanded={expanded} />
        </div>
      )}
    </section>
  );
}

// per-node profiler for the active model: bars sized by each node's share of the forward pass
function ProfilePanel() {
  const modelId = useCanvasStore((s) => s.modelId);
  const compiled = useCompileStore((s) => s.status === "ready");
  const trained = useCompileStore((s) => s.trained);
  const { profiling, profile, profileError, runProfile } = useBenchStore();

  const top = useMemo(() => (profile ? profile.nodes.slice(0, 8) : []), [profile]);
  // the profiler times the model in the worker's cache: it must be compiled and trained
  const gate = !compiled ? "Compile the model before profiling" : !trained ? "Train the model before profiling" : "";

  return (
    <div className="bench-block">
      <div className="bench-head">
        <span className="bench-sub">Per-node profile</span>
        <button
          type="button"
          className="btn tiny"
          disabled={profiling || !compiled || !trained}
          title={gate || undefined}
          onClick={() => runProfile({ id: modelId, mode: "decode" })}
        >
          {profiling ? "profiling…" : "Profile"}
        </button>
      </div>
      {profileError && <div className="run-note bad">{profileError}</div>}
      {top.length > 0 && (
        <div className="bench-nodes">
          {top.map((n) => (
            <BenchBar key={n.node_id} label={n.logical_id} pct={n.pct} sub={`${n.pct.toFixed(0)}%`} />
          ))}
        </div>
      )}
      {!profile && !profileError && <div className="bench-empty">Run to time each node’s share of a decode step.</div>}
    </div>
  );
}

// head-to-head compare: the active model vs. other saved projects, generation timing side by side
function ComparePanel({ expanded }: { expanded: boolean }) {
  const modelId = useCanvasStore((s) => s.modelId);
  const toGraph = useCanvasStore((s) => s.toGraph);
  const projects = useProjectsStore((s) => s.projects);
  const { comparing, columns, runCompare } = useBenchStore();

  const current = projects.find((p) => p.id === modelId);
  const others = projects.filter((p) => p.id !== modelId);
  const [picked, setPicked] = useState<string[]>([]);
  const [prompt, setPrompt] = useState("The ");

  // cap total columns at the backend's 5 (current model + up to 4 others)
  const toggle = (id: string) =>
    setPicked((s) => (s.includes(id) ? s.filter((x) => x !== id) : s.length >= 4 ? s : [...s, id]));

  const onRun = () => {
    const entries = [
      { id: modelId, graph: toGraph(), label: current?.title ?? "current" },
      ...picked
        .map((id) => {
          const g = graphForProject(id);
          return g ? { id, graph: g, label: projects.find((p) => p.id === id)?.title ?? id } : null;
        })
        .filter((e): e is { id: string; graph: ReturnType<typeof toGraph>; label: string } => e !== null),
    ];
    runCompare(entries, { prompt, max_new_tokens: 64, temperature: 0.8, top_k: 200 });
  };

  return (
    <div className="bench-block">
      <div className="bench-head">
        <span className="bench-sub">Compare models</span>
        <button type="button" className="btn tiny" disabled={comparing} onClick={onRun}>
          {comparing ? "running…" : "Run"}
        </button>
      </div>

      <input
        className="hp-in mono cmp-prompt"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="prompt"
        aria-label="compare prompt"
      />

      {others.length > 0 ? (
        <div className="cmp-picker">
          {others.map((p) => {
            const on = picked.includes(p.id);
            return (
              <button
                key={p.id}
                type="button"
                className={`chip${on ? " on" : ""}`}
                aria-pressed={on}
                disabled={!on && picked.length >= 4}
                onClick={() => toggle(p.id)}
              >
                {p.title}
              </button>
            );
          })}
        </div>
      ) : (
        <div className="bench-empty">Save more models to compare against.</div>
      )}

      {columns.length > 0 && <CompareTable expanded={expanded} />}
    </div>
  );
}

function CompareTable({ expanded }: { expanded: boolean }) {
  const columns = useBenchStore((s) => s.columns);
  const fmt = (v: number | null, d: number) => (v != null ? v.toFixed(d) : "—");
  return (
    <div className="cmp-scroll">
      <table className="cmp-table mono">
        <thead>
          <tr>
            <th />
            {columns.map((c) => (
              <th key={c.id}>{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          <CmpRow label="tok/s" cells={columns.map((c) => (c.error ? "err" : fmt(c.tokensPerSec, 0)))} best={bestIdx(columns, (c) => c.tokensPerSec, "max")} />
          <CmpRow label="ms/tok" cells={columns.map((c) => (c.error ? "err" : fmt(c.msPerToken, 1)))} best={bestIdx(columns, (c) => c.msPerToken, "min")} />
          <CmpRow label="prefill ms" cells={columns.map((c) => (c.error ? "err" : fmt(c.prefillMs, 0)))} best={bestIdx(columns, (c) => c.prefillMs, "min")} />
        </tbody>
      </table>
      {expanded &&
        columns.map((c) => (
          <div key={c.id} className="cmp-sample">
            <span className="cmp-sample-k">{c.label}</span>
            <span className="cmp-sample-v mono">{c.error ? c.error : c.text || (c.done ? "" : "…")}</span>
          </div>
        ))}
    </div>
  );
}

function CmpRow({ label, cells, best }: { label: string; cells: string[]; best: number }) {
  return (
    <tr>
      <td className="cmp-k">{label}</td>
      {cells.map((v, i) => (
        // index key is stable here: columns keep their order for the whole run
        <td key={i} className={i === best ? "cmp-best" : undefined}>
          {v}
        </td>
      ))}
    </tr>
  );
}

// index of the winning column for a metric (highest tok/s, lowest latency), ignoring errored/pending
function bestIdx(cols: { error: string | null }[], pick: (c: any) => number | null, dir: "max" | "min"): number {
  let best = -1;
  let val = dir === "max" ? -Infinity : Infinity;
  cols.forEach((c, i) => {
    const v = (pick as (c: unknown) => number | null)(c);
    if (c.error || v == null) return;
    if (dir === "max" ? v > val : v < val) {
      val = v;
      best = i;
    }
  });
  return best;
}

// --- bits ---

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <span className="stat-v mono">{value}</span>
      <span className="stat-k">{label}</span>
    </div>
  );
}

// uncontrolled hyperparameter (generate); controlled variant below for the train fields
function HP({ label, def }: { label: string; def: string }) {
  return (
    <label className="hp">
      <span className="hp-k">{label}</span>
      <input className="hp-in mono" defaultValue={def} />
    </label>
  );
}

function HPField({
  label,
  value,
  onChange,
  disabled,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  disabled?: boolean;
}) {
  return (
    <label className="hp">
      <span className="hp-k">{label}</span>
      <input
        className="hp-in mono"
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}

// live loss curve: the per-step train loss, auto-scaled to the observed loss range. It advances
// every step (no eval pauses), so the line just grows smoothly as training runs.
function LossGraph({ curve, maxSteps, tall }: { curve: { step: number; loss: number }[]; maxSteps: number; tall?: boolean }) {
  const train = useMemo(() => {
    if (curve.length === 0) return "";
    const losses = curve.map((p) => p.loss);
    const lo = Math.min(...losses);
    const hi = Math.max(...losses);
    const range = hi - lo || 1;
    const xMax = maxSteps || curve[curve.length - 1].step || 1;
    const x = (step: number) => (step / xMax) * 100;
    const y = (v: number) => 39 - ((v - lo) / range) * 37; // invert; pad 1 top / 2 bottom
    return curve.map((p) => `${x(p.step).toFixed(2)},${y(p.loss).toFixed(2)}`).join(" ");
  }, [curve, maxSteps]);

  return (
    <svg className="loss-graph" viewBox="0 0 100 40" preserveAspectRatio="none" style={{ height: tall ? 180 : 92 }} aria-hidden="true">
      {train ? (
        <polyline points={train} fill="none" stroke="var(--ice)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
      ) : (
        <line x1="0" y1="20" x2="100" y2="20" stroke="var(--line2)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
      )}
    </svg>
  );
}

function BenchBar({ label, pct, sub }: { label: string; pct: number; sub?: string }) {
  return (
    <div className="bench-bar">
      <span className="bb-label mono">{label}</span>
      <span className="bb-track">
        <span className="bb-fill" style={{ width: `${Math.min(100, pct)}%` }} />
      </span>
      {sub && <span className="bb-sub mono">{sub}</span>}
    </div>
  );
}
