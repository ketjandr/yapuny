// Right pane: mode-aware controls.
//  Train -> Config + Train (aggregate loss curve, current-model stats, hyperparams, stop) + a
//    training Benchmark section: pick up to 5 compiled models, Train runs them sequentially and
//    fills a fwd/bwd/steps-s/peak-vram table + per-model info & node profile.
//  Inference -> Generate + a Benchmark section (per-node profiler + head-to-head generate compare).
// The Benchmark toggle is persisted per project per mode and doubles as "benchmarking enabled";
// it's locked while a run is in flight. Wired to the worker over SSE (trainStore / benchTrainStore /
// benchStore). Collapsible + full-view expandable via PaneShell.
import { useEffect, useMemo, useState } from "react";
import { useTooltip } from "@/components/tooltipContext";
import { api } from "@/lib/api";
import { graphForProject, graphToCanvas } from "@/lib/graph";
import { structuralIds } from "@/lib/structuralId";
import type { GraphRequest } from "@/lib/types";
import { useBenchStore } from "@/store/benchStore";
import { type BenchModel, useBenchTrainStore } from "@/store/benchTrainStore";
import { N_LAYER_MAX, N_LAYER_MIN, useCanvasStore } from "@/store/canvasStore";
import { useCompileStore } from "@/store/compileStore";
import { useProjectsStore } from "@/store/projectsStore";
import { useTrainStore } from "@/store/trainStore";
import { PaneShell } from "./PaneShell";

// one distinct color per benchmarked model (curve line + row/chip swatch), current model first
const BENCH_COLORS = ["#e6a24d", "#8fce6a", "#77d3e6", "#d9738a", "#b892e0"];

interface CompareEntry {
  id: string;
  graph: GraphRequest;
  title: string;
}
interface ModelInfo {
  param_count: number;
  vocab_size: number;
  block_size: number;
  n_layer?: number;
  n_head?: number;
  n_embd?: number;
}
interface CompareInfo {
  ready: boolean;
  trained: boolean;
  info: ModelInfo | null;
}

export function RightPane() {
  const mode = useCanvasStore((s) => s.mode);
  const modelId = useCanvasStore((s) => s.modelId);
  const benchOpen = useCanvasStore((s) => s.benchOpen);
  const setBenchOpen = useCanvasStore((s) => s.setBenchOpen);
  const setSelected = useBenchTrainStore((s) => s.setSelected);
  const clearSelected = useBenchTrainStore((s) => s.clearSelected);
  const [expanded, setExpanded] = useState(false);
  const title = mode === "train" ? "Training" : "Inference";

  // On project switch / reload: reattach to whatever the worker is running so a run (single or
  // bench) survives, other projects can disable their Train, and the bench's selection + curves come
  // back. The compare selection is per-project, so it's restored from the run or cleared for a fresh
  // project. Runs live in the worker (background thread), independent of any open tab.
  useEffect(() => {
    let cancelled = false;
    const bt = useBenchTrainStore.getState();
    // the run is "owned" by its column-0 model; only that project shows the run + its selection
    const ownsRun = (models?: { id: string }[]) => models?.[0]?.id === modelId;

    // a run this session already tracks (no reload): restore selection only if we own it
    if (bt.models.length > 0 && ownsRun(bt.models)) {
      setSelected(bt.models.slice(1).map((m) => m.id));
      return;
    }
    if (bt.status === "running" || bt.status === "stopping") return; // someone else's run; keep our selection as-is

    clearSelected();
    // fresh / reload: ask the worker what's running and reattach
    api
      .trainBenchStatus()
      .then((s) => {
        if (cancelled) return;
        if (ownsRun(s?.models)) {
          setSelected(s.models.slice(1).map((m: { id: string }) => m.id)); // restore our compare set
          useBenchTrainStore.getState().follow(); // repopulate models + curves (running or just-finished)
        } else if (s?.status === "running") {
          useBenchTrainStore.getState().follow(); // reflect another project's run so Train disables here
        } else {
          api.trainStatus().then((t) => {
            if (!cancelled && t?.status === "running" && t.training_id) useTrainStore.getState().follow(t.training_id);
          });
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [modelId, setSelected, clearSelected]);

  return (
    <PaneShell side="right" title={title} expanded={expanded} onToggleExpand={() => setExpanded((v) => !v)}>
      <div className="pane-body">
        {mode === "train" ? (
          <TrainControls expanded={expanded} open={benchOpen.train} onToggle={(v) => setBenchOpen("train", v)} />
        ) : (
          <>
            <GenerateSection expanded={expanded} />
            <InferenceBenchmark expanded={expanded} open={benchOpen.inference} onToggle={(v) => setBenchOpen("inference", v)} />
          </>
        )}
      </div>
    </PaneShell>
  );
}

// --- train mode: shared state for the train controls + training benchmark ---

function TrainControls({ expanded, open, onToggle }: { expanded: boolean; open: boolean; onToggle: (v: boolean) => void }) {
  const modelId = useCanvasStore((s) => s.modelId);
  const toGraph = useCanvasStore((s) => s.toGraph);
  const nodes = useCanvasStore((s) => s.nodes);
  const edges = useCanvasStore((s) => s.edges);
  const meta = useCanvasStore((s) => s.meta);
  const projects = useProjectsStore((s) => s.projects);
  const selected = useBenchTrainStore((s) => s.selected);
  const runStatus = useBenchTrainStore((s) => s.status);

  const titleOf = (id: string) => projects.find((p) => p.id === id)?.title ?? id;

  // benchmark set = the open model (always column 0) + selected others, each as {id, graph, title}
  // biome deps: recompute when the graph or selection changes
  const entries = useMemo<CompareEntry[]>(() => {
    const cur: CompareEntry = { id: modelId, graph: toGraph(), title: titleOf(modelId) };
    const others = selected
      .map((id) => {
        const g = graphForProject(id);
        return g ? { id, graph: g, title: titleOf(id) } : null;
      })
      .filter((e): e is CompareEntry => e !== null);
    return [cur, ...others];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelId, selected, projects, nodes, edges, meta]);

  const infos = useCompareInfos(entries, runStatus);

  return (
    <>
      <ConfigSection expanded={expanded} />
      <TrainSection expanded={expanded} benchOn={open} entries={entries} infos={infos} />
      <TrainBenchmark expanded={expanded} open={open} onToggle={onToggle} entries={entries} infos={infos} />
    </>
  );
}

// fetch each model's readiness + info (param count, config) via /model/status; refetch when the set
// changes or a run ends (so trained/param stats refresh). The open model's live compile status is
// tracked separately by compileStore - this is for the info panel + gating the other models.
function useCompareInfos(entries: CompareEntry[], trigger: string): Record<string, CompareInfo> {
  const [map, setMap] = useState<Record<string, CompareInfo>>({});
  const key = `${entries.map((e) => e.id).join(",")}|${trigger}`;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const stable = useMemo(() => entries, [key]);
  useEffect(() => {
    let cancelled = false;
    Promise.all(
      stable.map(async (e) => {
        try {
          const r = await api.modelStatus({ id: e.id, graph: e.graph });
          return [e.id, { ready: r.status === "ready", trained: !!r.trained, info: r.model_info ?? null }] as const;
        } catch {
          return [e.id, { ready: false, trained: false, info: null }] as const;
        }
      }),
    ).then((rows) => {
      if (!cancelled) setMap(Object.fromEntries(rows));
    });
    return () => {
      cancelled = true;
    };
  }, [stable]);
  return map;
}

// --- config ---

function ConfigSection({ expanded }: { expanded: boolean }) {
  const meta = useCanvasStore((s) => s.meta);
  const setMeta = useCanvasStore((s) => s.setMeta);
  const snapEmbd = (n: number) => Math.max(meta.n_head, Math.round(n / meta.n_head) * meta.n_head);
  return (
    <section className="grp">
      <h3>Config</h3>
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
      <input className="cfg-range" type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  );
}

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

function TrainSection({
  expanded,
  benchOn,
  entries,
  infos,
}: {
  expanded: boolean;
  benchOn: boolean;
  entries: CompareEntry[];
  infos: Record<string, CompareInfo>;
}) {
  const modelId = useCanvasStore((s) => s.modelId);
  const projects = useProjectsStore((s) => s.projects);
  const compiledCurrent = useCompileStore((s) => s.status === "ready");
  const single = useTrainStore();
  const bench = useBenchTrainStore();

  const [steps, setSteps] = useState("500");
  const [batch, setBatch] = useState("16");
  const [lr, setLr] = useState("3e-4");

  // (reconnect to an in-progress run + selection restore is handled once in RightPane)

  const titleOf = (id: string) => projects.find((p) => p.id === id)?.title ?? id;

  // run-in-progress detection. A bench run is "owned" by the project that started it (column 0);
  // only the owner shows the run - every other project greys out (the worker is busy), even models
  // that happen to be in the compare set, since they can't start their own run.
  const singleRunning = single.status === "running" || single.status === "stopping";
  const benchRunning = bench.status === "running" || bench.status === "stopping";
  const benchOwner = bench.models[0]?.id === modelId;
  const singleMine = single.modelId === modelId;
  const otherBusy = (singleRunning && !singleMine) || (benchRunning && !benchOwner);
  const running = benchOn ? benchRunning && benchOwner : singleRunning && singleMine;

  // results are isolated per project: only show the bench run on the project that owns it, and the
  // single run on the project it belongs to. Everything else shows an empty (idle) view.
  const showBench = benchOn && benchOwner;
  const showSingle = !benchOn && singleMine;
  const curModel: BenchModel | null = showBench ? (bench.models[bench.current] ?? null) : null;
  const activeName = !running ? "" : benchOn ? (curModel ? titleOf(curModel.id) : "") : titleOf(modelId);
  // in a benchmark run, the heading name + tokenizing badge take the current model's assigned color
  const activeColor = showBench ? BENCH_COLORS[bench.current % BENCH_COLORS.length] : "var(--ice)";

  // gating: bench needs every selected model compiled; single needs the open model compiled
  const othersReady = entries.slice(1).every((e) => infos[e.id]?.ready);
  const allCompiled = compiledCurrent && othersReady;
  const canStart = benchOn ? allCompiled : compiledCurrent;

  const hp = () => ({
    max_steps: Math.max(1, Math.round(Number(steps)) || 500),
    batch_size: Math.max(1, Math.round(Number(batch)) || 16),
    learning_rate: Number(lr) || 3e-4,
  });

  const onTrain = () => {
    if (running) {
      (benchOn ? bench.stop : single.stop)();
      return;
    }
    if (benchOn) bench.start(entries.map((e) => e.id), hp());
    else single.start({ id: modelId, ...hp(), bench: false });
  };

  const gate = otherBusy
    ? `Training "${titleOf(otherBusyId(single, bench))}" is in progress`
    : benchOn && !allCompiled
      ? "Compile all selected models first"
      : !compiledCurrent
        ? "Compile the model before training"
        : "";
  const status = showBench ? bench.status : showSingle ? single.status : "idle";
  const btnLabel = status === "stopping" ? "stopping…" : running ? "Stop" : benchOn ? "Train & benchmark" : "Train";

  // stats: current model in this project's bench run, else its single run, else nothing
  const step = showBench ? (curModel?.step ?? 0) : showSingle ? single.step : 0;
  const maxSteps = showBench ? (curModel?.maxSteps ?? 0) : showSingle ? single.maxSteps : 0;
  const loss = showBench ? (curModel?.loss ?? null) : showSingle ? single.loss : null;
  const sps = showBench ? (curModel?.stepsPerSec ?? null) : showSingle ? single.stepsPerSec : null;

  // loss curve: one colored line per model for this project's bench run, else its single line
  const series = showBench
    ? bench.models.map((m, i) => ({ color: BENCH_COLORS[i % BENCH_COLORS.length], curve: m.curve }))
    : [{ color: "var(--ice)", curve: showSingle ? single.curve : [] }];
  const curveMax = showBench ? (curModel?.maxSteps ?? 0) : showSingle ? single.maxSteps : 0;

  // each model trains its own tokenizer first, which delays stepping - surface that phase
  const tokenizing = running && (benchOn ? curModel?.phase === "tokenizing" : single.phase === "tokenizing");

  const tip = useTooltip(gate);

  return (
    <section className="grp">
      <h3>
        Train
        {activeName && (
          <span className="grp-sub mono" style={{ color: activeColor }}>
            {activeName}
          </span>
        )}
      </h3>
      <div className="pan-body">
        <div className="loss-wrap">
          <LossGraph series={series} maxSteps={curveMax} tall={expanded} />
          {tokenizing && (
            <span className="loss-badge" style={{ color: activeColor, borderColor: activeColor }}>
              tokenizing corpus…
            </span>
          )}
        </div>
        <div className="stat-row">
          <Stat label="step" value={running || status !== "idle" ? `${step}/${maxSteps}` : "—"} />
          <Stat label="loss" value={loss != null ? loss.toFixed(3) : "—"} />
          <Stat label="steps/s" value={sps != null ? sps.toFixed(1) : "—"} />
        </div>
        <div className="hp-row">
          <HPField label="steps" value={steps} onChange={setSteps} disabled={running} />
          <HPField label="batch" value={batch} onChange={setBatch} disabled={running} />
          <HPField label="lr" value={lr} onChange={setLr} disabled={running} />
        </div>
        <span className="btn-wrap" {...tip}>
          <button
            type="button"
            className={`btn ${running ? "danger" : "primary"}`}
            disabled={!running && (otherBusy || !canStart)}
            onClick={onTrain}
          >
            {btnLabel}
          </button>
        </span>
        <RunNote status={status} error={benchOn ? bench.error : singleMine ? single.error : null} saved={status === "completed"} bench={benchOn} />
      </div>
    </section>
  );
}

// the model id of whatever run is currently occupying the worker (for the "in progress" tooltip)
function otherBusyId(single: { modelId: string | null }, bench: { models: BenchModel[]; current: number }): string {
  return bench.models[bench.current]?.id ?? single.modelId ?? "";
}

function RunNote({ status, error, saved, bench }: { status: string; error: string | null; saved: boolean; bench: boolean }) {
  if (status === "error") return <div className="run-note bad">{error ?? "training failed"}</div>;
  if (status === "stopped") return <div className="run-note">training stopped - not saved (last trained weights kept)</div>;
  if (saved) return <div className="run-note ok">{bench ? "✓ benchmark complete - models trained & saved" : "✓ trained & saved"}</div>;
  return null;
}

// --- training benchmark: model picker + fwd/bwd/steps-s/vram table + per-model info & profile ---

function TrainBenchmark({
  expanded,
  open,
  onToggle,
  entries,
  infos,
}: {
  expanded: boolean;
  open: boolean;
  onToggle: (v: boolean) => void;
  entries: CompareEntry[];
  infos: Record<string, CompareInfo>;
}) {
  const modelId = useCanvasStore((s) => s.modelId);
  const projects = useProjectsStore((s) => s.projects);
  const bench = useBenchTrainStore();
  const selected = bench.selected;
  const toggleSelected = bench.toggleSelected;
  // results are isolated per project: this project only sees a run it owns (column 0). Other projects
  // never see it - they show their own empty benchmark. The toggle + picker lock only on the owner.
  const owned = bench.models[0]?.id === modelId;
  const myModels = owned ? bench.models : [];
  const running = (bench.status === "running" || bench.status === "stopping") && owned;

  const others = projects.filter((p) => p.id !== modelId);

  // which model's detail (info + node profile) is shown; default to the open model
  const [detailId, setDetailId] = useState<string | null>(null);
  const detail = detailId ?? entries[0]?.id;

  return (
    <section className="grp">
      <h3 style={open ? undefined : { marginBottom: 0 }}>
        Benchmark
        <button
          type="button"
          role="switch"
          className={`sw${open ? " on" : ""}`}
          aria-checked={open}
          aria-label={open ? "Disable benchmark" : "Enable benchmark"}
          disabled={running}
          onClick={() => onToggle(!open)}
        >
          <span className="sw-knob" />
        </button>
      </h3>
      {open && (
        <div className="pan-body">
          {/* model picker: the open model is always column 0; add up to 4 others (5 total) */}
          <div className="cmp-picker">
            {entries.map((e, i) => (
              <span key={e.id} className="chip on" style={{ borderColor: BENCH_COLORS[i % BENCH_COLORS.length] }}>
                <span className="chip-dot" style={{ background: BENCH_COLORS[i % BENCH_COLORS.length] }} />
                {i === 0 ? `${e.title} (this)` : e.title}
              </span>
            ))}
            {!running &&
              others
                .filter((p) => !selected.includes(p.id))
                .map((p) => (
                  <button
                    key={p.id}
                    type="button"
                    className="chip"
                    disabled={selected.length >= 4}
                    onClick={() => toggleSelected(p.id)}
                  >
                    + {p.title}
                  </button>
                ))}
          </div>

          {/* the table rows double as the model selector for the detail panel below */}
          <BenchTable entries={entries} models={myModels} detail={detail} onSelect={setDetailId} expanded={expanded} />

          {/* per-model detail: config + params + node profile, for the selected row */}
          <div className="bench-detail">
            <ModelDetail
              info={infos[detail]?.info ?? null}
              model={myModels.find((m) => m.id === detail) ?? null}
              graph={entries.find((e) => e.id === detail)?.graph}
              expanded={expanded}
            />
          </div>
        </div>
      )}
    </section>
  );
}

// one row per benchmarked model: fwd/bwd ms, steps/s (with a throughput bar), peak vram. Rows are
// the selector for the detail panel - click one to show that model's config + node profile.
function BenchTable({
  entries,
  models,
  detail,
  onSelect,
  expanded,
}: {
  entries: CompareEntry[];
  models: BenchModel[];
  detail?: string;
  onSelect: (id: string) => void;
  expanded: boolean;
}) {
  const byId = new Map(models.map((m) => [m.id, m]));
  const maxSps = Math.max(1, ...models.map((m) => m.bench?.steps_per_sec ?? 0));
  const cell = (v: number | null | undefined, d: number) => (v != null ? v.toFixed(d) : "—");

  return (
    <div className="bt-wrap">
      <table className="bt-table mono">
        <thead>
          <tr>
            <th className="bt-name">model</th>
            <th>fwd ms</th>
            <th>bwd ms</th>
            <th className="bt-tp">steps / s</th>
            <th>peak vram</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((e, i) => {
            const m = byId.get(e.id);
            const b = m?.bench;
            const color = BENCH_COLORS[i % BENCH_COLORS.length];
            const pending = !b && m?.status !== "running";
            return (
              <tr key={e.id} className={`bt-row${detail === e.id ? " sel" : ""}`} onClick={() => onSelect(e.id)}>
                <td className="bt-name">
                  <span className="bt-bar" style={{ background: color }} />
                  {e.title}
                </td>
                <td>{cell(b?.forward_ms, 2)}</td>
                <td>{cell(b?.backward_ms, 2)}</td>
                <td className="bt-tp">
                  {b ? (
                    <span className="bt-tp-wrap">
                      <span className="bt-tp-num">{b.steps_per_sec.toFixed(1)}</span>
                      {/* the throughput bar only adds value with the room of full view */}
                      {expanded && (
                        <span className="bb-track">
                          <span className="bb-fill" style={{ width: `${(b.steps_per_sec / maxSps) * 100}%`, background: color }} />
                        </span>
                      )}
                    </span>
                  ) : m?.status === "running" ? (
                    <span className="bt-live">{m.phase === "tokenizing" ? "tokenizing…" : `training… ${m.step}/${m.maxSteps}`}</span>
                  ) : (
                    <span className="bench-empty">{pending ? "pending" : "—"}</span>
                  )}
                </td>
                <td>{b?.peak_vram_mb != null ? `${(b.peak_vram_mb / 1024).toFixed(2)} GB` : "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ModelDetail({
  info,
  model,
  graph,
  expanded,
}: {
  info: ModelInfo | null;
  model: BenchModel | null;
  graph?: GraphRequest;
  expanded: boolean;
}) {
  const nodes = model?.bench?.profile.nodes ?? [];
  // collapsed: top 8 with a "+N more" note; expanded: the full profile
  const top = expanded ? nodes : nodes.slice(0, 8);
  const more = nodes.length - top.length;
  const requestFocusNode = useCanvasStore((s) => s.requestFocusNode);
  // click-to-focus only makes sense for the open model - other models have a different canvas
  const isOpenModel = useCanvasStore((s) => s.modelId) === model?.id;
  // map each profiled node id to the same readable structural id the rest of the frontend uses
  const names = useMemo(() => {
    if (!graph) return new Map<string, string>();
    const { nodes: n, edges: e } = graphToCanvas(graph);
    return structuralIds(n, e);
  }, [graph]);
  return (
    <div className="bench-block">
      {info ? (
        <div className="info-grid">
          <Info k="params" v={fmtParams(info.param_count)} />
          <Info k="n_layer" v={String(info.n_layer ?? "—")} />
          <Info k="n_head" v={String(info.n_head ?? "—")} />
          <Info k="n_embd" v={String(info.n_embd ?? "—")} />
          <Info k="block" v={String(info.block_size)} />
          <Info k="vocab" v={String(info.vocab_size)} />
        </div>
      ) : (
        <div className="bench-empty">Compile this model to see its config.</div>
      )}
      {top.length > 0 ? (
        <div className="bench-nodes">
          <span className="bench-sub">Node profile (train step)</span>
          {(() => {
            const labels = top.map((n) => names.get(n.logical_id) ?? n.logical_id);
            // start every bar past the longest label so tracks line up (clamped so a long id can't eat the track)
            const labelCh = Math.min(18, Math.max(...labels.map((l) => l.length)) + 1);
            return top.map((n, i) => (
              <BenchBar
                key={n.node_id}
                label={labels[i]}
                pct={n.pct}
                sub={`${n.pct.toFixed(0)}%`}
                labelCh={labelCh}
                onClick={expanded || !isOpenModel ? undefined : () => requestFocusNode(n.logical_id)}
              />
            ));
          })()}
          {more > 0 && <span className="bench-more">+{more} nodes…</span>}
        </div>
      ) : (
        <div className="bench-empty">Run the benchmark to profile this model’s nodes.</div>
      )}
    </div>
  );
}

function Info({ k, v }: { k: string; v: string }) {
  return (
    <div className="info-cell">
      <span className="info-v mono">{v}</span>
      <span className="info-k">{k}</span>
    </div>
  );
}

function fmtParams(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

// --- inference: generate + benchmark (per-node profiler + generate compare) ---

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

function InferenceBenchmark({ expanded, open, onToggle }: { expanded: boolean; open: boolean; onToggle: (v: boolean) => void }) {
  return (
    <section className="grp">
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
          <ProfilePanel expanded={expanded} />
          <ComparePanel expanded={expanded} />
        </div>
      )}
    </section>
  );
}

function ProfilePanel({ expanded }: { expanded: boolean }) {
  const modelId = useCanvasStore((s) => s.modelId);
  const nodes = useCanvasStore((s) => s.nodes);
  const edges = useCanvasStore((s) => s.edges);
  const requestFocusNode = useCanvasStore((s) => s.requestFocusNode);
  const compiled = useCompileStore((s) => s.status === "ready");
  const trained = useCompileStore((s) => s.trained);
  const { profiling, profile, profileError, runProfile } = useBenchStore();

  // collapsed: top 8 with a "+N more" note; expanded: the full profile
  const top = useMemo(() => (profile ? (expanded ? profile.nodes : profile.nodes.slice(0, 8)) : []), [profile, expanded]);
  const more = (profile?.nodes.length ?? 0) - top.length;
  // map profiled node ids to the frontend's structural ids (same as the properties panel)
  const names = useMemo(() => structuralIds(nodes, edges), [nodes, edges]);
  const gate = !compiled ? "Compile the model before profiling" : !trained ? "Train the model before profiling" : "";

  return (
    <div className="bench-block">
      <div className="bench-head">
        <span className="bench-sub">Per-node profile</span>
        <button type="button" className="btn tiny" disabled={profiling || !compiled || !trained} title={gate || undefined} onClick={() => runProfile({ id: modelId, mode: "decode" })}>
          {profiling ? "profiling…" : "Profile"}
        </button>
      </div>
      {profileError && <div className="run-note bad">{profileError}</div>}
      {top.length > 0 && (
        <div className="bench-nodes">
          {(() => {
            const labels = top.map((n) => names.get(n.logical_id) ?? n.logical_id);
            const labelCh = Math.min(18, Math.max(...labels.map((l) => l.length)) + 1);
            return top.map((n, i) => (
              <BenchBar
                key={n.node_id}
                label={labels[i]}
                pct={n.pct}
                sub={`${n.pct.toFixed(0)}%`}
                labelCh={labelCh}
                onClick={expanded ? undefined : () => requestFocusNode(n.logical_id)}
              />
            ));
          })()}
          {more > 0 && <span className="bench-more">+{more} nodes…</span>}
        </div>
      )}
      {!profile && !profileError && <div className="bench-empty">Run to time each node’s share of a decode step.</div>}
    </div>
  );
}

function ComparePanel({ expanded }: { expanded: boolean }) {
  const modelId = useCanvasStore((s) => s.modelId);
  const toGraph = useCanvasStore((s) => s.toGraph);
  const projects = useProjectsStore((s) => s.projects);
  const { comparing, columns, runCompare } = useBenchStore();

  const current = projects.find((p) => p.id === modelId);
  const others = projects.filter((p) => p.id !== modelId);
  const [picked, setPicked] = useState<string[]>([]);
  const [prompt, setPrompt] = useState("The ");

  const toggle = (id: string) => setPicked((s) => (s.includes(id) ? s.filter((x) => x !== id) : s.length >= 4 ? s : [...s, id]));

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
      <input className="hp-in mono cmp-prompt" value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="prompt" aria-label="compare prompt" />
      {others.length > 0 ? (
        <div className="cmp-picker">
          {others.map((p) => {
            const on = picked.includes(p.id);
            return (
              <button key={p.id} type="button" className={`chip${on ? " on" : ""}`} aria-pressed={on} disabled={!on && picked.length >= 4} onClick={() => toggle(p.id)}>
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

function HP({ label, def }: { label: string; def: string }) {
  return (
    <label className="hp">
      <span className="hp-k">{label}</span>
      <input className="hp-in mono" defaultValue={def} />
    </label>
  );
}

function HPField({ label, value, onChange, disabled }: { label: string; value: string; onChange: (v: string) => void; disabled?: boolean }) {
  return (
    <label className="hp">
      <span className="hp-k">{label}</span>
      <input className="hp-in mono" value={value} disabled={disabled} onChange={(e) => onChange(e.target.value)} />
    </label>
  );
}

// aggregate loss curve: one colored line per model, auto-scaled to the observed loss range
function LossGraph({ series, maxSteps, tall }: { series: { color: string; curve: { step: number; loss: number }[] }[]; maxSteps: number; tall?: boolean }) {
  const lines = useMemo(() => {
    const all = series.flatMap((s) => s.curve.map((p) => p.loss));
    if (all.length === 0) return [] as { color: string; pts: string }[];
    const lo = Math.min(...all);
    const hi = Math.max(...all);
    const range = hi - lo || 1;
    const lastStep = Math.max(1, ...series.flatMap((s) => s.curve.map((p) => p.step)));
    const xMax = maxSteps || lastStep;
    const x = (s: number) => (s / xMax) * 100;
    const y = (v: number) => 39 - ((v - lo) / range) * 37;
    return series
      .map((s) => ({ color: s.color, pts: s.curve.map((p) => `${x(p.step).toFixed(2)},${y(p.loss).toFixed(2)}`).join(" ") }))
      .filter((l) => l.pts);
  }, [series, maxSteps]);

  return (
    <svg className="loss-graph" viewBox="0 0 100 40" preserveAspectRatio="none" style={{ height: tall ? 180 : 92 }} aria-hidden="true">
      {lines.length ? (
        lines.map((l, i) => (
          // biome index key: series order is stable within a run
          <polyline key={i} points={l.pts} fill="none" stroke={l.color} strokeWidth="1" vectorEffect="non-scaling-stroke" />
        ))
      ) : (
        <line x1="0" y1="20" x2="100" y2="20" stroke="var(--line2)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
      )}
    </svg>
  );
}

function BenchBar({
  label,
  pct,
  sub,
  labelCh,
  onClick,
}: {
  label: string;
  pct: number;
  sub?: string;
  labelCh?: number;
  onClick?: () => void; // set = clickable: flash-highlights briefly and focuses the node on the canvas
}) {
  // transient highlight on click (not a persistent selection); clears itself after the flash
  const [flash, setFlash] = useState(false);
  const click = () => {
    setFlash(true);
    setTimeout(() => setFlash(false), 450);
    onClick?.();
  };
  const cls = `bench-bar${onClick ? " clickable" : ""}${flash ? " flash" : ""}`;
  return (
    <div
      className={cls}
      {...(onClick ? { role: "button", tabIndex: 0, onClick: click } : {})}
    >
      <span className="bb-label mono" style={labelCh ? { flexBasis: `${labelCh}ch` } : undefined}>
        {label}
      </span>
      <span className="bb-track">
        <span className="bb-fill" style={{ width: `${Math.min(100, pct)}%` }} />
      </span>
      {sub && <span className="bb-sub mono">{sub}</span>}
    </div>
  );
}
