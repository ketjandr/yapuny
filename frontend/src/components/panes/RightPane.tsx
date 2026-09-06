// Right pane: mode-aware controls.
//  Train -> Config + Train (aggregate loss curve, current-model stats, hyperparams, stop) + a
//    training Benchmark section: pick up to 5 compiled models, Train runs them sequentially and
//    fills a fwd/bwd/steps-s/peak-vram table + per-model info & node profile.
//  Inference -> Generate (streaming) + a Benchmark section that mirrors training: pick up to 5
//    trained models, generate the same prompt on each, and fill a prefill/ms-tok/tok-s/vram table +
//    per-model info, generated output, and decode node profile.
// The Benchmark toggle is persisted per project per mode and doubles as "benchmarking enabled";
// it's locked while a run is in flight. Wired to the worker over SSE (trainStore / benchTrainStore /
// benchStore / inferStore). Training and inference are mutually exclusive on the GPU - a universal
// /worker/activity signal (workerStore) gates every project's Train / Generate accordingly.
import { type Dispatch, type SetStateAction, useEffect, useMemo, useRef, useState } from "react";
import { CorpusButton } from "@/components/CorpusModal";
import { HelpDot } from "@/components/HelpDot";
import { useTooltip } from "@/components/tooltipContext";
import { api } from "@/lib/api";
import { graphForProject, graphToCanvas } from "@/lib/graph";
import { structuralIds } from "@/lib/structuralId";
import type { GraphRequest } from "@/lib/types";
import { type BenchColumn, type NodeProfile, useBenchStore } from "@/store/benchStore";
import { type BenchModel, useBenchTrainStore } from "@/store/benchTrainStore";
import {
  BATCH_MAX, BATCH_MIN, N_LAYER_MAX, N_LAYER_MIN, STEPS_MAX, STEPS_MIN,
  TEMP_MAX, TEMP_MIN, TOKENS_MAX, TOKENS_MIN, TOPK_MAX, TOPK_MIN, useCanvasStore,
} from "@/store/canvasStore";
import { useCompileStore } from "@/store/compileStore";
import { useInferStore } from "@/store/inferStore";
import { toast } from "@/store/toastStore";
import { useProjectsStore } from "@/store/projectsStore";
import { useTrainStore } from "@/store/trainStore";
import { type Activity, blockingActivity, startActivityTracking, useWorkerStore } from "@/store/workerStore";
import { PaneShell } from "./PaneShell";

// one distinct color per benchmarked model (curve line + row/chip swatch), current model first
const BENCH_COLORS = ["#e6a24d", "#8fce6a", "#77d3e6", "#d9738a", "#b892e0"];

// shared compare-set picker for both benchmarks: column 0 is the open model (fixed), the other
// selected models are removable chips (click to deselect), and remaining projects are "+ add" chips.
// The selection is transient run config - not persisted - so it clears on reload / project switch.
function ComparePicker({
  entries,
  others,
  running,
  canAdd,
  onAdd,
  onRemove,
}: {
  entries: CompareEntry[];
  others: { id: string; title: string }[]; // projects not already in the set
  running: boolean;
  canAdd: boolean;
  onAdd: (id: string) => void;
  onRemove: (id: string) => void;
}) {
  return (
    <div className="cmp-picker">
      {entries.map((e, i) => {
        const color = BENCH_COLORS[i % BENCH_COLORS.length];
        const dot = <span className="chip-dot" style={{ background: color }} />;
        // the open model (column 0) is fixed; while a run is in flight nothing can change
        if (i === 0 || running) {
          return (
            <span key={e.id} className="chip on" style={{ borderColor: color }}>
              {dot}
              {i === 0 ? `${e.title} (this)` : e.title}
            </span>
          );
        }
        return (
          <button
            key={e.id}
            type="button"
            className="chip on rm"
            style={{ borderColor: color }}
            onClick={() => onRemove(e.id)}
            title="Remove from benchmark"
          >
            {dot}
            {e.title}
            <span className="chip-x">×</span>
          </button>
        );
      })}
      {!running &&
        others.map((p) => (
          <button key={p.id} type="button" className="chip" disabled={!canAdd} onClick={() => onAdd(p.id)}>
            + {p.title}
          </button>
        ))}
    </div>
  );
}

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

  // track the universal GPU-busy signal (initial fetch + refresh on tab focus); also refresh on a
  // project switch so the newly-opened project reflects a run another project may be holding
  useEffect(() => startActivityTracking(), []);
  useEffect(() => {
    useWorkerStore.getState().refresh();
  }, [modelId]);

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
    <PaneShell
      side="right"
      title={title}
      expanded={expanded}
      onToggleExpand={() => setExpanded((v) => !v)}
      tools={mode === "train" ? <CorpusButton /> : undefined}
    >
      <div className="pane-body">
        {mode === "train" ? (
          <TrainControls expanded={expanded} open={benchOpen.train} onToggle={(v) => setBenchOpen("train", v)} />
        ) : (
          <InferControls expanded={expanded} open={benchOpen.inference} onToggle={(v) => setBenchOpen("inference", v)} />
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
  const singleStatus = useTrainStore((s) => s.status);

  const titleOf = (id: string) => projects.find((p) => p.id === id)?.title ?? id;

  // a run finishing saves weights, but CanvasStatus only re-polls model_status on graph edits, so the
  // compiled/trained indicators (and the Generate gate) would stay "untrained" until a reload. When a
  // run reaches a terminal state, re-poll the open model's status to refresh compileStore.
  useEffect(() => {
    if (singleStatus !== "completed" && runStatus !== "completed") return;
    let cancelled = false;
    api
      .modelStatus({ id: modelId, graph: toGraph() })
      .then((r) => {
        if (cancelled) return;
        const ready = r.status === "ready";
        useCompileStore.getState().setResult(ready ? "ready" : "needs_compile", ready && !!r.trained);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [singleStatus, runStatus, modelId, toGraph]);

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

  const infos = useCompareInfos(entries, `${runStatus}|${singleStatus}`);

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
        <CfgSlider label="n_layer" help="Transformer blocks stacked in a row, each refining the model's understanding. How many times the canvas block repeats." value={meta.n_layer} min={N_LAYER_MIN} max={N_LAYER_MAX} step={1} onChange={(v) => setMeta({ n_layer: v })} />
        <CfgSlider label="n_head" help="Attention heads per block. Each spots a different relationship between tokens - grammar, meaning, position." value={meta.n_head} min={1} max={16} step={1} onChange={(v) => setMeta({ n_head: v })} />
        <CfgSlider
          label="n_embd"
          help="How much information each token carries through the model - wider holds more nuance."
          hint={`head_dim ${Math.floor(meta.n_embd / meta.n_head)}`}
          value={meta.n_embd}
          min={meta.n_head}
          max={1024}
          step={meta.n_head}
          snap={snapEmbd}
          onChange={(v) => setMeta({ n_embd: v })}
        />
        <CfgSlider label="block_size" help="Context window - how many tokens back the model can look when predicting the next. Longer sees more but trains slower." value={meta.block_size} min={16} max={1024} step={16} onChange={(v) => setMeta({ block_size: v })} />
        <CfgSlider label="dropout" help="Randomly ignores part of the network each step so it can't just memorize the corpus - pushing it to generalize." value={meta.dropout} min={0} max={0.5} step={0.05} float onChange={(v) => setMeta({ dropout: v })} />
        <CfgSlider label="vocab_size" help="How many distinct tokens (word-pieces) the model knows, learned from your corpus. More captures rarer words but costs more." value={meta.vocab_size} min={256} max={50000} step={256} onChange={(v) => setMeta({ vocab_size: v })} />
      </div>
    </section>
  );
}

function CfgSlider({
  label,
  hint,
  help,
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
  help?: string; // plain-language explanation shown from the pressable "?" popover
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
          {help && <HelpDot label={label} text={help} />}
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

  // hyperparameters are per-project (persisted in the canvas store), not ephemeral component state
  const train = useCanvasStore((s) => s.train);
  const setTrainHp = useCanvasStore((s) => s.setTrainHp);

  // (reconnect to an in-progress run + selection restore is handled once in RightPane)

  const titleOf = (id: string) => projects.find((p) => p.id === id)?.title ?? id;

  // run-in-progress detection. A bench run is "owned" by the project that started it (column 0);
  // only the owner shows the run - every other project greys out (the worker is busy), even models
  // that happen to be in the compare set, since they can't start their own run.
  const singleRunning = single.status === "running" || single.status === "stopping";
  const benchRunning = bench.status === "running" || bench.status === "stopping";
  const benchOwner = bench.models[0]?.id === modelId;
  const singleMine = single.modelId === modelId;
  const running = benchOn ? benchRunning && benchOwner : singleRunning && singleMine;
  // the worker runs one job at a time: anything not our own training run blocks Train (another
  // project's training, or ANY inference - they all occupy the GPU)
  const activity = useWorkerStore((s) => s.activity);
  const blocked = blockingActivity(activity, modelId, ["train", "train_bench"]);
  const otherBusy = !!blocked;

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
    max_steps: train.maxSteps,
    batch_size: train.batchSize,
    learning_rate: train.learningRate,
  });

  const onTrain = async () => {
    if (running) {
      (benchOn ? bench.stop : single.stop)();
      return;
    }
    // corpus pre-flight: the worker only errors once the run is spawned, which briefly flips the
    // button to "Stop" and back (a flicker). Check up front and bail with a toast, no state change.
    try {
      const ds = await api.dataStatus();
      if (!ds?.corpus_uploaded) {
        toast.error("No corpus - upload one before training");
        return;
      }
    } catch {
      /* worker unreachable: fall through and let start() surface the failure */
    }
    if (benchOn) bench.start(entries.map((e) => e.id), hp());
    else single.start({ id: modelId, ...hp(), bench: false });
  };

  const gate = blocked
    ? busyLabel(blocked, titleOf(blocked.modelId))
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
          <HpNumField label="steps" value={train.maxSteps} min={STEPS_MIN} max={STEPS_MAX} onCommit={(v) => setTrainHp({ maxSteps: v })} disabled={running} />
          <HpNumField label="batch" value={train.batchSize} min={BATCH_MIN} max={BATCH_MAX} onCommit={(v) => setTrainHp({ batchSize: v })} disabled={running} />
          <LrField value={train.learningRate} onChange={(v) => setTrainHp({ learningRate: v })} disabled={running} />
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

// tooltip for a control blocked because the GPU is busy elsewhere (another project or the other run)
function busyLabel(a: Activity, title: string): string {
  const what = a.kind === "generate" || a.kind === "gen_bench" ? "Generating" : "Training";
  return `${what} "${title}" is in progress`;
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
  // a plain training run for this project also locks the toggle - can't switch modes mid-run
  const single = useTrainStore();
  const singleRunning = (single.status === "running" || single.status === "stopping") && single.modelId === modelId;
  const locked = running || singleRunning;

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
          disabled={locked}
          onClick={() => onToggle(!open)}
        >
          <span className="sw-knob" />
        </button>
      </h3>
      {open && (
        <div className="pan-body">
          {/* model picker: the open model is always column 0; add up to 4 others (5 total) */}
          <ComparePicker
            entries={entries}
            others={others.filter((p) => !selected.includes(p.id))}
            running={running}
            canAdd={selected.length < 4}
            onAdd={toggleSelected}
            onRemove={toggleSelected}
          />

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

// the profile has one entry per unrolled node (l0_rope, l1_rope, ...); collapse them onto their
// logical id so the breakdown reads per node type (rope, qkv_proj, ...) with each type's total share
function aggregateProfile(nodes: NodeProfile[]): NodeProfile[] {
  const by = new Map<string, NodeProfile>();
  for (const n of nodes) {
    const g = by.get(n.logical_id);
    if (g) {
      g.self_us += n.self_us;
      g.pct += n.pct;
    } else {
      by.set(n.logical_id, { ...n });
    }
  }
  return [...by.values()].sort((a, b) => b.self_us - a.self_us);
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
  const nodes = useMemo(() => aggregateProfile(model?.bench?.profile.nodes ?? []), [model]);
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
                sub={`${n.pct.toFixed(1)}%`}
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

// --- inference: streaming generate + a multi-model benchmark (mirrors the training benchmark) ---

// sampling controls shared by Generate + the inference benchmark (persisted per project in `gen`)
function GenParams({ disabled }: { disabled: boolean }) {
  const gen = useCanvasStore((s) => s.gen);
  const setGenHp = useCanvasStore((s) => s.setGenHp);
  return (
    <div className="hp-row">
      <HpNumField label="temp" value={gen.temperature} min={TEMP_MIN} max={TEMP_MAX} float onCommit={(v) => setGenHp({ temperature: v })} disabled={disabled} />
      <HpNumField label="top_k" value={gen.topK} min={TOPK_MIN} max={TOPK_MAX} onCommit={(v) => setGenHp({ topK: v })} disabled={disabled} />
      <HpNumField label="tokens" value={gen.maxTokens} min={TOKENS_MIN} max={TOKENS_MAX} onCommit={(v) => setGenHp({ maxTokens: v })} disabled={disabled} />
    </div>
  );
}

// Inference controls: Generate (the canonical prompt + streamed output + live stats) plus a
// Benchmark section (per-model table + config/profile detail). Mirrors TrainControls - the parent
// owns the compare set (this model + up to 4 picked others), and the benchmark toggle doubles as
// "benchmark on": when on, the Generate button becomes "Generate & benchmark" and runs the compare.
function InferControls({ expanded, open, onToggle }: { expanded: boolean; open: boolean; onToggle: (v: boolean) => void }) {
  const modelId = useCanvasStore((s) => s.modelId);
  const toGraph = useCanvasStore((s) => s.toGraph);
  const nodes = useCanvasStore((s) => s.nodes);
  const edges = useCanvasStore((s) => s.edges);
  const meta = useCanvasStore((s) => s.meta);
  const projects = useProjectsStore((s) => s.projects);
  const running = useBenchStore((s) => s.comparing && s.compareOwner === modelId);
  const titleOf = (id: string) => projects.find((p) => p.id === id)?.title ?? id;

  const [picked, setPicked] = useState<string[]>([]);

  // reattach the inference benchmark after a reload: the worker keeps the last compare's results,
  // so rehydrate the columns AND restore the compare selection (the other benchmarked models) from
  // them, so every model comes back - not just the open one. Isolated to the owner, like training.
  useEffect(() => {
    let cancelled = false;
    useBenchStore
      .getState()
      .follow()
      .then(() => {
        if (cancelled) return;
        const bs = useBenchStore.getState();
        if (bs.compareOwner === modelId && bs.columns.length > 1) {
          setPicked(bs.columns.slice(1).map((c) => c.id));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [modelId]);

  const entries = useMemo<CompareEntry[]>(() => {
    const cur: CompareEntry = { id: modelId, graph: toGraph(), title: titleOf(modelId) };
    const rest = picked
      .map((id) => {
        const g = graphForProject(id);
        return g ? { id, graph: g, title: titleOf(id) } : null;
      })
      .filter((e): e is CompareEntry => e !== null);
    return [cur, ...rest];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelId, picked, projects, nodes, edges, meta]);

  const infos = useCompareInfos(entries, running ? "run" : "idle");

  const [detailId, setDetailId] = useState<string | null>(null);
  const detail = detailId ?? entries[0]?.id;

  return (
    <>
      <GenerateSection
        expanded={expanded}
        benchOn={open}
        entries={entries}
        infos={infos}
        detail={detail}
        onBenchStart={() => setDetailId(entries[0]?.id ?? null)}
      />
      <InferenceBenchmark
        expanded={expanded}
        open={open}
        onToggle={onToggle}
        entries={entries}
        infos={infos}
        picked={picked}
        setPicked={setPicked}
        detail={detail}
        onSelect={setDetailId}
        running={running}
      />
    </>
  );
}

// Generate: the canonical prompt, sampling params, run button, streamed output, and a live stat row
// (tok/s, tokens, ttft) - the inference analog of the training Train section. With the benchmark
// toggle on, the button runs the compare instead and the output/stats follow the model currently
// generating, exactly like the training section tracks the current model during a bench run.
function GenerateSection({
  expanded,
  benchOn,
  entries,
  infos,
  detail,
  onBenchStart,
}: {
  expanded: boolean;
  benchOn: boolean;
  entries: CompareEntry[];
  infos: Record<string, CompareInfo>;
  detail?: string;
  onBenchStart: () => void;
}) {
  const modelId = useCanvasStore((s) => s.modelId);
  const projects = useProjectsStore((s) => s.projects);
  const gen = useCanvasStore((s) => s.gen);
  const setGenHp = useCanvasStore((s) => s.setGenHp);
  const compiled = useCompileStore((s) => s.status === "ready");
  const trained = useCompileStore((s) => s.trained);
  const infer = useInferStore();
  const bench = useBenchStore();
  const activity = useWorkerStore((s) => s.activity);
  const blocked = blockingActivity(activity, modelId, benchOn ? ["gen_bench"] : ["generate"]);
  const titleOf = (id: string) => projects.find((p) => p.id === id)?.title ?? id;

  // run-in-progress: a compare owned by this project (benchOn) or this project's single generate
  const single = infer.modelId === modelId;
  const running = benchOn ? bench.comparing && bench.compareOwner === modelId : infer.status === "running" && single;

  // gating: bench needs every selected model compiled+trained; single needs the open model ready
  const ready = (id: string) => (id === modelId ? compiled && trained : Boolean(infos[id]?.ready && infos[id]?.trained));
  const allReady = entries.every((e) => ready(e.id));
  const canStart = !blocked && (benchOn ? allReady : compiled && trained);
  const gate = blocked
    ? busyLabel(blocked, titleOf(blocked.modelId))
    : benchOn && !allReady
      ? "Compile and train every selected model first"
      : !compiled
        ? "Compile the model before generating"
        : !trained
          ? "Train the model before generating"
          : "";
  const tip = useTooltip(gate);

  const opts = () => ({ prompt: gen.prompt, max_new_tokens: gen.maxTokens, temperature: gen.temperature, top_k: gen.topK });

  const onGenerate = () => {
    if (running) {
      if (benchOn) bench.stopCompare();
      else infer.stop();
      return;
    }
    if (benchOn) {
      onBenchStart();
      bench.runCompare(entries.map((e) => ({ id: e.id, graph: e.graph, label: e.title })), opts());
    } else {
      infer.start({ id: modelId, ...opts() });
    }
  };

  // output + live stats. Single: from inferStore. Bench: from the selected model's column - its
  // streamed text is the canonical output and its live counters feed the stat row, so selecting a
  // row in the table swaps the output/stats to that model (its counters go live while it generates).
  const owned = bench.compareOwner === modelId;
  const cur = benchOn && owned ? (bench.columns.find((c) => c.id === detail) ?? bench.columns[0] ?? null) : null;
  const errored = benchOn ? !!cur?.error : single && infer.status === "error";
  const errMsg = benchOn ? cur?.error : infer.error;
  const output = benchOn ? (cur?.text ?? "") : single ? infer.text : "";
  const tps = benchOn ? (cur?.tokensPerSec ?? null) : single ? infer.tokensPerSec : null;
  const tokens = benchOn ? (cur?.genTokens ?? 0) : single ? infer.tokens : 0;
  const prefill = benchOn ? (cur?.prefillMs ?? null) : single ? infer.prefillMs : null;
  const activeName = benchOn && cur ? titleOf(cur.id) : "";
  const activeColor = benchOn && cur ? BENCH_COLORS[Math.max(0, entries.findIndex((e) => e.id === cur.id)) % BENCH_COLORS.length] : "var(--ice)";

  return (
    <section className="grp">
      <h3>
        Generate
        {activeName && (
          <span className="grp-sub mono" style={{ color: activeColor }}>
            {activeName}
          </span>
        )}
      </h3>
      <div className="pan-body">
        <textarea
          className={`gen-prompt${expanded ? " tall" : ""}`}
          placeholder="Enter a prompt…"
          value={gen.prompt}
          disabled={running}
          onChange={(e) => setGenHp({ prompt: e.target.value })}
        />
        <GenParams disabled={running} />
        <span className="btn-wrap" {...tip}>
          <button type="button" className={`btn ${running ? "danger" : "primary"}`} disabled={!running && !canStart} onClick={onGenerate}>
            {running ? "Stop" : benchOn ? "Generate & benchmark" : "Generate"}
          </button>
        </span>
        {/* fixed-height output so streaming never pushes the stat row / benchmark down the pane */}
        <div className={`gen-output mono${expanded ? " tall" : ""}${errored ? " bad" : ""}`}>
          {errored ? errMsg : output || (running ? "…" : "Output will stream here…")}
        </div>
        <div className="stat-row">
          <Stat label="tok/s" value={tps != null ? tps.toFixed(1) : "—"} />
          <Stat label="tokens" value={running || tokens > 0 ? `${tokens}/${gen.maxTokens}` : "—"} />
          <Stat label="prefill" value={prefill != null ? `${prefill.toFixed(0)} ms` : "—"} />
        </div>
      </div>
    </section>
  );
}

// inference benchmark: the toggle, the model picker, the throughput/latency/vram table, and a
// per-model config + node-profile detail. The prompt and generated output are NOT duplicated here -
// they're canonical in the Generate section above; a run is started by its "Generate & benchmark".
function InferenceBenchmark({
  expanded,
  open,
  onToggle,
  entries,
  infos,
  picked,
  setPicked,
  detail,
  onSelect,
  running,
}: {
  expanded: boolean;
  open: boolean;
  onToggle: (v: boolean) => void;
  entries: CompareEntry[];
  infos: Record<string, CompareInfo>;
  picked: string[];
  setPicked: Dispatch<SetStateAction<string[]>>;
  detail?: string;
  onSelect: (id: string) => void;
  running: boolean;
}) {
  const modelId = useCanvasStore((s) => s.modelId);
  const projects = useProjectsStore((s) => s.projects);
  const bench = useBenchStore();
  // results isolate to the project that started the run (column 0), like the training benchmark
  const owned = bench.compareOwner === modelId;
  const columns = owned ? bench.columns : [];
  const others = projects.filter((p) => p.id !== modelId);
  const toggle = (id: string) => setPicked((s) => (s.includes(id) ? s.filter((x) => x !== id) : s.length >= 4 ? s : [...s, id]));
  // a plain generate for this project also locks the toggle - can't switch modes mid-run
  const genRunning = useInferStore((s) => s.status === "running" && s.modelId === modelId);
  const locked = running || genRunning;

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
          disabled={locked}
          onClick={() => onToggle(!open)}
        >
          <span className="sw-knob" />
        </button>
      </h3>
      {open && (
        <div className="pan-body">
          {/* model picker: the open model is always column 0; add up to 4 others (5 total) */}
          <ComparePicker
            entries={entries}
            others={others.filter((p) => !picked.includes(p.id))}
            running={running}
            canAdd={picked.length < 4}
            onAdd={toggle}
            onRemove={toggle}
          />

          <InferBenchTable entries={entries} columns={columns} detail={detail} onSelect={onSelect} expanded={expanded} />

          <div className="bench-detail">
            <InferDetail
              info={infos[detail ?? ""]?.info ?? null}
              column={columns.find((c) => c.id === detail) ?? null}
              graph={entries.find((e) => e.id === detail)?.graph}
              expanded={expanded}
            />
          </div>
        </div>
      )}
    </section>
  );
}

const fmtVram = (mb: number | null | undefined) =>
  mb == null ? "—" : mb < 1024 ? `${mb.toFixed(0)} MB` : `${(mb / 1024).toFixed(2)} GB`;

// one row per model: prefill ms, ms/tok, tok/s (with a throughput bar), peak vram. Rows are the
// selector for the detail panel below - click one to inspect that model's output + node profile.
function InferBenchTable({
  entries,
  columns,
  detail,
  onSelect,
  expanded,
}: {
  entries: CompareEntry[];
  columns: BenchColumn[];
  detail?: string;
  onSelect: (id: string) => void;
  expanded: boolean;
}) {
  const byId = new Map(columns.map((c) => [c.id, c]));
  const maxTps = Math.max(1, ...columns.map((c) => c.tokensPerSec ?? 0));
  const cell = (v: number | null | undefined, d: number) => (v != null ? v.toFixed(d) : "—");

  return (
    <div className="bt-wrap">
      <table className="bt-table mono">
        <thead>
          <tr>
            <th className="bt-name">model</th>
            <th>prefill</th>
            <th>ms/tok</th>
            <th className="bt-tp">tok / s</th>
            <th>peak vram</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((e, i) => {
            const c = byId.get(e.id);
            const color = BENCH_COLORS[i % BENCH_COLORS.length];
            const pending = !c || (!c.done && !c.running);
            return (
              <tr key={e.id} className={`bt-row${detail === e.id ? " sel" : ""}`} onClick={() => onSelect(e.id)}>
                <td className="bt-name">
                  <span className="bt-bar" style={{ background: color }} />
                  {e.title}
                </td>
                <td>{c?.error ? "err" : cell(c?.prefillMs, 0)}</td>
                <td>{c?.error ? "err" : cell(c?.msPerToken, 1)}</td>
                <td className="bt-tp">
                  {c?.tokensPerSec != null ? (
                    <span className="bt-tp-wrap">
                      <span className="bt-tp-num">{c.tokensPerSec.toFixed(1)}</span>
                      {expanded && (
                        <span className="bb-track">
                          <span className="bb-fill" style={{ width: `${(c.tokensPerSec / maxTps) * 100}%`, background: color }} />
                        </span>
                      )}
                    </span>
                  ) : c?.running ? (
                    <span className="bt-live">generating…</span>
                  ) : (
                    <span className="bench-empty">{pending ? "pending" : "—"}</span>
                  )}
                </td>
                <td>{fmtVram(c?.peakVramMb)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// detail for the selected model: config + decode profile (the output lives in the Generate section)
function InferDetail({
  info,
  column,
  graph,
  expanded,
}: {
  info: ModelInfo | null;
  column: BenchColumn | null;
  graph?: GraphRequest;
  expanded: boolean;
}) {
  const modelId = useCanvasStore((s) => s.modelId);
  const requestFocusNode = useCanvasStore((s) => s.requestFocusNode);
  const isOpenModel = column?.id === modelId;
  const profNodes = useMemo(() => aggregateProfile(column?.profile ?? []), [column]);
  const top = expanded ? profNodes : profNodes.slice(0, 8);
  const more = profNodes.length - top.length;
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

      {top.length > 0 && (
        <div className="bench-nodes">
          <span className="bench-sub">Node profile (decode step)</span>
          {(() => {
            const labels = top.map((n) => names.get(n.logical_id) ?? n.logical_id);
            const labelCh = Math.min(18, Math.max(...labels.map((l) => l.length)) + 1);
            return top.map((n, i) => (
              <BenchBar
                key={n.node_id}
                label={labels[i]}
                pct={n.pct}
                sub={`${n.pct.toFixed(1)}%`}
                labelCh={labelCh}
                onClick={expanded || !isOpenModel ? undefined : () => requestFocusNode(n.logical_id)}
              />
            ));
          })()}
          {more > 0 && <span className="bench-more">+{more} nodes…</span>}
        </div>
      )}
    </div>
  );
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

// labeled number field for the hp-row: free typing, clamp to [min, max] on blur/Enter. Integers by
// default; `float` keeps up to 2 decimals (e.g. temperature).
function HpNumField({
  label,
  value,
  min,
  max,
  float,
  onCommit,
  disabled,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  float?: boolean;
  onCommit: (v: number) => void;
  disabled?: boolean;
}) {
  const [text, setText] = useState(String(value));
  useEffect(() => setText(String(value)), [value]);
  const commit = () => {
    const n = Number(text);
    if (!Number.isFinite(n)) {
      setText(String(value));
      return;
    }
    let clamped = Math.min(max, Math.max(min, float ? n : Math.round(n)));
    if (float) clamped = Math.round(clamped * 100) / 100;
    onCommit(clamped);
    setText(String(clamped));
  };
  return (
    <label className="hp">
      <span className="hp-k">{label}</span>
      <input
        className="hp-in mono"
        type="text"
        inputMode={float ? "decimal" : "numeric"}
        value={text}
        disabled={disabled}
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => { if (e.key === "Enter") e.currentTarget.blur(); }}
      />
    </label>
  );
}

// learning rate as a preset dropdown - avoids typing "3e-4" and keeps the value sensible
const LR_PRESETS = [1e-2, 5e-3, 3e-3, 1e-3, 5e-4, 3e-4, 1e-4, 5e-5, 3e-5, 1e-5];
const fmtLr = (v: number) => v.toExponential(0); // e.g. 3e-4

// custom (themed) dropdown - the native <select> popup can't be styled to match the dark UI
function LrField({ value, onChange, disabled }: { value: number; onChange: (v: number) => void; disabled?: boolean }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  // include the current value if it isn't one of the presets (e.g. an older persisted number)
  const options = LR_PRESETS.includes(value) ? LR_PRESETS : [value, ...LR_PRESETS].sort((a, b) => b - a);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  return (
    <label className="hp">
      <span className="hp-k">lr</span>
      <div className={`lr-sel${open ? " open" : ""}`} ref={ref}>
        <button
          type="button"
          className="hp-in lr-btn mono"
          disabled={disabled}
          aria-haspopup="listbox"
          aria-expanded={open}
          onClick={() => setOpen((o) => !o)}
        >
          {fmtLr(value)}
        </button>
        {open && (
          <ul className="lr-menu mono" role="listbox">
            {options.map((v) => (
              <li key={v}>
                <button
                  type="button"
                  role="option"
                  aria-selected={v === value}
                  className={`lr-opt${v === value ? " sel" : ""}`}
                  onClick={() => { onChange(v); setOpen(false); }}
                >
                  {fmtLr(v)}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
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
