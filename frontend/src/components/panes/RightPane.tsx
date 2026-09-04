// Right pane: mode-aware controls. Train -> Config + Train (loss graph, stats, hyperparams);
// Inference -> Generate (prompt -> output). A Benchmark toggle reveals a benchmark section whose
// open-state is decoupled per mode. Collapsible + full-view expandable via the shared PaneShell;
// `expanded` lives here so the sections can widen their layout in full view. Content is a
// placeholder for now (no backend wiring), but the structure is built to support it.
import { useEffect, useState } from "react";
import { type CanvasMode, N_LAYER_MAX, N_LAYER_MIN, useCanvasStore } from "@/store/canvasStore";
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
        <BenchmarkSection open={bench[mode]} onToggle={(v) => setBench((b) => ({ ...b, [mode]: v }))} />
      </div>
    </PaneShell>
  );
}

// --- sections ---

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

function TrainSection({ expanded }: { expanded: boolean }) {
  return (
    <section className="grp">
      <h3>Train</h3>
      <div className="pan-body">
        <LossGraph tall={expanded} />
        <div className="stat-row">
          <Stat label="step" value="—" />
          <Stat label="loss" value="—" />
          <Stat label="tok/s" value="—" />
        </div>
        <div className="hp-row">
          <HP label="steps" def="2000" />
          <HP label="batch" def="16" />
          <HP label="lr" def="3e-4" />
        </div>
        <button type="button" className="btn primary">
          Train
        </button>
      </div>
    </section>
  );
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

function BenchmarkSection({ open, onToggle }: { open: boolean; onToggle: (v: boolean) => void }) {
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
          <div className="stat-row">
            <Stat label="throughput" value="— tok/s" />
            <Stat label="latency" value="— ms/tok" />
          </div>
          <div className="bench-nodes">
            <BenchBar label="flash_attn" pct={72} />
            <BenchBar label="mlp·w8" pct={48} />
            <BenchBar label="lm_head" pct={31} />
          </div>
        </div>
      )}
    </section>
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

function HP({ label, def }: { label: string; def: string }) {
  return (
    <label className="hp">
      <span className="hp-k">{label}</span>
      <input className="hp-in mono" defaultValue={def} />
    </label>
  );
}

function LossGraph({ tall }: { tall?: boolean }) {
  return (
    <svg
      className="loss-graph"
      viewBox="0 0 100 40"
      preserveAspectRatio="none"
      style={{ height: tall ? 180 : 92 }}
      aria-hidden="true"
    >
      <polyline
        points="0,5 12,11 24,17 38,22 55,27 72,30 88,33 100,34"
        fill="none"
        stroke="var(--ice)"
        strokeWidth="1"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}

function BenchBar({ label, pct }: { label: string; pct: number }) {
  return (
    <div className="bench-bar">
      <span className="bb-label mono">{label}</span>
      <span className="bb-track">
        <span className="bb-fill" style={{ width: `${pct}%` }} />
      </span>
    </div>
  );
}
