// Navbar: brand (returns to the Models home) + active project title (inline-editable) + the worker
// connection indicator / Connect popover.
import { useEffect, useRef, useState } from "react";
import { BrandButton } from "@/components/BrandButton";
import { TITLE_MAX } from "@/lib/projects";
import { useCanvasStore } from "@/store/canvasStore";
import { useConnStore } from "@/store/connStore";
import { useProjectsStore } from "@/store/projectsStore";

export function Navbar() {
  const modelId = useCanvasStore((s) => s.modelId);
  const title = useProjectsStore(
    (s) => s.projects.find((p) => p.id === modelId)?.title ?? "untitled",
  );
  const rename = useProjectsStore((s) => s.rename);

  return (
    <header className="nav">
      <div className="brand">
        <BrandButton />
        <span className="proj">
          model{" "}
          {/* uncontrolled + keyed on modelId so it resets on project switch; commit on blur/Enter */}
          <input
            key={modelId}
            className="proj-title"
            defaultValue={title}
            size={Math.max(title.length, 6)}
            maxLength={TITLE_MAX}
            spellCheck={false}
            aria-label="Model title"
            onBlur={(e) => rename(modelId, e.target.value.trim() || "Untitled model")}
            onKeyDown={(e) => {
              if (e.key === "Enter") e.currentTarget.blur();
              if (e.key === "Escape") {
                e.currentTarget.value = title;
                e.currentTarget.blur();
              }
            }}
          />
        </span>
      </div>
      <div className="nav-r">
        <WorkerConnect />
      </div>
    </header>
  );
}

function WorkerConnect() {
  const status = useConnStore((s) => s.status);
  const mode = useConnStore((s) => s.mode);
  const info = useConnStore((s) => s.info);
  const check = useConnStore((s) => s.check);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // verify the persisted target (or the dev proxy) once on load so the indicator reflects reality
  useEffect(() => {
    check();
  }, [check]);

  // close the popover on outside click / Escape
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    // capture phase: the canvas (React Flow) stops mousedown propagation, so a bubble-phase listener
    // would miss clicks there - capture fires before any child can stop it
    document.addEventListener("mousedown", onDown, true);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown, true);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const dot = status === "online" ? "" : status === "connecting" ? " warn" : " off";
  const label =
    status === "online"
      ? mode === "shared"
        ? "worker: shared"
        : `worker: ${info?.gpu || info?.device || "online"}`
      : status === "connecting"
        ? "connecting…"
        : "worker: offline";

  return (
    <div className="wconn" ref={ref}>
      <div className="tele" title={label}>
        <span className={`d${dot}`} />
        <span className="tele-label">{label}</span>
      </div>
      <button className="nl" type="button" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        {status === "online" ? "manage" : "connect"}
      </button>
      {open && <ConnectPanel />}
    </div>
  );
}

function ConnectPanel() {
  const { mode, url, token, status, error, info, hasShared, connectCustom, connectShared, disconnect } =
    useConnStore();
  const connected = status === "online";

  // connected: read-only view of the active target + Disconnect (no editable fields, no Connect)
  if (connected) {
    return (
      <div className="wpanel">
        <div className="wpanel-title">{mode === "shared" ? "Shared worker" : "Your worker"}</div>
        {mode === "shared" ? (
          <p className="wpanel-note">
            Free community worker, no setup. Runs on CPU and is <strong>session-only</strong>: your
            graphs are saved in this browser, but trained weights aren’t kept. Connect your own
            worker for GPU speed and persistence.
          </p>
        ) : (
          <>
            <label className="wpanel-field">
              <span>Worker URL</span>
              <input type="text" value={url || "local (dev proxy)"} disabled />
            </label>
            <label className="wpanel-field">
              <span>Token</span>
              <input type="password" value={token} placeholder="none" disabled />
            </label>
          </>
        )}
        <div className="wpanel-status">
          <span className="wpanel-ok">✓ connected{info?.device ? ` · ${info.gpu || info.device}` : ""}</span>
        </div>
        <div className="wpanel-row">
          <button type="button" className="btn" onClick={disconnect}>
            Disconnect
          </button>
        </div>
      </div>
    );
  }

  // disconnected: choose your own worker or opt into the shared worker
  return <ConnectChooser {...{ url, token, status, error, hasShared, connectCustom, connectShared }} />;
}

function ConnectChooser({
  url,
  token,
  status,
  error,
  hasShared,
  connectCustom,
  connectShared,
}: {
  url: string;
  token: string;
  status: string;
  error: string | null;
  hasShared: boolean;
  connectCustom: (url: string, token: string) => Promise<boolean>;
  connectShared: () => Promise<boolean>;
}) {
  const [u, setU] = useState(url);
  const [t, setT] = useState(token);
  const [busy, setBusy] = useState(false);

  // on success the panel stays open and re-renders into the connected (Disconnect) view, since
  // ConnectPanel branches on status === "online" - so we don't close it here
  const run = async (fn: () => Promise<boolean>) => {
    setBusy(true);
    await fn();
    setBusy(false);
  };

  return (
    <div className="wpanel">
      <div className="wpanel-title">Connect a worker</div>
      <p className="wpanel-note">Point at your own worker (local or remote), or use the free shared one.</p>

      <label className="wpanel-field">
        <span>Worker URL</span>
        <input
          type="text"
          placeholder="http://localhost:8000"
          spellCheck={false}
          value={u}
          onChange={(e) => setU(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run(() => connectCustom(u, t))}
        />
      </label>
      <label className="wpanel-field">
        <span>Token (optional)</span>
        <input
          type="password"
          placeholder="only if the worker requires one"
          spellCheck={false}
          value={t}
          onChange={(e) => setT(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run(() => connectCustom(u, t))}
        />
      </label>

      <div className="wpanel-status">
        {status === "error" && error ? (
          <span className="wpanel-err">Couldn't reach worker: {error}</span>
        ) : status === "connecting" ? (
          <span className="wpanel-muted">connecting…</span>
        ) : null}
      </div>

      <div className="wpanel-row">
        <button type="button" className="btn primary" disabled={busy} onClick={() => run(() => connectCustom(u, t))}>
          {busy ? "connecting…" : "Connect"}
        </button>
      </div>

      {hasShared && (
        <>
          <div className="wpanel-or">or</div>
          <button type="button" className="wpanel-shared" disabled={busy} onClick={() => run(connectShared)}>
            Use the free shared worker
            <span className="wpanel-shared-sub">no setup · CPU · session-only</span>
          </button>
        </>
      )}
    </div>
  );
}
