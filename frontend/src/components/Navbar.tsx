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
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const dot = status === "online" ? "" : status === "connecting" ? " warn" : " off";
  const label =
    status === "online"
      ? `worker: ${info?.gpu || info?.device || "online"}`
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
      {open && <ConnectPanel onClose={() => setOpen(false)} />}
    </div>
  );
}

function ConnectPanel({ onClose }: { onClose: () => void }) {
  const { url, token, status, error, connect, disconnect } = useConnStore();
  const [u, setU] = useState(url);
  const [t, setT] = useState(token);
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    setBusy(true);
    const ok = await connect(u, t);
    setBusy(false);
    if (ok) onClose();
  };

  return (
    <div className="wpanel">
      <div className="wpanel-title">Connect a worker</div>
      <p className="wpanel-note">
        Run the worker locally and paste its URL, or point at a remote one. Leave the URL blank in
        local dev to use the built-in proxy.
      </p>
      <label className="wpanel-field">
        <span>Worker URL</span>
        <input
          type="text"
          placeholder="http://localhost:8000"
          spellCheck={false}
          value={u}
          onChange={(e) => setU(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") submit();
          }}
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
          onKeyDown={(e) => {
            if (e.key === "Enter") submit();
          }}
        />
      </label>
      {/* always reserve a line so the button doesn't jump as the message appears/clears */}
      <div className="wpanel-status">
        {status === "error" && error ? (
          <span className="wpanel-err">Couldn't reach worker: {error}</span>
        ) : status === "online" ? (
          <span className="wpanel-ok">✓ connected</span>
        ) : status === "connecting" ? (
          <span className="wpanel-muted">connecting…</span>
        ) : null}
      </div>
      <div className="wpanel-row">
        <button type="button" className="btn primary" disabled={busy} onClick={submit}>
          {busy ? "connecting…" : "Connect"}
        </button>
        {status === "online" && (
          <button
            type="button"
            className="btn"
            onClick={() => {
              disconnect();
              setU("");
              setT("");
            }}
          >
            Disconnect
          </button>
        )}
      </div>
    </div>
  );
}
