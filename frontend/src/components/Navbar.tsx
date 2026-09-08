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

      <div className="wpanel-row">
        <button type="button" className="btn primary" disabled={busy} onClick={() => run(() => connectCustom(u, t))}>
          {busy ? "connecting…" : "Connect"}
        </button>
      </div>

      {/* only rendered when there's something to say, so the button sits right under the token box and
          the message just pushes things down when it appears */}
      {(status === "error" && error) || status === "connecting" ? (
        <div className="wpanel-status">
          {status === "error" && error ? (
            <span className="wpanel-err">Couldn't reach worker: {error}</span>
          ) : (
            <span className="wpanel-muted">connecting…</span>
          )}
        </div>
      ) : null}

      {/* grouped with the own-worker options above: don't have a worker yet? set one up locally */}
      <LocalWorkerHelp busy={busy} onConnectLocal={() => run(() => connectCustom(LOCAL_URL, ""))} />

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

const LOCAL_URL = "http://localhost:8000";
// one-line installers hosted in the repo; they install uv + the worker and start it (see install.sh)
const INSTALL_CMD: Record<"unix" | "win", string> = {
  unix: "curl -LsSf https://raw.githubusercontent.com/ketjandr/yapuny/main/install.sh | sh",
  win: "irm https://raw.githubusercontent.com/ketjandr/yapuny/main/install.ps1 | iex",
};

// Collapsible "run the worker on your own machine" guide: one copy-paste line (no Docker) that
// installs and starts the worker, auto-using your GPU if you have one, then a one-click connect.
function LocalWorkerHelp({ busy, onConnectLocal }: { busy: boolean; onConnectLocal: () => void }) {
  // default the OS to the visitor's, but let them switch
  const [os, setOs] = useState<"unix" | "win">(() =>
    /win/i.test(navigator.userAgent) ? "win" : "unix",
  );
  const [copied, setCopied] = useState(false);
  const cmd = INSTALL_CMD[os];

  // switching OS changes the command, so a prior "copied" no longer applies
  const pickOs = (next: "unix" | "win") => {
    setOs(next);
    setCopied(false);
  };

  // no auto-reset: the check persists until something re-renders it away (OS switch, panel reopen)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(cmd);
      setCopied(true);
    } catch {
      /* clipboard blocked - the command is selectable in the box */
    }
  };

  return (
    <details className="wpanel-local">
      <summary>Run a worker on your computer</summary>
      {/* body is a separate flex div: `gap` is unreliable on <details> itself */}
      <div className="wpanel-local-body">
        <p className="wpanel-note">
          This command installs/updates and starts the worker, using your GPU automatically if you
          have one. Paste it into a terminal:
        </p>
        <div className="wpanel-seg">
          <button type="button" className={os === "unix" ? "on" : ""} onClick={() => pickOs("unix")}>
            macOS / Linux
          </button>
          <button type="button" className={os === "win" ? "on" : ""} onClick={() => pickOs("win")}>
            Windows (PS)
          </button>
        </div>
        <div className="wpanel-cmd">
          <code>{cmd}</code>
          <button
            type="button"
            className={copied ? "copied" : ""}
            onClick={copy}
            aria-label={copied ? "copied" : "copy command"}
          >
            {copied ? <CheckIcon /> : <CopyIcon />}
          </button>
        </div>
        <p className="wpanel-note">Then, once it says it's running:</p>
        <button type="button" className="btn" disabled={busy} onClick={onConnectLocal}>
          {busy ? "connecting…" : "Connect to localhost:8000"}
        </button>
      </div>
    </details>
  );
}

const iconProps = {
  width: 13,
  height: 13,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};
function CopyIcon() {
  return (
    <svg {...iconProps} aria-hidden="true">
      <rect x="9" y="9" width="13" height="13" rx="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}
function CheckIcon() {
  return (
    <svg {...iconProps} aria-hidden="true">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}
