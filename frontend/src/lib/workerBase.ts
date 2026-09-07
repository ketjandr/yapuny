// The configured worker target: a mode (the user's own worker vs the shared worker) + optional
// token, persisted to localStorage so lib/api can build request URLs synchronously with no React
// dependency. A stable per-browser session id is sent on shared requests so the gateway can route
// each session to its own throwaway worker process.
const MODE_KEY = "yapuny.worker.mode";
const URL_KEY = "yapuny.worker.url";
const TOKEN_KEY = "yapuny.worker.token";
const SESSION_KEY = "yapuny.session";

export type WorkerMode = "custom" | "shared" | "off"; // "off" = disconnected (url/token remembered)

// the hosted shared worker, baked in at build time (empty in dev unless you set VITE_SHARED_WORKER_URL)
const SHARED_URL = (import.meta.env.VITE_SHARED_WORKER_URL ?? "").replace(/\/+$/, "");
export function sharedUrl(): string {
  return SHARED_URL;
}
export function hasShared(): boolean {
  return SHARED_URL !== "";
}

function load(key: string): string {
  try {
    return localStorage.getItem(key) ?? "";
  } catch {
    return "";
  }
}
function save(key: string, val: string): void {
  try {
    localStorage.setItem(key, val);
  } catch {
    /* storage disabled - keep in-memory for this session */
  }
}

// default: dev (no shared baked in) -> own worker with an empty url (the Vite proxy); prod -> shared
const _persisted = load(MODE_KEY);
let mode: WorkerMode =
  _persisted === "custom" || _persisted === "shared" || _persisted === "off"
    ? _persisted
    : hasShared()
      ? "shared"
      : "custom";
// custom url/token are remembered even when disconnected ("off"), so reconnecting is one click
let customUrl = load(URL_KEY); // "" = same-origin (dev proxy in dev; nothing in prod)
let workerToken = load(TOKEN_KEY);

let sessionId = load(SESSION_KEY);
if (!sessionId) {
  sessionId = crypto.randomUUID();
  save(SESSION_KEY, sessionId);
}

export function getMode(): WorkerMode {
  return mode;
}
export function getWorkerUrl(): string {
  return customUrl;
}
export function getWorkerToken(): string {
  return workerToken;
}

export function setMode(m: WorkerMode): void {
  mode = m;
  save(MODE_KEY, m);
}
export function setCustom(url: string, token: string): void {
  customUrl = url.trim().replace(/\/+$/, "");
  workerToken = token.trim();
  save(URL_KEY, customUrl);
  save(TOKEN_KEY, workerToken);
}

// the base the requests actually hit: shared url in shared mode, the user's url in custom mode, and
// nothing when disconnected ("off") - so a disconnected client is offline even though its url is kept
function base(): string {
  if (mode === "shared") return SHARED_URL;
  if (mode === "off") return "";
  return customUrl;
}
export function workerUrlFor(path: string): string {
  return base() + path;
}

// Per-request headers: shared mode sends the session id (so the gateway routes it); custom mode
// sends the bearer token when set. Kept minimal so a token-less custom worker stays preflight-free.
export function requestHeaders(): Record<string, string> {
  if (mode === "shared") return { "x-yapuny-session": sessionId };
  if (mode === "custom" && workerToken) return { authorization: `Bearer ${workerToken}` };
  return {};
}
