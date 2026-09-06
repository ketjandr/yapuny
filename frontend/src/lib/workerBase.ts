// The configured worker target: base URL + optional bearer token. Plain module state backed by
// localStorage so lib/api can build request URLs synchronously, with no React/store dependency.
// An empty base means "same origin": in dev the Vite proxy serves /api -> worker (no connect
// needed); in prod an empty base points at the static host (no worker), so the Connect UI must set
// one. Trailing slashes are stripped so `${base}/api/...` is always well-formed.
const URL_KEY = "yapuny.worker.url";
const TOKEN_KEY = "yapuny.worker.token";

function load(key: string): string {
  try {
    return localStorage.getItem(key) ?? "";
  } catch {
    return "";
  }
}

let workerUrl = load(URL_KEY);
let workerToken = load(TOKEN_KEY);

export function getWorkerUrl(): string {
  return workerUrl;
}
export function getWorkerToken(): string {
  return workerToken;
}

export function setWorkerConfig(url: string, token: string): void {
  workerUrl = url.trim().replace(/\/+$/, "");
  workerToken = token.trim();
  try {
    localStorage.setItem(URL_KEY, workerUrl);
    localStorage.setItem(TOKEN_KEY, workerToken);
  } catch {
    /* storage disabled - keep the in-memory values for this session */
  }
}

// full request URL for a path (the path carries its own leading /api or /health)
export function workerUrlFor(path: string): string {
  return workerUrl + path;
}

// Authorization header when a token is set; omitted otherwise so a local worker stays
// preflight-free (a custom header would force a CORS preflight on every request)
export function authHeaders(): Record<string, string> {
  return workerToken ? { authorization: `Bearer ${workerToken}` } : {};
}
