// Typed worker client. Every request is built from the configured worker base (lib/workerBase):
// dev leaves it empty and the Vite proxy serves /api -> worker; prod sets it via the Connect UI and
// calls the worker directly (CORS + optional bearer token on the worker).
import type {
  GraphRequest, ModelGraphRequest, GenerateRequest, TrainRequest, TrainBenchRequest, BenchRunRequest, ProfileRequest,
} from "./types";
import { requestHeaders, workerUrlFor } from "./workerBase";

const j = (r: Response) => { if (!r.ok) throw new Error(`${r.status} ${r.statusText}`); return r.json(); };

// generic request against the worker base, with the session/token header attached. `path` carries
// its own leading /api (or /health). Callers add method/body/signal via `init`.
const req = (path: string, init?: RequestInit) =>
  fetch(workerUrlFor(path), { ...init, headers: { ...requestHeaders(), ...(init?.headers ?? {}) } });

const post = (path: string, body: unknown, signal?: AbortSignal) =>
  req(`/api${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

export const api = {
  health: () => req("/api/health").then(j), // liveness + device (drives the Connect check + indicator)

  validate: (graph: GraphRequest) => post("/graph/validate", graph).then(j),
  compile: (req: ModelGraphRequest) => post("/graph/compile", req).then(j),
  modelStatus: (req: ModelGraphRequest) => post("/model/status", req).then(j),
  listModels: () => req("/api/models").then(j),
  deleteModel: (id: string) => req(`/api/model/${id}`, { method: "DELETE" }).then(j),

  dataStatus: () => req("/api/data/status").then(j),
  uploadCorpus: (file: File) => {
    const fd = new FormData(); fd.append("file", file);
    return req("/api/data/upload", { method: "POST", body: fd }).then(j);
  },
  getCorpus: () => req("/api/data/corpus").then(j),
  saveCorpus: (text: string) => post("/data/corpus", { text }).then(j),
  deleteCorpus: () => req("/api/data/corpus", { method: "DELETE" }).then(j),

  fusionAvailable: () => req("/api/fusion/available").then(j),
  quantizationAvailable: () => req("/api/quantization/available").then(j),

  trainStop: () => post("/train/stop", {}).then(j),
  trainStatus: () => req("/api/train/status").then(j),
  trainBenchStatus: () => req("/api/train/bench/status").then(j),
  benchGenStatus: () => req("/api/generate/bench/status").then(j), // last compare's results (reload)
  workerActivity: () => req("/api/worker/activity").then(j), // what occupies the GPU (train/infer)

  // streaming endpoints return the raw Response; read with lib/sse.ts
  trainStream: (r: TrainRequest) => post("/train/stream", r),
  trainFollow: () => req("/api/train/follow"), // reattach to an in-progress run (no start)
  trainBench: (r: TrainBenchRequest) => post("/train/bench", r), // sequential multi-model bench
  trainBenchFollow: () => req("/api/train/bench/follow"),
  generateStream: (r: GenerateRequest, signal?: AbortSignal) => post("/generate/stream", r, signal),
  generateStop: () => post("/generate/stop", {}).then(j), // cooperative cancel for generate + bench
  benchStream: (r: BenchRunRequest, signal?: AbortSignal) => post("/generate/bench", r, signal),

  profile: (r: ProfileRequest) => post("/bench/profile", r).then(j),
};
