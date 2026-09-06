// Typed worker client (dev: /api via Vite proxy; prod: worker URL over CORS).
import type {
  GraphRequest, ModelGraphRequest, GenerateRequest, TrainRequest, TrainBenchRequest, BenchRunRequest, ProfileRequest,
} from "./types";

const j = (r: Response) => { if (!r.ok) throw new Error(`${r.status} ${r.statusText}`); return r.json(); };
const post = (path: string, body: unknown) =>
  fetch(`/api${path}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });

export const api = {
  validate: (graph: GraphRequest) => post("/graph/validate", graph).then(j),
  compile: (req: ModelGraphRequest) => post("/graph/compile", req).then(j),
  modelStatus: (req: ModelGraphRequest) => post("/model/status", req).then(j),
  listModels: () => fetch("/api/models").then(j),
  deleteModel: (id: string) => fetch(`/api/model/${id}`, { method: "DELETE" }).then(j),

  dataStatus: () => fetch("/api/data/status").then(j),
  uploadCorpus: (file: File) => {
    const fd = new FormData(); fd.append("file", file);
    return fetch("/api/data/upload", { method: "POST", body: fd }).then(j);
  },
  getCorpus: () => fetch("/api/data/corpus").then(j),
  saveCorpus: (text: string) => post("/data/corpus", { text }).then(j),
  deleteCorpus: () => fetch("/api/data/corpus", { method: "DELETE" }).then(j),

  fusionAvailable: () => fetch("/api/fusion/available").then(j),
  quantizationAvailable: () => fetch("/api/quantization/available").then(j),

  trainStop: () => post("/train/stop", {}).then(j),
  trainStatus: () => fetch("/api/train/status").then(j),
  trainBenchStatus: () => fetch("/api/train/bench/status").then(j),

  // streaming endpoints return the raw Response; read with lib/sse.ts
  trainStream: (req: TrainRequest) => post("/train/stream", req),
  trainFollow: () => fetch("/api/train/follow"), // reattach to an in-progress run (no start)
  trainBench: (req: TrainBenchRequest) => post("/train/bench", req), // sequential multi-model bench
  trainBenchFollow: () => fetch("/api/train/bench/follow"),
  generateStream: (req: GenerateRequest) => post("/generate/stream", req),
  benchStream: (req: BenchRunRequest) => post("/bench/generate", req),

  profile: (req: ProfileRequest) => post("/bench/profile", req).then(j),
};
