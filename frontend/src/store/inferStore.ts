// Single-model text generation, driven by POST /generate/stream (SSE). The store owns the stream so
// output keeps flowing while GenerateSection is unmounted (mode/full-view toggle). Stop aborts the
// fetch, which closes the SSE and frees the GPU on the worker (its generator's finally clears the
// inference lock). tok/s is measured client-side from token arrivals - no profiling overhead.
import { create } from "zustand";
import { api } from "@/lib/api";
import { readSSE } from "@/lib/sse";
import type { GenerateRequest } from "@/lib/types";
import { toast } from "@/store/toastStore";
import { useWorkerStore } from "@/store/workerStore";

export type InferStatus = "idle" | "running" | "done" | "error";

interface InferState {
  status: InferStatus;
  modelId: string | null; // the model this run belongs to (output hides when it isn't the open one)
  text: string; // streamed output so far
  tokensPerSec: number | null;
  tokens: number; // tokens generated so far (the live count)
  prefillMs: number | null; // prefill latency (time to first token), measured client-side
  error: string | null;
  start: (req: GenerateRequest) => Promise<void>;
  stop: () => void;
  reset: () => void;
}

const IDLE = {
  status: "idle" as InferStatus,
  text: "",
  tokensPerSec: null as number | null,
  tokens: 0,
  prefillMs: null as number | null,
  error: null as string | null,
};

let ctrl: AbortController | null = null;

export const useInferStore = create<InferState>((set, get) => ({
  ...IDLE,
  modelId: null,

  start: async (req) => {
    if (get().status === "running") return;
    set({ ...IDLE, status: "running", modelId: req.id });
    ctrl = new AbortController();
    let res: Response;
    try {
      res = await api.generateStream(req, ctrl.signal);
    } catch (e) {
      if ((e as Error).name === "AbortError") return; // user pressed Stop before headers
      set({ ...IDLE, modelId: null });
      toast.error(`Couldn't generate: ${(e as Error).message}`);
      return;
    }
    // a rejected start (409 - the GPU is busy) arrives as a non-OK body, not a throw
    if (!res.ok) {
      const detail = await res.json().then((b) => b?.detail as string | undefined).catch(() => undefined);
      set({ ...IDLE, modelId: null });
      toast.error(`Couldn't generate: ${detail ?? `${res.status} ${res.statusText}`}`);
      useWorkerStore.getState().refresh(); // reflect whoever holds the GPU
      return;
    }
    useWorkerStore.getState().refresh(); // this run now owns the GPU - update the busy signal

    const t0 = performance.now();
    let count = 0;
    let firstAt = 0; // timestamp of the first token, so tok/s measures decode only (excludes prefill)
    try {
      for await (const ev of readSSE(res)) {
        if (ev.event === "token") {
          count += 1;
          const now = performance.now();
          if (count === 1) firstAt = now;
          // decode rate over the tokens after the first (the first's cost is the prefill/ttft)
          const secs = (now - firstAt) / 1000;
          set((s) => ({
            text: s.text + (ev.data.text ?? ""),
            tokens: count,
            prefillMs: s.prefillMs ?? now - t0,
            tokensPerSec: count > 1 && secs > 0 ? (count - 1) / secs : s.tokensPerSec,
          }));
        } else if (ev.event === "error") {
          set({ status: "error", error: ev.data.error ?? "generation failed" });
        } else if (ev.event === "done") {
          set((s) => (s.status === "running" ? { status: "done" } : {}));
        }
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") set({ status: "error", error: (e as Error).message });
    } finally {
      ctrl = null;
      set((s) => (s.status === "running" ? { status: "done" } : {}));
      useWorkerStore.getState().refresh(); // GPU freed
    }
  },

  stop: () => {
    ctrl?.abort();
    ctrl = null;
    set((s) => (s.status === "running" ? { status: "done" } : {}));
    useWorkerStore.getState().refresh();
  },

  reset: () => set({ ...IDLE, modelId: null }),
}));
