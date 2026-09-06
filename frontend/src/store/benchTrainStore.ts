// Sequential multi-model training benchmark (POST /train/bench, SSE). Trains up to 5 compiled
// models one after another; the worker streams the whole run's state each tick (all models, with
// `current` marking the one in flight). The store owns the loop so a run survives re-renders / a
// page reload (via follow()), mirroring trainStore. Also holds the compare selection (config).
import { create } from "zustand";
import { api } from "@/lib/api";
import { reconnectToBusyRun } from "@/lib/runReconnect";
import { readSSE } from "@/lib/sse";
import type { NodeProfile } from "@/store/benchStore";
import { toast } from "@/store/toastStore";

export type BenchTrainStatus = "idle" | "running" | "stopping" | "completed" | "stopped" | "error";

export interface ModelBench {
  forward_ms: number;
  backward_ms: number;
  steps_per_sec: number;
  peak_vram_mb: number | null;
  profile: { nodes: NodeProfile[]; total_us: number };
}

export interface BenchModel {
  id: string;
  status: "pending" | "running" | "done" | "stopped";
  phase: "tokenizing" | "training" | null; // trains its own tokenizer before stepping
  step: number;
  maxSteps: number;
  loss: number | null;
  stepsPerSec: number | null;
  curve: { step: number; loss: number }[]; // accumulated per-step (this session)
  bench: ModelBench | null; // final summary once the model finishes
}

interface BenchTrainState {
  status: BenchTrainStatus;
  current: number; // index of the model training now
  models: BenchModel[]; // per-model progress for the run
  error: string | null;
  selected: string[]; // OTHER project ids to compare (besides the open model); run config, not saved
  toggleSelected: (id: string) => void;
  setSelected: (ids: string[]) => void; // restore selection when reattaching to a run
  clearSelected: () => void;
  start: (ids: string[], hp: { max_steps: number; batch_size: number; learning_rate: number }) => Promise<void>;
  follow: () => Promise<void>;
  stop: () => Promise<void>;
  reset: () => void;
}

const MAX_OTHERS = 4; // 5 columns total including the open model

const IDLE_RUN = {
  status: "idle" as BenchTrainStatus,
  current: 0,
  models: [] as BenchModel[],
  error: null as string | null,
};

const mapStatus = (s: string | undefined): BenchTrainStatus =>
  s === "completed" || s === "stopped" || s === "error" ? s : "running";

const emptyModel = (id: string, maxSteps: number): BenchModel => ({
  id,
  status: "pending",
  phase: null,
  step: 0,
  maxSteps,
  loss: null,
  stepsPerSec: null,
  curve: [],
  bench: null,
});

export const useBenchTrainStore = create<BenchTrainState>((set, get) => {
  const consume = async (res: Response) => {
    let sawFrame = false;
    try {
      for await (const ev of readSSE(res)) {
        if (ev.event !== "update") continue;
        sawFrame = true;
        const d = ev.data; // { status, current, models: [{id, status, step, max_steps, train_loss, steps_per_sec, bench}] }
        set((s) => {
          const models: BenchModel[] = (d.models ?? []).map((m: any, i: number) => {
            const prev = s.models[i];
            // curve is server-authoritative (downsampled [step, loss]) so it survives a reload
            const curve: { step: number; loss: number }[] = m.curve
              ? m.curve.map(([step, loss]: [number, number]) => ({ step, loss }))
              : (prev?.curve ?? []);
            return {
              id: m.id,
              status: m.status,
              phase: m.phase ?? null,
              step: m.step ?? 0,
              maxSteps: m.max_steps ?? prev?.maxSteps ?? 0,
              loss: m.train_loss ?? null,
              stepsPerSec: m.steps_per_sec ?? null,
              curve,
              bench: m.bench ?? prev?.bench ?? null,
            };
          });
          return {
            status: mapStatus(d.status),
            current: d.current ?? s.current,
            models,
            error: d.status === "error" ? (d.error ?? "benchmark failed") : s.error,
          };
        });
      }
    } catch (e) {
      set({ status: "error", error: (e as Error).message });
      return;
    }
    // stream closed: keep a terminal status; a dropped run settles to completed; an empty stream
    // (followed a run that just ended) resets to idle rather than fake-completing.
    set((s) => {
      if (s.status === "stopping") return { status: "stopped" };
      if (s.status === "running") return sawFrame ? { status: "completed" } : { ...IDLE_RUN };
      return {};
    });
  };

  return {
    ...IDLE_RUN,
    selected: [],

    toggleSelected: (id) =>
      set((s) => ({
        selected: s.selected.includes(id)
          ? s.selected.filter((x) => x !== id)
          : s.selected.length >= MAX_OTHERS
            ? s.selected
            : [...s.selected, id],
      })),
    setSelected: (ids) => set({ selected: ids.slice(0, MAX_OTHERS) }),
    clearSelected: () => set({ selected: [] }),

    start: async (ids, hp) => {
      if (get().status === "running" || get().status === "stopping") return;
      set({ ...IDLE_RUN, status: "running", models: ids.map((id) => emptyModel(id, hp.max_steps)) });
      let res: Response;
      try {
        res = await api.trainBench({ model_ids: ids, ...hp });
      } catch (e) {
        set({ ...IDLE_RUN });
        toast.error(`Couldn't start benchmark: ${(e as Error).message}`);
        return;
      }
      if (!res.ok) {
        const detail = await res
          .json()
          .then((j) => j?.detail as string | undefined)
          .catch(() => undefined);
        set({ ...IDLE_RUN });
        toast.error(`Couldn't start benchmark: ${detail ?? `${res.status} ${res.statusText}`}`);
        // a 409 means the worker is already busy - reattach to that run so this project's Train
        // button greys out on the failed click (same as vanilla training does)
        reconnectToBusyRun();
        return;
      }
      await consume(res);
    },

    follow: async () => {
      if (get().status === "running" || get().status === "stopping") return;
      set({ ...IDLE_RUN, status: "running" });
      let res: Response;
      try {
        res = await api.trainBenchFollow();
      } catch {
        set({ ...IDLE_RUN });
        return;
      }
      await consume(res);
    },

    stop: async () => {
      if (get().status !== "running") return;
      set({ status: "stopping" }); // shares /train/stop with single training (both key off worker.training)
      try {
        await api.trainStop();
      } catch {
        /* the run may have finished between click and post - the stream settles it */
      }
    },

    reset: () => set({ ...IDLE_RUN, selected: [] }),
  };
});
