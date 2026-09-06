// Live training run state, driven by POST /train/stream (SSE). The store owns the stream loop so a
// run keeps advancing while TrainSection is unmounted (mode switch / full-view toggle). Cancel is
// cooperative: stop() posts /train/stop; the backend breaks its loop and emits a final "stopped"
// update, which ends the stream here - we never tear the socket down mid-step.
import { create } from "zustand";
import { api } from "@/lib/api";
import { reconnectToBusyRun } from "@/lib/runReconnect";
import { readSSE } from "@/lib/sse";
import type { TrainRequest } from "@/lib/types";
import { toast } from "@/store/toastStore";
import { useWorkerStore } from "@/store/workerStore";

export type TrainStatus = "idle" | "running" | "stopping" | "completed" | "stopped" | "error";

// per-step minibatch train loss, streamed every step - the live curve (no eval, so it never stalls)
export interface StepLoss {
  step: number;
  loss: number;
}

interface TrainState {
  status: TrainStatus;
  modelId: string | null; // the model this run belongs to (stats hide when it isn't the active one)
  step: number;
  maxSteps: number;
  loss: number | null; // latest per-step train loss (headline stat)
  stepsPerSec: number | null; // live steps/s (rolling), streamed per step when bench is on
  curve: StepLoss[]; // dense per-step train loss (the whole curve)
  phase: "tokenizing" | "training" | null; // the model trains its own tokenizer before stepping
  error: string | null;
  start: (req: TrainRequest) => Promise<void>;
  follow: (id: string) => Promise<void>; // reattach to an in-progress run after a reload
  stop: () => Promise<void>;
  reset: () => void;
}

const IDLE = {
  status: "idle" as TrainStatus,
  step: 0,
  maxSteps: 0,
  loss: null,
  stepsPerSec: null,
  curve: [] as StepLoss[],
  phase: null as "tokenizing" | "training" | null,
  error: null as string | null,
};

// backend train_state.status -> our status ("running" while stepping, then a terminal state)
const mapStatus = (s: string | undefined): TrainStatus =>
  s === "completed" || s === "stopped" || s === "error" ? s : "running";

export const useTrainStore = create<TrainState>((set, get) => {
  // consume an SSE train stream (start or follow) to completion, projecting each frame into state
  const consume = async (res: Response) => {
    let sawFrame = false;
    try {
      for await (const ev of readSSE(res)) {
        if (ev.event !== "update") continue;
        sawFrame = true;
        const d = ev.data;
        set((s) => {
          // curve is server-authoritative (downsampled [step, loss]) so it survives a reload
          const curve: StepLoss[] = d.curve
            ? d.curve.map(([step, loss]: [number, number]) => ({ step, loss }))
            : s.curve;
          return {
            step: d.step ?? s.step,
            maxSteps: d.max_steps ?? s.maxSteps, // follow learns maxSteps from the frames
            loss: d.train_loss ?? s.loss,
            stepsPerSec: d.steps_per_sec ?? s.stepsPerSec,
            phase: d.phase ?? s.phase,
            curve,
            status: mapStatus(d.status),
            error: d.status === "error" ? (d.error ?? "training failed") : s.error,
          };
        });
      }
    } catch (e) {
      set({ status: "error", error: (e as Error).message });
      return;
    }
    // stream closed. If a terminal frame already set the status, keep it. Otherwise: a dropped
    // connection mid-run settles to completed; a stream that carried NO frames (e.g. we followed a
    // run that just ended) never really started here, so fall back to idle rather than fake-complete.
    set((s) => {
      if (s.status === "stopping") return { status: "stopped" };
      if (s.status === "running") return sawFrame ? { status: "completed" } : { ...IDLE, modelId: null };
      return {};
    });
    useWorkerStore.getState().refresh(); // run ended - the GPU is free, un-gate other controls
  };

  return {
    ...IDLE,
    modelId: null,

    start: async (req) => {
      if (get().status === "running" || get().status === "stopping") return;
      set({ ...IDLE, status: "running", modelId: req.id, maxSteps: req.max_steps ?? 2000 });
      let res: Response;
      try {
        res = await api.trainStream(req);
      } catch (e) {
        set({ ...IDLE, modelId: null });
        toast.error(`Couldn't start training: ${(e as Error).message}`);
        return;
      }
      // the stream endpoint returns the raw Response, so a rejected start (e.g. 409 - another model
      // is already training) arrives as a non-OK body, not an exception. Surface it as a toast and
      // don't leave the pane in a running/completed state it never earned.
      if (!res.ok) {
        const detail = await res
          .json()
          .then((j) => j?.detail as string | undefined)
          .catch(() => undefined);
        set({ ...IDLE, modelId: null });
        toast.error(`Couldn't start training: ${detail ?? `${res.status} ${res.statusText}`}`);
        // reflect whatever run (single or bench) is in progress so this tab's Train button disables
        reconnectToBusyRun();
        useWorkerStore.getState().refresh();
        return;
      }
      useWorkerStore.getState().refresh(); // this run now holds the GPU - gate other controls
      await consume(res);
    },

    // reattach to a run already in progress on the worker (e.g. after a page reload). The per-step
    // history before reconnect is lost (losses aren't persisted), so the curve resumes from here.
    follow: async (id) => {
      if (get().status === "running" || get().status === "stopping") return;
      set({ ...IDLE, status: "running", modelId: id });
      let res: Response;
      try {
        res = await api.trainFollow();
      } catch {
        set({ status: "idle", modelId: null });
        return;
      }
      await consume(res);
    },

    stop: async () => {
      if (get().status !== "running") return;
      set({ status: "stopping" }); // the stream stays open; it ends when the "stopped" update lands
      try {
        await api.trainStop();
      } catch {
        /* the run may have finished on its own between click and post - the stream settles it */
      }
    },

    reset: () => set({ ...IDLE, modelId: null }),
  };
});
