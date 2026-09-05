// Live training run state, driven by POST /train/stream (SSE). The store owns the stream loop so a
// run keeps advancing while TrainSection is unmounted (mode switch / full-view toggle). Cancel is
// cooperative: stop() posts /train/stop; the backend breaks its loop and emits a final "stopped"
// update, which ends the stream here - we never tear the socket down mid-step.
import { create } from "zustand";
import { api } from "@/lib/api";
import { readSSE } from "@/lib/sse";
import type { TrainRequest } from "@/lib/types";

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
  error: null as string | null,
};

// backend train_state.status -> our status ("running" while stepping, then a terminal state)
const mapStatus = (s: string | undefined): TrainStatus =>
  s === "completed" || s === "stopped" || s === "error" ? s : "running";

export const useTrainStore = create<TrainState>((set, get) => {
  // consume an SSE train stream (start or follow) to completion, projecting each frame into state
  const consume = async (res: Response) => {
    try {
      for await (const ev of readSSE(res)) {
        if (ev.event !== "update") continue;
        const d = ev.data;
        set((s) => {
          // append the per-step train loss once per new step (the stream ticks every step)
          const curve = s.curve;
          const nextCurve =
            d.train_loss != null && (curve.length === 0 || curve[curve.length - 1].step !== d.step)
              ? [...curve, { step: d.step ?? 0, loss: d.train_loss }]
              : curve;
          return {
            step: d.step ?? s.step,
            maxSteps: d.max_steps ?? s.maxSteps, // follow learns maxSteps from the frames
            loss: d.train_loss ?? s.loss,
            stepsPerSec: d.bench?.steps_per_sec ?? s.stepsPerSec,
            curve: nextCurve,
            status: mapStatus(d.status),
            error: d.status === "error" ? (d.error ?? "training failed") : s.error,
          };
        });
      }
    } catch (e) {
      set({ status: "error", error: (e as Error).message });
      return;
    }
    // stream closed: if no terminal update arrived (dropped connection), settle to a sane state
    set((s) => (s.status === "running" ? { status: "completed" } : s.status === "stopping" ? { status: "stopped" } : {}));
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
        set({ status: "error", error: (e as Error).message });
        return;
      }
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
