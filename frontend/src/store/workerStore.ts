// Universal GPU-busy signal. The worker runs one job at a time (training and inference are mutually
// exclusive), and GET /worker/activity reports what currently holds it: {kind, modelId} or null.
// Every project reads this to gate its own Train / Generate controls when another project (or another
// kind of run) is using the GPU. A single poller (started once, ref-counted) keeps it fresh across
// tabs; the active run also refreshes it immediately on start/stop so the UI doesn't wait a tick.
import { create } from "zustand";
import { api } from "@/lib/api";

export type ActivityKind = "train" | "train_bench" | "generate" | "gen_bench";
export interface Activity {
  kind: ActivityKind;
  modelId: string;
}

interface WorkerState {
  activity: Activity | null;
  refresh: () => Promise<void>;
}

export const useWorkerStore = create<WorkerState>((set) => ({
  activity: null,
  refresh: async () => {
    try {
      const r = await api.workerActivity();
      const a = r?.active;
      set({ activity: a ? { kind: a.kind, modelId: a.model_id } : null });
    } catch {
      /* worker unreachable - leave the last known activity */
    }
  },
}));

// Is the GPU busy with something this control can't own right now? Returns the blocking activity, or
// null when free / already owned by this model's matching run. `ownKinds` are the run kinds this
// button drives - anything else (another project, or another kind on this project) blocks it, since
// the worker runs one job at a time. e.g. a project's Generate is blocked by its own running
// benchmark, and vice-versa.
export function blockingActivity(
  activity: Activity | null,
  modelId: string,
  ownKinds: ActivityKind[],
): Activity | null {
  if (!activity) return null;
  const ownsIt = activity.modelId === modelId && ownKinds.includes(activity.kind);
  return ownsIt ? null : activity;
}

// ref-counted global poller so multiple mounts share one interval
let pollers = 0;
let timer: ReturnType<typeof setInterval> | null = null;
export function startActivityPolling(intervalMs = 1500): () => void {
  pollers += 1;
  if (timer === null) {
    useWorkerStore.getState().refresh();
    timer = setInterval(() => useWorkerStore.getState().refresh(), intervalMs);
  }
  return () => {
    pollers -= 1;
    if (pollers <= 0 && timer !== null) {
      clearInterval(timer);
      timer = null;
      pollers = 0;
    }
  };
}
