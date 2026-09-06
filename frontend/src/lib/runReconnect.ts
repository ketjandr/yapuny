// Reattach this session to whatever training run the worker is busy with - a single run or a
// multi-model benchmark. Used after a 409 (a Train/Train-&-benchmark click that lost the race),
// so the losing project's controls reflect the busy worker and its Train button greys out.
// Stores are imported lazily-at-call-time (via getState()), so the store<->helper cycle is safe.
import { api } from "@/lib/api";
import { useBenchTrainStore } from "@/store/benchTrainStore";
import { useTrainStore } from "@/store/trainStore";

export async function reconnectToBusyRun(): Promise<void> {
  const bt = useBenchTrainStore.getState();
  const ts = useTrainStore.getState();
  // already tracking a live run here - nothing to reattach
  if (bt.status === "running" || bt.status === "stopping" || ts.status === "running" || ts.status === "stopping") return;
  try {
    const s = await api.trainBenchStatus();
    if (s?.status === "running") {
      useBenchTrainStore.getState().follow();
      return;
    }
    const t = await api.trainStatus();
    if (t?.status === "running" && t.training_id) useTrainStore.getState().follow(t.training_id);
  } catch {
    /* worker unreachable - leave state idle */
  }
}
