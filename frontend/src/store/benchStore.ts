// Benchmark state: (1) the per-node profiler for the active model (POST /bench/profile, one-shot),
// and (2) a head-to-head compare of several trained models (POST /bench/generate, SSE) whose events
// are tagged by graph_idx - each column is one model, filled as its stream arrives. The store owns
// the compare loop so it survives re-renders (e.g. the full-view toggle) while running.
import { create } from "zustand";
import { api } from "@/lib/api";
import { readSSE } from "@/lib/sse";
import type { BenchRunRequest, ModelGraphRequest, ProfileRequest } from "@/lib/types";
import { useWorkerStore } from "@/store/workerStore";

let compareCtrl: AbortController | null = null; // aborts the in-flight compare stream on stop

export interface NodeProfile {
  node_id: string;
  logical_id: string;
  self_us: number;
  pct: number;
}

// one column of the compare table - a single model's run, filled in as its SSE events arrive
export interface BenchColumn {
  id: string;
  label: string;
  tokensPerSec: number | null;
  msPerToken: number | null;
  prefillMs: number | null;
  peakVramMb: number | null;
  profile: NodeProfile[]; // per-node decode profile (arrives once the model finishes)
  text: string; // streamed generation - the model's actual output, inspectable per model
  running: boolean; // this column is the one currently generating (runs are sequential)
  error: string | null;
  done: boolean;
}

interface BenchState {
  // per-node profiler (active model)
  profiling: boolean;
  profile: { nodes: NodeProfile[]; total_us: number } | null;
  profileError: string | null;
  runProfile: (req: ProfileRequest) => Promise<void>;

  // multi-model compare
  comparing: boolean;
  columns: BenchColumn[];
  compareOwner: string | null; // model id that started the run (column 0) - results isolate to it
  runCompare: (entries: (ModelGraphRequest & { label: string })[], opts: Omit<BenchRunRequest, "graphs">) => Promise<void>;
  stopCompare: () => void;
  clearCompare: () => void;
}

export const useBenchStore = create<BenchState>((set, get) => ({
  profiling: false,
  profile: null,
  profileError: null,

  runProfile: async (req) => {
    if (get().profiling) return;
    set({ profiling: true, profileError: null });
    try {
      const res = await api.profile(req);
      set({ profile: { nodes: res.nodes, total_us: res.total_us } });
    } catch (e) {
      set({ profileError: (e as Error).message, profile: null });
    } finally {
      set({ profiling: false });
    }
  },

  comparing: false,
  columns: [],
  compareOwner: null,

  runCompare: async (entries, opts) => {
    if (get().comparing) return;
    set({
      comparing: true,
      compareOwner: entries[0]?.id ?? null,
      columns: entries.map((e) => ({
        id: e.id,
        label: e.label,
        tokensPerSec: null,
        msPerToken: null,
        prefillMs: null,
        peakVramMb: null,
        profile: [],
        text: "",
        running: false,
        error: null,
        done: false,
      })),
    });

    const patch = (idx: number, fn: (c: BenchColumn) => BenchColumn) =>
      set((s) => ({ columns: s.columns.map((c, i) => (i === idx ? fn(c) : c)) }));

    compareCtrl = new AbortController();
    let res: Response;
    try {
      res = await api.benchStream({ graphs: entries.map(({ id, graph }) => ({ id, graph })), ...opts }, compareCtrl.signal);
    } catch (e) {
      if ((e as Error).name === "AbortError") { set({ comparing: false }); return; }
      set((s) => ({ comparing: false, columns: s.columns.map((c) => ({ ...c, error: (e as Error).message, done: true })) }));
      return;
    }
    if (!res.ok) {
      const detail = await res.json().then((b) => b?.detail as string | undefined).catch(() => undefined);
      set((s) => ({ comparing: false, columns: s.columns.map((c) => ({ ...c, error: detail ?? `${res.status}`, done: true })) }));
      useWorkerStore.getState().refresh();
      return;
    }
    useWorkerStore.getState().refresh(); // this run now owns the GPU

    try {
      for await (const ev of readSSE(res)) {
        const gi: number | undefined = ev.data?.graph_idx;
        // the trailing done (env only, no graph_idx) marks the whole run complete
        if (ev.event === "done" && gi === undefined) break;
        if (gi === undefined) continue;
        if (ev.event === "graph_start") patch(gi, (c) => ({ ...c, running: true }));
        else if (ev.event === "error") patch(gi, (c) => ({ ...c, error: ev.data.error, running: false, done: true }));
        else if (ev.event === "prefill") patch(gi, (c) => ({ ...c, prefillMs: ev.data.prefill_ms }));
        else if (ev.event === "token") patch(gi, (c) => ({ ...c, text: c.text + (ev.data.text ?? "") }));
        else if (ev.event === "profile") patch(gi, (c) => ({ ...c, profile: ev.data.nodes ?? [] }));
        else if (ev.event === "done")
          patch(gi, (c) => ({
            ...c,
            tokensPerSec: ev.data.bench?.tokens_per_sec ?? c.tokensPerSec,
            msPerToken: ev.data.bench?.decode_ms_per_token ?? c.msPerToken,
            prefillMs: ev.data.bench?.prefill_ms ?? c.prefillMs,
            peakVramMb: ev.data.bench?.peak_vram_mb ?? c.peakVramMb,
            running: false,
            done: true,
          }));
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        set((s) => ({ columns: s.columns.map((c) => (c.done ? c : { ...c, error: (e as Error).message, running: false, done: true })) }));
      }
    } finally {
      compareCtrl = null;
      set({ comparing: false });
      useWorkerStore.getState().refresh(); // GPU freed
    }
  },

  stopCompare: () => {
    compareCtrl?.abort();
    compareCtrl = null;
    set({ comparing: false });
    useWorkerStore.getState().refresh();
  },

  clearCompare: () => set({ comparing: false, columns: [], compareOwner: null }),
}));
