// Benchmark state: (1) the per-node profiler for the active model (POST /bench/profile, one-shot),
// and (2) a head-to-head compare of several trained models (POST /bench/generate, SSE) whose events
// are tagged by graph_idx - each column is one model, filled as its stream arrives. The store owns
// the compare loop so it survives re-renders (e.g. the full-view toggle) while running.
import { create } from "zustand";
import { api } from "@/lib/api";
import { readSSE } from "@/lib/sse";
import type { BenchRunRequest, ModelGraphRequest, ProfileRequest } from "@/lib/types";

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
  text: string; // streamed generation, shown as a sample under the numbers
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
  runCompare: (entries: (ModelGraphRequest & { label: string })[], opts: Omit<BenchRunRequest, "graphs">) => Promise<void>;
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

  runCompare: async (entries, opts) => {
    if (get().comparing) return;
    set({
      comparing: true,
      columns: entries.map((e) => ({
        id: e.id,
        label: e.label,
        tokensPerSec: null,
        msPerToken: null,
        prefillMs: null,
        text: "",
        error: null,
        done: false,
      })),
    });

    const patch = (idx: number, fn: (c: BenchColumn) => BenchColumn) =>
      set((s) => ({ columns: s.columns.map((c, i) => (i === idx ? fn(c) : c)) }));

    let res: Response;
    try {
      res = await api.benchStream({ graphs: entries.map(({ id, graph }) => ({ id, graph })), ...opts });
    } catch (e) {
      set((s) => ({ comparing: false, columns: s.columns.map((c) => ({ ...c, error: (e as Error).message, done: true })) }));
      return;
    }

    try {
      for await (const ev of readSSE(res)) {
        const gi: number | undefined = ev.data?.graph_idx;
        // the trailing done (env only, no graph_idx) marks the whole run complete
        if (ev.event === "done" && gi === undefined) break;
        if (gi === undefined) continue;
        if (ev.event === "error") patch(gi, (c) => ({ ...c, error: ev.data.error, done: true }));
        else if (ev.event === "prefill") patch(gi, (c) => ({ ...c, prefillMs: ev.data.prefill_ms }));
        else if (ev.event === "token") patch(gi, (c) => ({ ...c, text: c.text + (ev.data.text ?? "") }));
        else if (ev.event === "done")
          patch(gi, (c) => ({
            ...c,
            tokensPerSec: ev.data.bench?.tokens_per_sec ?? c.tokensPerSec,
            msPerToken: ev.data.bench?.decode_ms_per_token ?? c.msPerToken,
            prefillMs: ev.data.bench?.prefill_ms ?? c.prefillMs,
            done: true,
          }));
      }
    } catch (e) {
      set((s) => ({ columns: s.columns.map((c) => (c.done ? c : { ...c, error: (e as Error).message, done: true })) }));
    } finally {
      set({ comparing: false });
    }
  },

  clearCompare: () => set({ comparing: false, columns: [] }),
}));
