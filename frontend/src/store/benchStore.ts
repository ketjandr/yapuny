// Benchmark state: (1) the per-node profiler for the active model (POST /bench/profile, one-shot),
// and (2) a head-to-head compare of several trained models (POST /generate/bench, SSE) whose events
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
  text: string; // streamed generation - shown as the canonical output while this column runs
  genTokens: number; // tokens generated so far (live count, for the current-model stat row)
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
  follow: () => Promise<void>; // rehydrate the last run's results after a reload (the worker keeps them)
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
        genTokens: 0,
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

    let firstAt = 0; // first-token timestamp of the running column, for a live decode tok/s
    try {
      for await (const ev of readSSE(res)) {
        const gi: number | undefined = ev.data?.graph_idx;
        // the trailing done (env only, no graph_idx) marks the whole run complete
        if (ev.event === "done" && gi === undefined) break;
        if (gi === undefined) continue;
        if (ev.event === "graph_start") {
          firstAt = 0;
          patch(gi, (c) => ({ ...c, running: true, text: "", genTokens: 0 }));
        } else if (ev.event === "error") patch(gi, (c) => ({ ...c, error: ev.data.error, running: false, done: true }));
        else if (ev.event === "prefill") patch(gi, (c) => ({ ...c, prefillMs: ev.data.prefill_ms }));
        else if (ev.event === "token") {
          const now = performance.now();
          if (firstAt === 0) firstAt = now;
          const secs = (now - firstAt) / 1000;
          patch(gi, (c) => {
            const n = c.genTokens + 1;
            // live decode rate (tokens after the first); the final `done` overwrites with the
            // worker's authoritative number
            return { ...c, text: c.text + (ev.data.text ?? ""), genTokens: n, tokensPerSec: n > 1 && secs > 0 ? (n - 1) / secs : c.tokensPerSec };
          });
        } else if (ev.event === "profile") patch(gi, (c) => ({ ...c, profile: ev.data.nodes ?? [] }));
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

  follow: async () => {
    if (get().comparing) return; // a live run this session owns the state - don't clobber it
    let s: { owner: string | null; status: string; columns?: any[] };
    try {
      s = await api.benchGenStatus();
    } catch {
      return;
    }
    if (!s?.owner || !s.columns?.length) return; // nothing to rehydrate
    set({
      compareOwner: s.owner,
      comparing: s.status === "running",
      columns: s.columns.map((c) => ({
        id: c.id,
        label: c.label ?? c.id, // the table labels rows from project titles, not this
        tokensPerSec: c.tokens_per_sec ?? null,
        msPerToken: c.ms_per_token ?? null,
        prefillMs: c.prefill_ms ?? null,
        peakVramMb: c.peak_vram_mb ?? null,
        profile: c.profile ?? [],
        text: c.text ?? "",
        genTokens: c.gen_tokens ?? 0,
        running: c.running ?? false,
        error: c.error ?? null,
        done: c.done ?? false,
      })),
    });
  },

  stopCompare: () => {
    compareCtrl?.abort();
    compareCtrl = null;
    api.generateStop().catch(() => {}); // break the worker's loop so it stops mid-compare, frees GPU
    set({ comparing: false });
    useWorkerStore.getState().refresh();
  },

  clearCompare: () => set({ comparing: false, columns: [], compareOwner: null }),
}));
