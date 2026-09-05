// Backend-driven compile/train status for the active model. The worker is the source of truth:
// POST /model/status (an in-memory full-hash match) drives needs-compile vs compiled, and
// compile/model_status report `trained` (weights loaded from the persistent locker in store.py).
// Polled by CompileBar; also read by SaveBar's revert gate. This is intentionally separate from the
// canvas (editor) store — it mirrors worker state, not the graph being drawn.
import { create } from "zustand";

export type CompileStatus = "unknown" | "checking" | "ready" | "needs_compile" | "offline";

interface CompileState {
  status: CompileStatus;
  trained: boolean;
  compiling: boolean;
  polling: boolean; // a model_status check is pending/in-flight (Compile is disabled meanwhile)
  setResult: (status: CompileStatus, trained: boolean) => void;
  setChecking: () => void;
  setCompiling: (compiling: boolean) => void;
  setPolling: (polling: boolean) => void;
}

export const useCompileStore = create<CompileState>((set) => ({
  status: "unknown",
  trained: false,
  compiling: false,
  polling: false,
  setResult: (status, trained) => set({ status, trained }),
  // only surface "checking" before the first definitive answer, so the dot doesn't flicker on edits
  setChecking: () =>
    set((s) => (s.status === "unknown" || s.status === "offline" ? { status: "checking" } : s)),
  setCompiling: (compiling) => set({ compiling }),
  setPolling: (polling) => set({ polling }),
}));
