// Worker connection state for the navbar indicator + Connect UI. The target (mode + url + token)
// lives in lib/workerBase so lib/api reads it synchronously; this store is the React view + the
// connect/check actions. Two modes: "custom" (the user's own worker URL) and "shared" (the hosted
// shared worker, zero-setup). One target per browser, persisted.
import { create } from "zustand";
import { api } from "@/lib/api";
import {
  getMode,
  getWorkerToken,
  getWorkerUrl,
  hasShared,
  setCustom,
  setMode,
  type WorkerMode,
} from "@/lib/workerBase";

export type ConnStatus = "off" | "connecting" | "online" | "error";

interface WorkerInfo {
  device?: string;
  gpu?: string | null;
  shared?: boolean;
}

interface ConnState {
  mode: WorkerMode;
  url: string;
  token: string;
  hasShared: boolean;
  status: ConnStatus;
  info: WorkerInfo | null;
  error: string | null;
  connectCustom: (url: string, token: string) => Promise<boolean>;
  connectShared: () => Promise<boolean>;
  check: () => Promise<void>;
  disconnect: () => void;
}

export const useConnStore = create<ConnState>((set) => {
  // verify the current target via /api/health; drives the indicator + connect result
  const verify = async (): Promise<boolean> => {
    set((s) => ({ status: s.status === "online" ? "online" : "connecting" }));
    try {
      const h = await api.health();
      set({ status: "online", info: { device: h.device, gpu: h.gpu, shared: h.shared }, error: null });
      return true;
    } catch (e) {
      set({ status: "error", info: null, error: (e as Error).message });
      return false;
    }
  };

  return {
    mode: getMode(),
    url: getWorkerUrl(),
    token: getWorkerToken(),
    hasShared: hasShared(),
    status: "off",
    info: null,
    error: null,

    connectCustom: async (url, token) => {
      if (!url.trim() && import.meta.env.PROD) {
        set({ status: "error", error: "Enter your worker's URL", info: null });
        return false;
      }
      setMode("custom");
      setCustom(url, token);
      set({ mode: "custom", url: getWorkerUrl(), token: getWorkerToken() });
      return verify();
    },

    connectShared: async () => {
      setMode("shared");
      set({ mode: "shared" });
      return verify();
    },

    check: async () => {
      await verify();
    },

    // go offline but keep the remembered url/token so reconnecting is one click (mode "off" makes
    // the effective base empty regardless of the stored url)
    disconnect: () => {
      setMode("off");
      set({ mode: "off", status: "off", info: null, error: null });
    },
  };
});
