// Worker connection state for the navbar indicator + Connect UI. The actual target (url + token)
// lives in lib/workerBase so lib/api can read it synchronously; this store is the React view and
// the connect/check actions on top of it. One target per browser, persisted, shared across projects.
import { create } from "zustand";
import { api } from "@/lib/api";
import { getWorkerToken, getWorkerUrl, setWorkerConfig } from "@/lib/workerBase";

export type ConnStatus = "off" | "connecting" | "online" | "error";

interface WorkerInfo {
  device?: string;
  gpu?: string | null;
}

interface ConnState {
  url: string; // "" means same-origin (dev Vite proxy)
  token: string;
  status: ConnStatus;
  info: WorkerInfo | null;
  error: string | null;
  connect: (url: string, token: string) => Promise<boolean>; // save + verify; true when online
  check: () => Promise<void>; // re-verify the current target (on load / manual refresh)
  disconnect: () => void;
}

export const useConnStore = create<ConnState>((set) => ({
  url: getWorkerUrl(),
  token: getWorkerToken(),
  status: "off",
  info: null,
  error: null,

  connect: async (url, token) => {
    setWorkerConfig(url, token);
    set({ url: getWorkerUrl(), token: getWorkerToken(), status: "connecting", error: null });
    try {
      const h = await api.health();
      set({ status: "online", info: { device: h.device, gpu: h.gpu }, error: null });
      return true;
    } catch (e) {
      set({ status: "error", info: null, error: (e as Error).message });
      return false;
    }
  },

  check: async () => {
    set((s) => (s.status === "online" ? s : { ...s, status: "connecting" }));
    try {
      const h = await api.health();
      set({ status: "online", info: { device: h.device, gpu: h.gpu }, error: null });
    } catch (e) {
      set({ status: "error", info: null, error: (e as Error).message });
    }
  },

  disconnect: () => {
    setWorkerConfig("", "");
    set({ url: "", token: "", status: "off", info: null, error: null });
  },
}));
