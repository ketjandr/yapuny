// Reusable toasts via `toast.error(...)`: stacks up to MAX, auto-dismiss, closable.
import { create } from "zustand";

export type ToastKind = "error" | "info";
export interface Toast {
  id: string;
  kind: ToastKind;
  message: string;
  leaving?: boolean; // playing its exit animation before removal
}

const MAX = 5;
const TTL = 8000;
const EXIT_MS = 200; // keep in sync with the toast-out animation

interface ToastState {
  toasts: Toast[];
  push: (kind: ToastKind, message: string) => void;
  dismiss: (id: string) => void;
}

export const useToastStore = create<ToastState>((set, get) => ({
  toasts: [],
  push: (kind, message) => {
    const id = crypto.randomUUID();
    set((s) => ({ toasts: [...s.toasts, { id, kind, message }].slice(-MAX) }));
    setTimeout(() => get().dismiss(id), TTL);
  },
  // mark leaving (triggers the exit animation), then remove after it finishes
  dismiss: (id) => {
    const t = get().toasts.find((x) => x.id === id);
    if (!t || t.leaving) return;
    set((s) => ({ toasts: s.toasts.map((x) => (x.id === id ? { ...x, leaving: true } : x)) }));
    setTimeout(() => set((s) => ({ toasts: s.toasts.filter((x) => x.id !== id) })), EXIT_MS);
  },
}));

export const toast = {
  error: (message: string) => useToastStore.getState().push("error", message),
  info: (message: string) => useToastStore.getState().push("info", message),
};
