// Shared debounced graph-validation state (worker POST /graph/validate). ValidationOverlay runs the
// checks and renders the messages; CompileBar reads `validating` + validity to gate the Compile
// button: an invalid graph can't be compiled, and the button stays disabled while a check is in
// flight (alongside the model/status poll).
import { create } from "zustand";

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}
export type ValidationView =
  | { kind: "idle" }
  | { kind: "checking" }
  | { kind: "ok"; result: ValidationResult }
  | { kind: "offline" };

interface ValidationStore {
  view: ValidationView;
  validating: boolean; // a validation request is pending/in-flight
  setView: (view: ValidationView) => void;
  setValidating: (validating: boolean) => void;
}

export const useValidationStore = create<ValidationStore>((set) => ({
  view: { kind: "idle" },
  validating: false,
  setView: (view) => set({ view }),
  setValidating: (validating) => set({ validating }),
}));
