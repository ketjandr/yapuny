// Frontend-owned model registry in localStorage (source of truth for which models exist).
import type { RegistryEntry } from "./types";

const KEY = "yapuny.registry";

export function loadRegistry(): RegistryEntry[] {
  try { return JSON.parse(localStorage.getItem(KEY) ?? "[]"); } catch { return []; }
}
export function saveRegistry(entries: RegistryEntry[]) {
  try { localStorage.setItem(KEY, JSON.stringify(entries)); } catch {}
}
export function newModelId(): string {
  return crypto.randomUUID();
}
