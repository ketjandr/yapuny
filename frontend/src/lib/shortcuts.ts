// Declarative keyboard shortcuts: one window listener dispatches to a table of definitions.
// Add a shortcut by adding a row wherever the table is built - no new listener wiring.
import { useEffect } from "react";

export interface Shortcut {
  id: string;
  keys: string; // "mod+c" (mod = cmd/ctrl), "mod+shift+z", "Backspace", ...
  run: () => void;
  when?: () => boolean; // optional guard (e.g. only in a certain mode)
}

// exact match: "mod" = cmd or ctrl; unlisted modifiers must be absent
function matches(e: KeyboardEvent, keys: string): boolean {
  const parts = keys.toLowerCase().split("+");
  return (
    e.key.toLowerCase() === parts[parts.length - 1] &&
    parts.includes("mod") === (e.metaKey || e.ctrlKey) &&
    parts.includes("shift") === e.shiftKey &&
    parts.includes("alt") === e.altKey
  );
}

function isTyping(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  return !!el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable);
}

// Pass a stable (memoized) array; the first matching, allowed shortcut runs.
export function useShortcuts(shortcuts: Shortcut[]): void {
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (isTyping(e.target)) return; // never hijack typing
      const hit = shortcuts.find((s) => matches(e, s.keys) && (!s.when || s.when()));
      if (hit) {
        e.preventDefault();
        hit.run();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [shortcuts]);
}
