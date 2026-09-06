// Reusable pressable help affordance: a small "?" button that toggles a persistent popover
// (click-away or Escape closes it - no hovering needed). Styled to match the app tooltip (.tt).
// Drop <HelpDot label="..." text="..." /> next to any label that needs a plain-language explainer.
import { useEffect, useRef, useState } from "react";

export function HelpDot({ label, text }: { label: string; text: string }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <span className="help-dot" ref={ref}>
      <button
        type="button"
        className={`cfg-help${open ? " on" : ""}`}
        aria-label={`About ${label}`}
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        ?
      </button>
      {open && (
        <span className="help-pop" role="tooltip">
          {text}
        </span>
      )}
    </span>
  );
}
