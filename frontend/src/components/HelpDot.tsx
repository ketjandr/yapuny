// Reusable pressable help affordance: a small "?" button that toggles a persistent popover
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

const POP_MAX_W = 220;

export function HelpDot({ label, text }: { label: string; text: string }) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState({ top: 0, left: 0 });
  const btnRef = useRef<HTMLButtonElement>(null);
  const popRef = useRef<HTMLSpanElement>(null);

  // anchor the fixed popover just below the button, clamped so it stays on-screen (left pane opens
  // rightward into the canvas; right pane gets pulled left off its edge)
  useLayoutEffect(() => {
    if (!open || !btnRef.current) return;
    const r = btnRef.current.getBoundingClientRect();
    const left = Math.max(8, Math.min(r.left, window.innerWidth - POP_MAX_W - 8));
    setPos({ top: r.bottom + 7, left });
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      const t = e.target as Node;
      if (btnRef.current?.contains(t) || popRef.current?.contains(t)) return;
      setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    // the popover is fixed, so any scroll/resize would leave it stranded - just close it
    const onMove = () => setOpen(false);
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    window.addEventListener("scroll", onMove, true);
    window.addEventListener("resize", onMove);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("scroll", onMove, true);
      window.removeEventListener("resize", onMove);
    };
  }, [open]);

  return (
    <span className="help-dot">
      <button
        ref={btnRef}
        type="button"
        className={`cfg-help${open ? " on" : ""}`}
        aria-label={`About ${label}`}
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        ?
      </button>
      {open &&
        createPortal(
          <span
            ref={popRef}
            className="help-pop fixed"
            role="tooltip"
            style={{ top: pos.top, left: pos.left }}
          >
            {text}
          </span>,
          document.body,
        )}
    </span>
  );
}
