// Reusable dark-theme tooltip. Mount <TooltipLayer> once near the app root, then spread
// useTooltip(content) (from ./tooltipContext) onto any element. Portals to <body> so it never
// clips. While shown it re-reads the anchor's rect each frame, so it follows canvas zoom/pan;
// it also scales to the anchor's on-screen scale (so it tracks node size on zoom) and stays
// clamped inside the canvas (.cv). A 500ms dwell precedes it, and any press cancels it.
import { type ReactNode, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { TooltipCtx } from "./tooltipContext";

const DELAY = 500; // ms hover dwell before the tooltip appears

interface Active {
  content: ReactNode;
  el: Element;
}

export function TooltipLayer({ children }: { children: ReactNode }) {
  const [active, setActive] = useState<Active | null>(null);
  const tipRef = useRef<HTMLDivElement>(null);
  const timer = useRef<number | undefined>(undefined);

  const cancel = useCallback(() => {
    if (timer.current !== undefined) {
      clearTimeout(timer.current);
      timer.current = undefined;
    }
    setActive(null);
  }, []);
  const request = useCallback((content: ReactNode, el: Element) => {
    if (timer.current !== undefined) clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setActive({ content, el }), DELAY);
  }, []);
  const api = useMemo(() => ({ request, cancel }), [request, cancel]);

  // any press anywhere cancels a pending/shown tooltip - robustly covers node-drag start (capture
  // phase fires before React Flow's own drag handler, which swallows the React onMouseDown)
  useEffect(() => {
    const onDown = () => cancel();
    window.addEventListener("pointerdown", onDown, true);
    return () => window.removeEventListener("pointerdown", onDown, true);
  }, [cancel]);

  useLayoutEffect(() => {
    if (!active) return;
    let raf = 0;
    const place = () => {
      const tip = tipRef.current;
      if (tip) {
        const r = active.el.getBoundingClientRect();
        // on-screen scale of the anchor: offsetWidth ignores CSS transforms, so this is the canvas
        // zoom for a node and exactly 1 for an untransformed element (buttons, etc.)
        const ow = (active.el as HTMLElement).offsetWidth;
        const s = ow > 0 ? r.width / ow : 1;
        const cv = document.querySelector(".cv")?.getBoundingClientRect();
        const pad = 8;
        if (cv && (r.bottom < cv.top || r.top > cv.bottom || r.right < cv.left || r.left > cv.right)) {
          tip.style.visibility = "hidden";
        } else {
          tip.style.visibility = "visible";
          const halfW = (tip.offsetWidth * s) / 2;
          const h = tip.offsetHeight * s;
          const cx = r.left + r.width / 2;
          const left = cv ? Math.min(Math.max(cx, cv.left + pad + halfW), cv.right - pad - halfW) : cx;
          const top = cv ? Math.max(r.top - 13 * s, cv.top + pad + h) : r.top - 13 * s;
          tip.style.left = `${left}px`;
          tip.style.top = `${top}px`;
          // scale about the bottom-center (transform-origin in CSS), pinned just above the node
          tip.style.transform = `translate(-50%, -100%) scale(${s})`;
        }
      }
      raf = requestAnimationFrame(place);
    };
    place(); // position before first paint, then track
    return () => cancelAnimationFrame(raf);
  }, [active]);

  return (
    <TooltipCtx.Provider value={api}>
      {children}
      {active &&
        createPortal(
          <div ref={tipRef} className="tt">
            {active.content}
          </div>,
          document.body,
        )}
    </TooltipCtx.Provider>
  );
}
