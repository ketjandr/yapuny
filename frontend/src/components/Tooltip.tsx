// Reusable dark-theme tooltip. Mount <TooltipLayer> once near the app root, then spread
// useTooltip(content) (from ./tooltipContext) onto any element. Portals to <body> so it never
// clips. Two placement modes, chosen by the anchor:
//  - nodes: anchored to the node rect, re-read each frame so it follows canvas zoom/pan, scaled to
//    the node's on-screen size, pinned above (flips below when there's no room), clamped in .cv.
//  - everything else (toolbar buttons, pane handles): OS-style, follows the cursor and sits
//    below-right of it, flipping near the viewport edges to stay on screen.
// A 500ms dwell precedes it, and any press cancels it.
import { type ReactNode, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { TooltipCtx } from "./tooltipContext";

const DELAY = 500; // ms hover dwell before the tooltip appears

interface Active {
  content: ReactNode;
  el: Element;
  cursor: boolean; // static anchor -> place at the cursor; node -> anchored to the node rect
  at: { x: number; y: number }; // cursor pos snapshotted when the tooltip appeared (cursor mode)
}

export function TooltipLayer({ children }: { children: ReactNode }) {
  const [active, setActive] = useState<Active | null>(null);
  const tipRef = useRef<HTMLDivElement>(null);
  const timer = useRef<number | undefined>(undefined);
  const pointer = useRef({ x: 0, y: 0 }); // latest cursor pos; snapshotted into `at` when shown

  const cancel = useCallback(() => {
    if (timer.current !== undefined) {
      clearTimeout(timer.current);
      timer.current = undefined;
    }
    setActive(null);
  }, []);
  const request = useCallback((content: ReactNode, el: Element, x: number, y: number) => {
    pointer.current = { x, y };
    if (timer.current !== undefined) clearTimeout(timer.current);
    // nodes anchor to their rect (scaled with zoom); everything else places at the cursor. The
    // cursor pos is snapshotted at show time (tracked through the dwell), then held - it doesn't
    // follow the cursor in real time, so the next hover recomputes it fresh.
    const cursor = !el.closest(".react-flow__node");
    timer.current = window.setTimeout(
      () => setActive({ content, el, cursor, at: { ...pointer.current } }),
      DELAY,
    );
  }, []);
  const move = useCallback((x: number, y: number) => {
    pointer.current = { x, y };
  }, []);
  const api = useMemo(() => ({ request, move, cancel }), [request, move, cancel]);

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
    const pad = 8;

    // OS-style tooltip for static anchors: placed once, below-right of where the cursor was when it
    // appeared, flipping left/up near the viewport edges so it always stays on screen.
    const placeAtCursor = (tip: HTMLDivElement) => {
      const { x, y } = active.at;
      const w = tip.offsetWidth;
      const h = tip.offsetHeight;
      const offX = 14;
      const offY = 18;
      let left = x + offX;
      if (left + w > window.innerWidth - pad) left = x - offX - w; // flip to the cursor's left
      left = Math.max(pad, Math.min(left, window.innerWidth - pad - w));
      let top = y + offY;
      if (top + h > window.innerHeight - pad) top = y - offY - h; // flip above the cursor
      top = Math.max(pad, Math.min(top, window.innerHeight - pad - h));
      tip.style.visibility = "visible";
      tip.style.left = `${left}px`;
      tip.style.top = `${top}px`;
      tip.style.transform = "none";
    };

    const place = () => {
      const tip = tipRef.current;
      // cursor mode: place once at the snapshotted position and stop (no live follow)
      if (tip && active.cursor) {
        placeAtCursor(tip);
        return;
      }
      if (tip) {
        // anchored mode (nodes only): pin above the node rect and scale with the canvas zoom
        const r = active.el.getBoundingClientRect();
        // on-screen scale of the anchor: offsetWidth ignores CSS transforms, so this is the zoom
        const ow = (active.el as HTMLElement).offsetWidth;
        const s = ow > 0 ? r.width / ow : 1;
        const cv = document.querySelector(".cv")?.getBoundingClientRect();
        if (cv && (r.bottom < cv.top || r.top > cv.bottom || r.right < cv.left || r.left > cv.right)) {
          tip.style.visibility = "hidden";
        } else {
          tip.style.visibility = "visible";
          const halfW = (tip.offsetWidth * s) / 2;
          const h = tip.offsetHeight * s;
          const gap = 13 * s;
          const cx = r.left + r.width / 2;
          const left = cv ? Math.min(Math.max(cx, cv.left + pad + halfW), cv.right - pad - halfW) : cx;
          tip.style.left = `${left}px`;
          // prefer above the anchor (the good node behavior); flip below only when there isn't room
          // above within the canvas (e.g. the top-left toolbar buttons), so it never clips/overlaps
          const roomAbove = !cv || r.top - gap - h >= cv.top + pad;
          if (roomAbove) {
            tip.style.top = `${r.top - gap}px`;
            tip.style.transformOrigin = "50% 100%"; // scale about bottom-center, pinned above
            tip.style.transform = `translate(-50%, -100%) scale(${s})`;
          } else {
            tip.style.top = `${r.bottom + gap}px`;
            tip.style.transformOrigin = "50% 0"; // scale about top-center, pinned below
            tip.style.transform = `translate(-50%, 0) scale(${s})`;
          }
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
