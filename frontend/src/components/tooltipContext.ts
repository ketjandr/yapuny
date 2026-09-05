// Tooltip context + hook (kept out of Tooltip.tsx so that file only exports a component).
// Spread useTooltip(content) onto any element: onMouseEnter requests a delayed show, onMouseMove
// reports the cursor (static anchors follow it), onMouseLeave cancels. The delay, drag-cancel,
// positioning and scaling all live in TooltipLayer.
import { createContext, type MouseEvent, type ReactNode, useContext, useMemo } from "react";

export interface TooltipApi {
  request: (content: ReactNode, el: Element, x: number, y: number) => void;
  move: (x: number, y: number) => void;
  cancel: () => void;
}

export const TooltipCtx = createContext<TooltipApi | null>(null);

export function useTooltip(content: ReactNode) {
  const api = useContext(TooltipCtx);
  return useMemo(() => {
    if (!api || content == null || content === "") return {};
    return {
      onMouseEnter: (e: MouseEvent) => api.request(content, e.currentTarget, e.clientX, e.clientY),
      onMouseMove: (e: MouseEvent) => api.move(e.clientX, e.clientY),
      onMouseLeave: api.cancel,
    };
  }, [api, content]);
}
