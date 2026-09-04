// Reusable collapsible pane shell, shared by both side panes (LeftPane + RightPane) so the
// collapse/expand chrome lives in exactly one place. It owns the collapse-to-rail state and the
// edge handle; the full-view overlay is opt-in (pass `onToggleExpand`). `expanded` stays controlled
// by the consumer so a pane's own content can react to the full-view layout (e.g. wider grids).
//
// `side` mirrors the chrome symmetrically: the collapse handle sits on the pane's inner edge and
// the reopen tab docks to the matching screen edge, with the chevrons pointing the way it travels.
import { type ReactNode, useState } from "react";
import { createPortal } from "react-dom";
import { useTooltip } from "@/components/tooltipContext";

interface PaneShellProps {
  side: "left" | "right";
  title?: ReactNode; // header eyebrow; a header renders when a title or the full-view toggle exists
  expanded?: boolean; // controlled full-view state; pair with onToggleExpand to enable full view
  onToggleExpand?: () => void;
  children: ReactNode; // body content (each pane defines its own inner layout)
}

export function PaneShell({ side, title, expanded = false, onToggleExpand, children }: PaneShellProps) {
  const [collapsed, setCollapsed] = useState(false);
  const collapseTip = useTooltip("Collapse view");
  const expandTip = useTooltip("Expand view");
  const expandable = onToggleExpand != null;

  // chevrons point the way the pane travels: left tucks left («) / reopens right (»); right mirrors
  const collapseChev = side === "left" ? "«" : "»";
  const reopenChev = side === "left" ? "»" : "«";

  // fully collapsed: only the reopen tab (fixed to the matching screen edge) remains
  if (collapsed) {
    return (
      <button
        type="button"
        className={`pane-handle reopen ${side}`}
        onClick={() => setCollapsed(false)}
        {...expandTip}
      >
        <span className="pane-chev">{reopenChev}</span>
      </button>
    );
  }

  const header = (title != null || expandable) && (
    <div className="pane-head">
      <span className="side-eyebrow pane-title">{title}</span>
      {expandable && (
        <div className="pane-tools">
          <button
            type="button"
            className={`pane-icon${expanded ? " close" : ""}`}
            title={expanded ? "Exit full view" : "Full view"}
            onClick={() => onToggleExpand?.()}
          >
            {expanded ? "✕" : "⤢"}
          </button>
        </div>
      )}
    </div>
  );

  // full view: portal the same header + body into a centered overlay (click the backdrop to exit)
  if (expandable && expanded) {
    return createPortal(
      <div
        className="pane-full"
        onClick={(e) => {
          if (e.target === e.currentTarget) onToggleExpand?.();
        }}
      >
        <div className="pane-full-inner">
          {header}
          {children}
        </div>
      </div>,
      document.body,
    );
  }

  return (
    <aside className={`pane ${side}`}>
      <button
        type="button"
        className={`pane-handle ${side}`}
        onClick={() => setCollapsed(true)}
        {...collapseTip}
      >
        <span className="pane-chev">{collapseChev}</span>
      </button>
      {header}
      {children}
    </aside>
  );
}
