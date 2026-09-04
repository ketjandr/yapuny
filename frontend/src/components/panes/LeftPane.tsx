// Left pane: a scrollable Node Registry (draggable palette with its own search + collapsible
// groups) over a fixed-height Inspector, so the split never jumps as the selection changes.
// Collapsible via the shared PaneShell (no header / no full view).
import { PaneShell } from "./PaneShell";
import { NodePalette } from "./NodePalette";
import { Inspector } from "./Inspector";

export function LeftPane() {
  return (
    <PaneShell side="left">
      <section className="pane-reg">
        <NodePalette />
      </section>
      <div className="pane-insp">
        <Inspector />
      </div>
    </PaneShell>
  );
}
