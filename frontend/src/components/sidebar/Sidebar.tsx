// Sidebar: a scrollable Node Registry (palette with its own search + collapsible groups) above a
// fixed-height Inspector (node / edge properties), so the split never jumps as the selection changes.
import { NodePalette } from "./NodePalette";
import { Inspector } from "./Inspector";

export function Sidebar() {
  return (
    <aside className="side">
      <section className="side-reg">
        <NodePalette />
      </section>
      <div className="side-insp">
        <Inspector />
      </div>
    </aside>
  );
}
