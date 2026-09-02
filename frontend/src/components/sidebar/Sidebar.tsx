// Sidebar: NodePalette + fusion/quant panels.
import { NodePalette } from "./NodePalette";
import { FusionPanel } from "./FusionPanel";
import { QuantizationPanel } from "./QuantizationPanel";

export function Sidebar() {
  return (
    <aside className="side">
      <NodePalette />
      <FusionPanel />
      <QuantizationPanel />
    </aside>
  );
}
