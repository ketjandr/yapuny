// App shell: nav + (left pane, canvas, right pane).
import { Navbar } from "@/components/Navbar";
import { Canvas } from "@/components/canvas/Canvas";
import { LeftPane } from "@/components/panes/LeftPane";
import { ToastContainer } from "@/components/Toast";
import { TooltipLayer } from "@/components/Tooltip";
import { RightPane } from "@/components/panes/RightPane";

export default function App() {
  return (
    <TooltipLayer>
      <div className="app">
        <Navbar />
        <div className="body">
          <LeftPane />
          <Canvas />
          <RightPane />
        </div>
        <ToastContainer />
      </div>
    </TooltipLayer>
  );
}
