// App shell: nav + (sidebar, canvas, right dock).
import { Navbar } from "@/components/Navbar";
import { Canvas } from "@/components/canvas/Canvas";
import { Sidebar } from "@/components/sidebar/Sidebar";
import { ToastContainer } from "@/components/Toast";
import { TooltipLayer } from "@/components/Tooltip";
import { ConfigPanel } from "@/components/panels/ConfigPanel";
import { TrainingPanel } from "@/components/panels/TrainingPanel";
import { GenerationPanel } from "@/components/panels/GenerationPanel";
import { BenchmarkPanel } from "@/components/panels/BenchmarkPanel";

export default function App() {
  return (
    <TooltipLayer>
      <div className="app">
        <Navbar />
        <div className="body">
          <Sidebar />
          <Canvas />
          <aside className="dock">
            <ConfigPanel />
            <TrainingPanel />
            <GenerationPanel />
            <BenchmarkPanel />
          </aside>
        </div>
        <ToastContainer />
      </div>
    </TooltipLayer>
  );
}
