// App: client-side router (SPA). "/" is the Models home; "/m/:id" is the playground for one project.
// Routing is entirely client-side — one bundle, no server round-trips; the URL just mirrors which
// model is open, so deep-links and refreshes land on the right place.
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { ModelsPage } from "@/pages/ModelsPage";
import { PlaygroundRoute } from "@/pages/Playground";
import { ToastContainer } from "@/components/Toast";
import { TooltipLayer } from "@/components/Tooltip";

export default function App() {
  return (
    <BrowserRouter>
      <TooltipLayer>
        <Routes>
          <Route path="/" element={<ModelsPage />} />
          <Route path="/m/:id" element={<PlaygroundRoute />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
        <ToastContainer />
      </TooltipLayer>
    </BrowserRouter>
  );
}
