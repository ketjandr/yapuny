// Playground: the canvas editor for one project. The route wrapper validates the :id and keys the
// editor by it, so switching projects remounts with a clean load. The project's canvas is loaded
// synchronously on mount (before the canvas renders) to avoid a flash of the previous graph.
import { useState } from "react";
import { Navigate, useParams } from "react-router-dom";
import { Navbar } from "@/components/Navbar";
import { Canvas } from "@/components/canvas/Canvas";
import { LeftPane } from "@/components/panes/LeftPane";
import { RightPane } from "@/components/panes/RightPane";
import { useCanvasStore } from "@/store/canvasStore";
import { useProjectsStore } from "@/store/projectsStore";

export function PlaygroundRoute() {
  const { id } = useParams();
  const exists = useProjectsStore((s) => (id ? s.projects.some((p) => p.id === id) : false));
  if (!id || !exists) return <Navigate to="/" replace />;
  return <Playground key={id} id={id} />;
}

function Playground({ id }: { id: string }) {
  const loadProject = useCanvasStore((s) => s.loadProject);
  // load the project's canvas once, synchronously, before children render (keyed remount per id)
  useState(() => {
    loadProject(id);
    return null;
  });

  return (
    <div className="app">
      <Navbar />
      <div className="body">
        <LeftPane />
        <Canvas />
        <RightPane />
      </div>
    </div>
  );
}
