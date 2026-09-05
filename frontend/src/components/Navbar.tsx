// Navbar: brand (returns to the Models home) + active project title + worker status.
import { BrandButton } from "@/components/BrandButton";
import { useCanvasStore } from "@/store/canvasStore";
import { useProjectsStore } from "@/store/projectsStore";

export function Navbar() {
  const modelId = useCanvasStore((s) => s.modelId);
  const title = useProjectsStore(
    (s) => s.projects.find((p) => p.id === modelId)?.title ?? "untitled",
  );

  return (
    <header className="nav">
      <div className="brand">
        <BrandButton />
        <span className="proj">
          model <b>{title}</b>
        </span>
      </div>
      <div className="nav-r">
        <div className="tele">
          <span className="d off" />
          worker: —
        </div>
        {/* TODO: Connect worker (URL + optional token), save/load, settings */}
        <button className="nl" type="button">
          connect
        </button>
      </div>
    </header>
  );
}
