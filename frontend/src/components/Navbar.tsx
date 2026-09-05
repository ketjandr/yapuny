// Navbar: brand (returns to the Models home) + active project title (inline-editable) + status.
import { BrandButton } from "@/components/BrandButton";
import { useCanvasStore } from "@/store/canvasStore";
import { TITLE_MAX } from "@/lib/projects";
import { useProjectsStore } from "@/store/projectsStore";

export function Navbar() {
  const modelId = useCanvasStore((s) => s.modelId);
  const title = useProjectsStore(
    (s) => s.projects.find((p) => p.id === modelId)?.title ?? "untitled",
  );
  const rename = useProjectsStore((s) => s.rename);

  return (
    <header className="nav">
      <div className="brand">
        <BrandButton />
        <span className="proj">
          model{" "}
          {/* uncontrolled + keyed on modelId so it resets on project switch; commit on blur/Enter */}
          <input
            key={modelId}
            className="proj-title"
            defaultValue={title}
            size={Math.max(title.length, 6)}
            maxLength={TITLE_MAX}
            spellCheck={false}
            aria-label="Model title"
            onBlur={(e) => rename(modelId, e.target.value.trim() || "Untitled model")}
            onKeyDown={(e) => {
              if (e.key === "Enter") e.currentTarget.blur();
              if (e.key === "Escape") {
                e.currentTarget.value = title;
                e.currentTarget.blur();
              }
            }}
          />
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
