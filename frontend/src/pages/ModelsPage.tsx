// Models home: a grid of the user's projects.
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { GraphThumbnail } from "@/components/GraphThumbnail";
import { TEMPLATES, type Project, type TemplateKey } from "@/lib/projects";
import { useProjectsStore } from "@/store/projectsStore";

function relativeTime(ts: number): string {
  const s = Math.max(0, Math.floor((Date.now() - ts) / 1000));
  if (s < 45) return "just now";
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  if (d < 7) return `${d}d ago`;
  return new Date(ts).toLocaleDateString();
}

export function ModelsPage() {
  const projects = useProjectsStore((s) => s.projects);
  const create = useProjectsStore((s) => s.create);
  const navigate = useNavigate();
  const [picking, setPicking] = useState(false);

  const onPick = (t: TemplateKey) => {
    const id = create(t);
    setPicking(false);
    navigate(`/m/${id}`);
  };

  const sorted = [...projects].sort((a, b) => b.updatedAt - a.updatedAt);

  return (
    <div className="home">
      <header className="home-nav">
        <div className="brand">
          <span className="mark" />
          <span className="brand-name">Yapuny</span>
        </div>
      </header>

      <main className="home-main">
        <div className="home-head">
          <h1 className="home-title">Your models</h1>
          <button className="home-new" type="button" onClick={() => setPicking(true)}>
            + New model
          </button>
        </div>

        {sorted.length === 0 ? (
          <div className="home-empty">
            <p>No models yet.</p>
            <button className="home-new" type="button" onClick={() => setPicking(true)}>
              + Create your first model
            </button>
          </div>
        ) : (
          <div className="home-grid">
            {sorted.map((p) => (
              <ModelCard key={p.id} project={p} onOpen={() => navigate(`/m/${p.id}`)} />
            ))}
          </div>
        )}
      </main>

      {picking && <CreateModal onClose={() => setPicking(false)} onPick={onPick} />}
    </div>
  );
}

function ModelCard({ project, onOpen }: { project: Project; onOpen: () => void }) {
  const rename = useProjectsStore((s) => s.rename);
  const remove = useProjectsStore((s) => s.remove);
  const [confirming, setConfirming] = useState(false);

  return (
    <div className="mcard">
      <button className="mcard-preview" type="button" onClick={onOpen} title="Open model">
        <GraphThumbnail id={project.id} rev={project.updatedAt} />
      </button>
      <div className="mcard-foot">
        {/* uncontrolled so typing doesn't re-sort the grid mid-edit; committed on blur/Enter */}
        <input
          className="mcard-title"
          defaultValue={project.title}
          spellCheck={false}
          onBlur={(e) => rename(project.id, e.target.value.trim() || "Untitled model")}
          onKeyDown={(e) => {
            if (e.key === "Enter") e.currentTarget.blur();
          }}
        />
        <div className="mcard-sub">edited {relativeTime(project.updatedAt)}</div>
      </div>
      <button
        className="mcard-del"
        type="button"
        title="Delete model"
        onClick={() => setConfirming(true)}
        aria-label="Delete model"
      >
        ✕
      </button>
      {confirming && (
        <div className="mcard-confirm">
          <span>Delete this model?</span>
          <div className="mcard-confirm-row">
            <button type="button" className="danger" onClick={() => remove(project.id)}>
              Delete
            </button>
            <button type="button" onClick={() => setConfirming(false)}>
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function CreateModal({ onClose, onPick }: { onClose: () => void; onPick: (t: TemplateKey) => void }) {
  return (
    <div
      className="modal-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="modal">
        <div className="modal-head">
          <h2>New model</h2>
          <button className="modal-x" type="button" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>
        <p className="modal-sub">Start from a template</p>
        <div className="tpl-grid">
          {TEMPLATES.map((t) => (
            <button key={t.key} className="tpl" type="button" onClick={() => onPick(t.key)}>
              <span className="tpl-glyph" data-tpl={t.key} />
              <span className="tpl-label">{t.label}</span>
              <span className="tpl-desc">{t.desc}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
