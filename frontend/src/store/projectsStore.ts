// Projects store: the list shown on the Models page. Create writes a fresh canvas blob from a
// template; delete removes the local canvas + index entry AND the backend weight package (store.py)
// via the /model DELETE api; touch bumps the "last edited" timestamp (called by the canvas autosave).
import { create } from "zustand";
import { api } from "@/lib/api";
import { deleteCanvas, loadCanvas, writeCanvas } from "@/lib/persist";
import {
  loadProjects,
  newProjectId,
  type Project,
  type TemplateKey,
  templateCanvas,
  writeProjects,
} from "@/lib/projects";

interface ProjectsState {
  projects: Project[];
  create: (template: TemplateKey) => string; // returns the new project id
  duplicate: (id: string) => string | null; // copy a project's canvas to a new id; returns it
  remove: (id: string) => void;
  rename: (id: string, title: string) => void;
  touch: (id: string) => void;
}

// persist the list on every mutation, keeping the store the single writer of the index
function save(projects: Project[]): Project[] {
  writeProjects(projects);
  return projects;
}

export const useProjectsStore = create<ProjectsState>((set, get) => ({
  projects: loadProjects(),

  create: (template) => {
    const id = newProjectId();
    writeCanvas(id, templateCanvas(template)); // canvas exists before the playground loads it
    const now = Date.now();
    const project: Project = { id, title: "Untitled model", createdAt: now, updatedAt: now };
    set((s) => ({ projects: save([project, ...s.projects]) }));
    return id;
  },

  // copy the source project's canvas (the design) to a fresh id. The backend model is keyed by id,
  // so the copy starts uncompiled / untrained - it's a duplicate of the graph, not the weights.
  duplicate: (id) => {
    const src = loadCanvas(id);
    if (!src) return null; // no canvas to copy (shouldn't happen for a real project)
    const newId = newProjectId();
    writeCanvas(newId, src); // loadCanvas returns a fresh parse, so no aliasing with the original
    const now = Date.now();
    const title = `${get().projects.find((p) => p.id === id)?.title ?? "Untitled model"} copy`;
    const project: Project = { id: newId, title, createdAt: now, updatedAt: now };
    set((s) => ({ projects: save([project, ...s.projects]) }));
    return newId;
  },

  remove: (id) => {
    set((s) => ({ projects: save(s.projects.filter((p) => p.id !== id)) }));
    deleteCanvas(id);
    api.deleteModel(id).catch(() => {}); // best-effort: the package may not exist (never trained)
  },

  // rename does not bump updatedAt — "last edited" tracks canvas edits, not title changes
  rename: (id, title) =>
    set((s) => ({ projects: save(s.projects.map((p) => (p.id === id ? { ...p, title } : p))) })),

  touch: (id) =>
    set((s) => ({
      projects: save(s.projects.map((p) => (p.id === id ? { ...p, updatedAt: Date.now() } : p))),
    })),
}));
