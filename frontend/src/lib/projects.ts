// Projects index (the Models page) + starter templates. A "project" is one canvas the user builds;
// its id doubles as the backend model id (model cache + weight locker key). The index (titles +
// timestamps) is stored separately from each project's canvas blob (lib/persist.ts).
import { DEFAULT_META } from "./defaultGraph";
import { blankToCanvas, seedToCanvas } from "./graph";
import { cleanEdges, cleanNodes, type PersistedCanvas } from "./persist";

export interface Project {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number; // last canvas edit (drives the "edited …" subtitle)
}

const PROJECTS_KEY = "yapuny.projects.v1";

export function newProjectId(): string {
  try {
    return crypto.randomUUID();
  } catch {
    return `m_${Date.now()}_${Math.random().toString(36).slice(2)}`;
  }
}

export function loadProjects(): Project[] {
  try {
    const raw = localStorage.getItem(PROJECTS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as { projects?: Project[] };
    return Array.isArray(parsed.projects) ? parsed.projects : [];
  } catch {
    return [];
  }
}

export function writeProjects(projects: Project[]): void {
  try {
    localStorage.setItem(PROJECTS_KEY, JSON.stringify({ projects }));
  } catch {
    /* ignore */
  }
}

// --- starter templates (blank + the unfused default; fused/quantized come later) ---
export type TemplateKey = "blank" | "unfused";

export interface Template {
  key: TemplateKey;
  label: string;
  desc: string;
}

export const TEMPLATES: Template[] = [
  { key: "blank", label: "Blank", desc: "An empty canvas — just the input and output endpoints." },
  { key: "unfused", label: "Unfused GPT", desc: "The default nanoGPT-style transformer, unfused." },
];

// build the initial persisted canvas for a template
export function templateCanvas(key: TemplateKey): PersistedCanvas {
  const unfused = key === "unfused";
  const { nodes, edges } = unfused ? seedToCanvas() : blankToCanvas();
  return {
    nodes: cleanNodes(nodes),
    edges: cleanEdges(edges),
    meta: DEFAULT_META,
    mode: "train",
    blockStart: unfused ? "ln1" : null,
    blockEnd: unfused ? "res2" : null,
    lastCompiled: null,
    viewport: null,
  };
}
