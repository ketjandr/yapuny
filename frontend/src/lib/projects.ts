// Projects index (the Models page) + starter templates. A "project" is one canvas the user builds;
// its id doubles as the backend model id (model cache + weight locker key). The index (titles +
// timestamps) is stored separately from each project's canvas blob (lib/persist.ts).
import type { Edge, Node } from "@xyflow/react";
import { ABS_VARIANT, DEFAULT_META, OPT_VARIANT, type SeedVariant } from "./defaultGraph";
import { blankToCanvas, makeFusionEdge, seedToCanvas, type YNodeData } from "./graph";
import { cleanEdges, cleanNodes, type PersistedCanvas } from "./persist";

export interface Project {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number; // last canvas edit (drives the "edited …" subtitle)
}

const PROJECTS_KEY = "yapuny.projects.v1";

export const TITLE_MAX = 24; // model title character limit (shared by the navbar + projects page)

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

// --- starter templates ---
export type TemplateKey =
  | "blank"
  | "unfused"
  | "cached"
  | "fused"
  | "quantized"
  | "quantized_w4"
  | "fused_quant"
  | "optimized";

export interface Template {
  key: TemplateKey;
  label: string;
  desc: string;
}

export const TEMPLATES: Template[] = [
  { key: "blank", label: "Blank", desc: "An empty canvas with just the input and output endpoints." },
  { key: "unfused", label: "Unfused GPT", desc: "The default transformer without optimizations." },
  {
    key: "cached",
    label: "Cached GPT",
    desc: "Rotary positions + a rolling KV cache + flash attention for fast generation.",
  },
  { key: "fused", label: "Fused GPT", desc: "Every fusable op merged into a single Triton kernel." },
  { key: "quantized", label: "Quantized GPT (W8)", desc: "Every linear-weight node quantized to W8." },
  { key: "quantized_w4", label: "Quantized GPT (W4)", desc: "Every linear-weight node quantized to W4." },
  {
    key: "fused_quant",
    label: "Fused + Quantized",
    desc: "Fusable ops fused, the remaining linear weights quantized to W8.",
  },
  {
    key: "optimized",
    label: "Fully Optimized GPT",
    desc: "Everything on: rope, rolling KV cache, flash attention, fusion, and W8 quantization.",
  },
];

// fusable chains per seed variant (pipeline order), each mapping to one registry kernel. Picked to
// satisfy the fusion rule that no mid-chain node has a consumer outside the chain. Flash attention
// already fuses the score/mask/softmax chain, so the flash variant drops it.
const ABS_FUSION_CHAINS: string[][] = [
  ["attn", "mask", "smax"], // FusedScaleMaskSoftmax
  ["attn_drop", "res1"], // FusedDropoutResidual
  ["mlp_up", "gelu"], // FusedLinearGELU
  ["mlp_down", "mlp_drop", "res2"], // FusedLinearDropoutResidual
];
const FLASH_FUSION_CHAINS: string[][] = [
  ["attn_drop", "res1"],
  ["mlp_up", "gelu"],
  ["mlp_down", "mlp_drop", "res2"],
];

// seed ids of the quantizable linear-weight nodes (qkv/out proj, mlp up/down, lm head)
const QUANTIZABLE_IDS = ["qkv", "oproj", "mlp_up", "mlp_down", "lm_head"];

// bottom-diamond fusion edges wiring each chain into one connected fusion group
function fusionEdges(chains: string[][]): Edge[] {
  return chains.flatMap((chain) => chain.slice(1).map((to, i) => makeFusionEdge(chain[i], to)));
}

// tag the given nodes with a quantization mode, leaving the rest untouched
function quantize(nodes: Node[], ids: Set<string>, mode: string): Node[] {
  return nodes.map((n) =>
    ids.has(n.id) ? { ...n, data: { ...(n.data as YNodeData), quantized: mode } } : n,
  );
}

// build the initial persisted canvas for a template
export function templateCanvas(key: TemplateKey): PersistedCanvas {
  const gpt = key !== "blank";
  // cached + optimized use rope + rolling kv cache + flash attention; the rest are the absolute base
  const variant: SeedVariant = key === "cached" || key === "optimized" ? OPT_VARIANT : ABS_VARIANT;
  const { nodes: seedNodes, edges: seedEdges } = gpt ? seedToCanvas(variant) : blankToCanvas();

  const fuse = key === "fused" || key === "fused_quant" || key === "optimized";
  const chains = variant.flash ? FLASH_FUSION_CHAINS : ABS_FUSION_CHAINS;
  const fusedIds = new Set(chains.flat());
  const quantMode = key === "quantized_w4" ? "w4" : "w8";
  // fully quantized: every quantizable node; combined/optimized: only those not fused (a node can't be both)
  const quantIds =
    key === "quantized" || key === "quantized_w4"
      ? new Set(QUANTIZABLE_IDS)
      : key === "fused_quant" || key === "optimized"
        ? new Set(QUANTIZABLE_IDS.filter((id) => !fusedIds.has(id)))
        : new Set<string>();

  const nodes = quantIds.size ? quantize(seedNodes, quantIds, quantMode) : seedNodes;
  const edges = fuse ? [...seedEdges, ...fusionEdges(chains)] : seedEdges;

  return {
    nodes: cleanNodes(nodes),
    edges: cleanEdges(edges),
    meta: DEFAULT_META,
    // fusion, quantization and the kv cache all show their effect in inference mode (beams, W8
    // badges, the cache active instead of greyed), so open those templates there
    mode: fuse || quantIds.size || variant.cache ? "inference" : "train",
    blockStart: gpt ? "ln1" : null,
    blockEnd: gpt ? "res2" : null,
    lastCompiled: null,
    viewport: null,
  };
}
