// Request/response types mirrored from server/api/schemas.py. Keep in sync with the backend.

export interface NodeSchema {
  id: string;
  type: string;
  config?: Record<string, unknown>;
  quantized?: string | null; // "w8" | "w4" | null
}
export interface EdgeSchema {
  from_node: string;
  to_node: string;
  from_port?: string; // default "out"
  to_port?: string; // default "x"
}
export interface FusionGroupSchema {
  nodes: string[];
}
export interface BlockSchema {
  nodes: string[];
}
export interface GraphMetaSchema {
  n_layer: number;
  n_head: number;
  n_embd: number;
  block_size: number;
  dropout: number;
  vocab_size: number;
}
export interface GraphRequest {
  nodes: NodeSchema[];
  edges: EdgeSchema[];
  fusion_groups?: FusionGroupSchema[];
  block?: BlockSchema | null; // the repeated slice; backend unrolls it n_layer times
  meta?: Partial<GraphMetaSchema>;
}
export interface ModelGraphRequest {
  id: string;
  graph: GraphRequest;
}
export interface GenerateRequest {
  id: string;
  prompt: string;
  max_new_tokens?: number;
  temperature?: number;
  top_k?: number | null;
  bench?: boolean;
}
export interface TrainRequest {
  id: string;
  max_steps?: number;
  batch_size?: number;
  learning_rate?: number;
  eval_interval?: number;
  eval_iters?: number;
  bench?: boolean;
}
export interface BenchRunRequest {
  graphs: ModelGraphRequest[];
  prompt: string;
  max_new_tokens?: number;
  temperature?: number;
  top_k?: number | null;
}
export interface ProfileRequest {
  id: string;
  mode?: "decode" | "train";
  prompt_tokens?: number;
  new_tokens?: number;
  warmup?: number;
}

export type ModelStatus = "ready" | "needs_compile";

// frontend-owned model registry entry (localStorage)
export interface RegistryEntry {
  id: string;
  name: string;
  graph: GraphRequest;
  vocabSize: number;
  corpus?: string;
  createdAt: number;
  trainedHash?: string; // structure_hash last trained against, for local dirty hints
}
