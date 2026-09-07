/// <reference types="vite/client" />

interface ImportMetaEnv {
  // the hosted shared worker (gateway) URL, baked in at build time; empty in dev
  readonly VITE_SHARED_WORKER_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
