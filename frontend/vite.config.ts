import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// dev only: proxy /api -> worker so there's no CORS locally.
// prod: the browser calls the worker URL directly (CORS on the worker).
const WORKER = process.env.WORKER_URL || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname, "src") } },
  server: { proxy: { "/api": { target: WORKER, changeOrigin: true } } },
});
