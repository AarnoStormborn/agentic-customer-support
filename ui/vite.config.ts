import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Dev proxy: the Fastify API runs on :8000 (PORT=8000 in the backend .env).
// The UI never hard-codes CORS — same-origin /api requests are forwarded here.
// VITE_API_URL in ui/.env overrides the base for non-proxy deployments.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true },
      "/health": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
});
