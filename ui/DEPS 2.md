# DEPS.md — new dependencies (web-ui track)

The UI is a **separate npm project** in `ui/` (own `package.json` + lockfile),
so none of these touch the backend's `package.json`. Versions pinned in
`ui/package-lock.json`; `^` ranges shown for reference.

## Runtime

| Package | Version | Why |
|---|---|---|
| `react` / `react-dom` | ^19.2.8 | UI runtime (locked: React 19) |
| `vite` | ^7.3.6 | Dev server + bundler (locked: Vite 7.x per ui.md §1) |
| `@vitejs/plugin-react` | ^5.2.0 | React fast-refresh; 5.x is the line that supports Vite 7 (6.x requires Vite 8) |
| `react-router` | ^7.18.2 | Routing (v7 — import from `"react-router"`, not `react-router-dom`) |
| `zustand` | ^5.0.14 | Chat/session/settings stores + `persist` middleware |
| `tailwindcss` + `@tailwindcss/vite` | ^4.3.3 | CSS-first styling (v4 `@theme` in `src/styles/index.css`) |
| `react-markdown` | ^10.1.0 | Assistant message rendering (ESM-only — fine under Vite) |
| `remark-gfm` | ^4.0.1 | GFM tables/lists in markdown |
| `lucide-react` | ^1.31.0 | Icons (tree-shaken) |

## Dev

| Package | Version | Why |
|---|---|---|
| `typescript` | ~5.9.3 | Type checking (TS 7 native compiler exists but isn't default; ui.md pins 5.9) |
| `vitest` | ^4.1.10 | Unit tests (matches backend's vitest major) |
| `@testing-library/react` | ^16.3.2 | Component tests |
| `happy-dom` | ^20.11.2 | DOM environment for vitest (lighter than jsdom) |
| `@types/react` / `@types/react-dom` / `@types/node` | 19.x / 19.x / 22.x | Types |

## Explicitly NOT added

- `eventsource-parser` — only needed if EventSource is replaced by a fetch
  stream (auth headers); the wrapper is designed to hide that swap
  (`ui/src/lib/sse.ts`).
- `@tanstack/react-query` — ui.md §7: deferred; plain `fetch` in stores suffices
  for v1.
- `@tailwindcss/forms`, `prettier`, `eslint` — not required by the phase spec.

## Reconciliation notes for the orchestrator

- The UI needs the backend on **PORT 8000** (see `ui/vite.config.ts` proxy).
- `ui/` has its own `tsconfig.json` + `vitest.config.ts`; root `tsc`/`vitest`
  configs include `src/**` and `tests/**` only, so the UI never leaks into
  backend builds or the backend's test suite.
- `.env` (ui) holds only the optional `VITE_API_URL`; it is gitignored.
