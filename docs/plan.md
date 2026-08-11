# Project Plan — Agentic Customer Support v2 (pi SDK)

**Decision (owner):** rebuild on the **pi agents SDK** (`@earendil-works/pi-coding-agent`, Node/TypeScript) — in-process embedding (`createAgentSession`), Fastify instead of FastAPI. Full rebuild to the planned scope: SQL + vector + web retrieval, SSE/WebSocket streaming, task queue, guardrails.

---

## Phase 1 — Project setup migration (Python → TS)

| # | Item | Exit criteria |
|---|---|---|
| 1.1 | Scaffold TS project: `package.json`, `tsconfig.json`, `.env.example`, `src/` layout (runtime, tools, retrieval, streaming, server, queue, mcp, guardrails, config) | `npm install` clean; structure in place |
| 1.2 | Move legacy Python code to `legacy/` via `git mv` (preserve history) | `git status` shows renames; history intact |
| 1.3 | SDK spike: `createAgentSession` in-process + custom tool + event stream + restricted tools | spike script runs; tool called; events observed |
| 1.4 | Isolate SDK behind `src/runtime/`; pin `@earendil-works/pi-coding-agent` exact version | single module touches SDK; upgrade path documented |
| 1.5 | Baseline: `npm run typecheck` passes, README updated, initial commit | clean commit |

## Phase 2 — Scrap old implementation, extract learnings

| # | Item | Exit criteria |
|---|---|---|
| 2.1 | Remove old Python implementation from active tree (kept in `legacy/` + git history) | active tree clean of Python |
| 2.2 | Write `docs/learnings.md` — distill `docs/project-analysis.md` + git history: P0 `box`/`python-box` conflict, f-string SQL injection, sync DB engine in async tools, LiteLLM↔ADK streaming bug, dead guardrails, stub API/empty Dockerfile/missing data, print-based tests, plus what worked (router/sub-agent split, pgvector HNSW, async batch embeddings) | learnings doc complete |
| 2.3 | Update `docs/README.md` index (plan + learnings + research docs) | index current |

## Phase 3 — Full design planning

| # | Area | Items |
|---|---|---|
| 3.1 | Backend | **Framework evaluation: Fastify vs Express vs NestJS vs Hono (recommend one).** Fastify API: `POST /api/chat`, `GET /api/chat/:id/events` (SSE), WS endpoint, steer/cancel, session registry + persistence, BullMQ tasks, health |
| 3.2 | Agent loop | `createAgentSession` wiring, support-agent system prompt, sub-agents via `route_to_agent` custom tool (child sessions), guardrails via extension hooks (`input`, `context`, `tool_call`, `tool_result`), model config, event→SSE mapping |
| 3.3 | Retrieval | hybrid search (pgvector + Postgres FTS + RRF), reranker (Cohere/BGE), embeddings, structural chunking, ingest pipeline (PDF/CSV), tickets schema, read-only SQL policy |
| 3.4 | Data management | dataset selection (from `docs/data-research.md`), schema mapping, provisioning scripts, seed data |
| 3.5 | UI | **Framework evaluation: React+Vite (SPA) vs Next.js (recommend one, owner knows only React).** Chat interface, streaming display, sources/citations, ticket views, manual browser |
| 3.6 | Learning | Maintain `lessons.md` — owner-facing knowledge log; every framework/stack choice must include a beginner-level "why" entry |
| 3.7 | Output | `docs/design/backend-agent-retrieval.md`, `docs/design/data-management.md`, `docs/design/ui.md`, consolidated plan |

## Phase 4 — Parallel design agents (herdr)

| # | Item | Exit criteria |
|---|---|---|
| 4.1 | Close `project-analysis` + `stack-research` herdr agents (work delivered) | agents closed, panes freed |
| 4.2 | Keep `data-research` agent → extend task: **data-management design** → `docs/design/data-management.md` | doc written |
| 4.3 | Launch agent `design-backend-retrieval` → **backend framework eval + agent loop + retrieval design** → `docs/design/backend-agent-retrieval.md`; must append a beginner-friendly framework comparison to `lessons.md` | doc + lessons entry |
| 4.4 | Launch agent `design-ui` → **UI framework eval (React vs Next.js) + UI design** → `docs/design/ui.md`; must append a beginner-friendly framework comparison to `lessons.md` | doc + lessons entry |
| 4.5 | Consolidate designs → `docs/design/consolidated.md` (owner review) | consolidated plan |

## Deliverables so far (Phase 1–2 pre-work)

- `AGENTS.md` (repo root) — pi auto-loads this every session: locked stack decisions, architecture rules, workflow conventions, learning rule. Owner never repeats these.
- `lessons.md` (repo root) — learning log (Node/TS, pi SDK, retrieval modes, framework candidates, glossary).
- `docs/plan.md` — this file.

## Phase 5+ — Implementation (in progress — parallel worktree build)

- **5a DONE (parallel, 3 worktrees, integrated 2026-08-11):**
  - `retrieval-core`: schema (tickets/documents/document_chunks, HNSW+GIN), async pg pool, hybrid search (pgvector+FTS+RRF), chunker, ingest CLI → **8,469 suraj520 tickets + 282 chunks ingested, `npm run query` verified**
  - `agent-runtime`: `createSupportRuntime` (pi SDK, noTools:builtin), kb_search/tickets_query/web_search/route_to_agent tools, guardrails extension (input/tool_call/tool_result), `npm run chat` CLI
  - `api-streaming`: Fastify app (cors/sse/ws/rate-limit/pino), SSE+WS bridges, BullMQ scaffold, MCP scaffold
  - Integration: branches merged to main; mock runtime swapped for real; sources flow fixed (tool details → `agent_settled`); sql-tool real-schema + sources; `guardrail_blocked` SSE event; rate limiting live
- **5b NEXT:** UI (React+Vite SPA), BullMQ real ingest workers, MCP handlers, Dockerfile, vitest suite + eval, CFPB full-dump ingest (~2.9M rows)

---

## Cross-cutting principles

- **Isolate the SDK**: all `@earendil-works/pi-coding-agent` usage lives in `src/runtime/` so a version bump touches one module.
- **No secrets in code**: `.env.example` only; `DATABASE_URL`, `OPENAI_API_KEY`, `TAVILY_API_KEY`, etc. via env.
- **Guardrails by default**: input validation, SQL read-only allowlist, tool-call blocking — wired at the SDK interception layer.
- **Docs drive code**: every phase produces/updates a doc in `docs/`.
