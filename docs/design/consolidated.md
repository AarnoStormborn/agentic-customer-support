# Consolidated Design — v2 (for owner review)

Synthesis of the three parallel design docs. **Owner review checkpoint — decisions locked here feed Phase 5+ implementation.**

## Decisions (locked by design agents, pending owner approval)

| Area | Decision | Doc |
|---|---|---|
| Backend framework | **Fastify** — only candidate with first-class SSE plugin (`@fastify/sse`), 2–3× Express throughput, TS-native, plugin model matches our modules | `backend-agent-retrieval.md` §1 |
| Agent runtime | **pi SDK in-process** — `createAgentSession`, custom tools, extension hooks | `backend-agent-retrieval.md` §3 |
| Sub-agents | **`route_to_agent` custom tool** spawning child sessions (rag/sql/web, one tool each, disposed after use) | `backend-agent-retrieval.md` §3.2 |
| Guardrails | 4 extension hooks: `input` (block attacks), `context` (safety), `tool_call` (SQL SELECT-only allowlist), `tool_result` (PII scrub) | `backend-agent-retrieval.md` §3.3 |
| Real-time | SSE for events (`token`/`tool`/`turn`/`done`+sources/`error`), WebSocket for steer/cancel | `backend-agent-retrieval.md` §2.3 |
| Security | **No auth** (learner project) but **solid rate limiting required** — `@fastify/rate-limit` per-IP on chat/steer/cancel/tasks + per-session turn/token caps + connection caps | `backend-agent-retrieval.md` §2.5 |
| Retrieval | Hybrid: pgvector (cosine) + Postgres FTS (GIN tsvector) + **RRF**; rerank Cohere → BGE self-host optional | `backend-agent-retrieval.md` §4 |
| SQL safety | Read-only Postgres role + SELECT-only validation at `tool_call` | `backend-agent-retrieval.md` §4.6 |
| Queue | BullMQ jobs: ingestion, evals | `backend-agent-retrieval.md` §4.7 |
| UI framework | **React + Vite SPA** (not Next.js — no SSR need, backend already separate, lowest learning curve for owner) | `ui.md` §1 |
| UI state | **zustand** (typed stores: chat, session, tickets, manual, settings) | `ui.md` §3 |
| UI styling | Tailwind + minimal design tokens | `ui.md` §4 |
| Data bundle | **suraj520 tickets** (CC0) + **CFPB FULL dump ~2.9M rows** (owner: scale to large datasets; filter at ingest time) + verified manuals | `data-management.md` §1, §3.3 |
| Data model | `tickets` + `documents`/`document_chunks` (HNSW + GIN + btree) — **schema rebuilt from scratch, no v1 field carryover** | `data-management.md` §2 |
| Provisioning | Idempotent `scripts/provision-data.sh` (verified URLs), dry-run ingest, hash-based synthesis for missing fields | `data-management.md` §3 |

## API surface (summary)

- `POST /api/chat` → `{ sessionId }` · `GET /api/chat/:id/events` (SSE) · `POST /api/chat/:id/steer|cancel` · `WS /api/chat/:id`
- `GET /api/tickets?query=` · `GET /api/tickets/:id` · `GET /api/manuals` · `GET /api/manuals/:id` · `GET /health`
- SSE event types: `token` (text delta) · `tool` (start/end + name/args) · `turn` · `done` (final + sources[]) · `error`

## Implementation order (Phase 5+, pending approval)

1. **Retrieval layer** — pg pool, hybrid SQL, embeddings, ingest (needs DB up + data provisioned)
2. **Agent runtime** — session factory, 3 custom tools, `route_to_agent`, guardrails extension
3. **API + streaming** — Fastify routes, SSE/WS bridges over `session.subscribe()`
4. **Queue + MCP** — BullMQ workers, MCP server export of retrieval tools
5. **UI** — Vite+React SPA (chat, tickets, manuals, sources panel)
6. **Docker + tests + eval** — Dockerfile, vitest suite, RAGAS-style eval, data provisioning

## Open items (resolved — owner, 2026-08)

| Question | Decision |
|---|---|
| Auth for the UI | **None** — learner project; replace with solid rate limiting |
| CFPB size | **Full 1.4 GB dump / ~2.9M rows** — scale test for the retrieval system; filter at ingest |
| Ticket schema | **Rebuild from scratch** — nothing carried from v1 |
| Implementation order | Approved as proposed (retrieval → runtime → API → queue+MCP → UI → Docker/tests) |

## Docs map

- `docs/design/backend-agent-retrieval.md` (771 lines) — backend, agent loop, retrieval
- `docs/design/ui.md` (425 lines) — UI framework + screens + state
- `docs/design/data-management.md` (361 lines) — datasets, DDL, provisioning, ingest
- `lessons.md` §8–9 — why Fastify, React+Vite vs Next.js (owner learning)
