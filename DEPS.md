# DEPS — dependency reconciliation log (per integration-contract rule 3)

## retrieval-core (track 1)

- **New npm packages added: none.** All runtime deps (`pg`, `openai`, `pdf-parse`,
  `dotenv`) were already in `package.json` from Phase 1 scaffolding. `@types/pg`
  already present.
- Added npm scripts only: `db:migrate`, `query` (existing `ingest` reused).
- **Non-npm tool requirements (PATH):**
  - `python3` + `pyarrow` (suraj520 parquet → CSV; `scripts/convert-suraj520.py`).
    Installed on this machine (python3 3.11 + pyarrow 24.0.0). Orchestrator note:
    ingest auto-converts only when the CSV is missing — a committed CSV would
    remove the python requirement entirely.
- Reconciliation: merge the three tracks' `package.json` carefully; this track's
  additions are scripts only, no version conflicts expected.
# DEPS — dependency & env notes (agent-runtime track)

## npm dependencies

**No new npm packages added** — this track uses only packages already in
`package.json` (`@earendil-works/pi-coding-agent`, `typebox`, `dotenv`).
Web search (Tavily + DuckDuckGo) uses `fetch` (Node ≥ 22), no HTTP client dep.

## Env vars consumed (all optional, from `.env.example`)

| Var | Default | Used by |
|---|---|---|
| `PI_MODEL` | (first available) | supervisor model, "provider/id" or bare id |
| `PI_SPECIALIST_MODEL` | (cheapest available) | route_to_agent child model |
| `RETRIEVAL_MODE` | `mock` | `mock` = mock KB; `real` = load `RETRIEVAL_IMPL` |
| `RETRIEVAL_IMPL` | — | path to the real retrieval module (retrieval-core) |
| `SQL_MODE` | `mock` | `mock` = in-memory tickets; `real` = load `SQL_IMPL` |
| `SQL_IMPL` | — | path to a module exporting `getPool()` (retrieval-core's `src/db/pool.ts`) |
| `TAVILY_API_KEY` | (unset) | web search primary engine |
| `WEB_SEARCH_ENGINE` | `tavily` if key else `duckduckgo` | override web search engine |
| `DATABASE_URL` | — | only used in `SQL_MODE=real` via `SQL_IMPL` |

## Integration notes (orchestrator)

- At merge, retrieval-core's `src/retrieval/index.ts` replaces this track's local
  copy (same signatures). Set `RETRIEVAL_MODE=real` + `RETRIEVAL_IMPL=<path>` or
  just let the merged module take over (it bypasses the mock switch entirely).
- `SQL_MODE=real` + `SQL_IMPL=<path to src/db/pool.ts>` switches tickets_query to
  Postgres (read-only transaction, 1s statement timeout).
# DEPS.md — new dependencies per track (orchestrator reconciles at integration)

Track branches: `retrieval-core` · `agent-runtime` · `api-streaming`.

## api-streaming (this worktree)

| Package | Version | Added for | Used in |
|---|---|---|---|
| `@fastify/cors` | ^11.3.0 | CORS for the React SPA (UI phase) | `src/server/app.ts` |
| `@fastify/rate-limit` | ^11.2.0 | mandatory per-IP rate limiting (§2.5) | `src/server/app.ts` + chat/tasks routes |
| `@types/ws` (dev) | ^8 | WS socket types for `@fastify/websocket` handlers | `src/streaming/websocket.ts` |

Already present in the scaffold `package.json` and used as-is (no version changes):
`@fastify/sse` (SSE + replay), `@fastify/websocket` (WS channel), `bullmq` + `ioredis`
(task queue), `@modelcontextprotocol/sdk` + `zod` (MCP tools), `dotenv` (env loading).
No new top-level deps beyond the three above.

Note: `pino` logging is Fastify-native (bundled with `fastify`) — no separate dep.
