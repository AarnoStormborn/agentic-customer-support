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
