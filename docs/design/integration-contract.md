# Integration Contract — Phase 5 parallel tracks

Three tracks build in parallel on **git worktrees** off `main`; I (orchestrator) integrate by
merging the three branches. This file is the contract every track codes against.

## Tracks

| Track (branch / worktree) | Agent (pane name) | Builds | Exports (contract) |
|---|---|---|---|
| `retrieval-core` → `.worktrees/retrieval-core` | `ingest-retrieval-core` | `src/db/`, `src/retrieval/`, `scripts/provision-*.sh` | retrieval API + schema + ingest CLI |
| `agent-runtime` → `.worktrees/agent-runtime` | `tools-agent-runtime` | `src/runtime/`, `src/agent/`, `src/tools/`, `src/guardrails/` | runtime API (session factory, tools, guardrails) + chat CLI |
| `api-streaming` → `.worktrees/api-streaming` | `server-api-streaming` | `src/server/`, `src/streaming/`, `src/queue/`, `src/mcp/` | Fastify app + SSE/WS bridges + queue + MCP server |

Each track may carry a **local copy of the interface it consumes + a mock implementation** so it
can develop and test independently. Integration replaces mocks with the real modules (same
signatures, verified by `tsc --noEmit` after merge).

## Cross-track contracts

### retrieval-core exports (consumed by agent-runtime tools & api-streaming read routes)

```ts
// src/retrieval/index.ts
export interface HybridResult {
  text: string;
  source: {
    type: "kb" | "sql";
    title?: string;
    docName?: string;
    sectionPath?: string;
    page?: number;
    url?: string | null;
    row?: Record<string, unknown>; // for sql results
  };
  score: number;
}
export interface HybridSearchOptions {
  query: string;
  topK?: number;            // default 5
  sourceTypes?: ("kb" | "sql")[]; // default ["kb","sql"]
  filter?: Record<string, unknown>;
}
export function searchHybrid(opts: HybridSearchOptions): Promise<{ results: HybridResult[]; queryTimeMs: number }>;
export function embedTexts(texts: string[]): Promise<number[][]>;
export function runSchema(): Promise<void>;                       // applies src/db/schema.sql
export function ingestTickets(source: "suraj520" | "cfpb" | "comcast"): Promise<IngestSummary>;
export function ingestManuals(dir: string): Promise<IngestSummary>;
// src/db/pool.ts
export function getPool(): import("pg").Pool;                     // DATABASE_URL
```

### agent-runtime exports (consumed by api-streaming)

```ts
// src/runtime/index.ts
export interface SupportRuntime {
  prompt(text: string, opts?: { images?: unknown[] }): Promise<void>;
  steer(text: string): Promise<void>;
  abort(): Promise<void>;
  subscribe(fn: (event: unknown) => void): () => void;   // pi SDK AgentSessionEvent
  getLastMessages(): unknown[];
  dispose(): void;
}
export function createSupportRuntime(opts?: {
  model?: string;                 // "provider/model" from PI_MODEL
  chatId?: string;
  sessionDir?: string;            // undefined = in-memory
}): Promise<SupportRuntime>;
// src/guardrails/extension.ts — extension factory used by the runtime
export function guardrailsExtension(pi: unknown): void;
```

### SSE event schema (api-streaming owns the bridge; runtime supplies raw SDK events)

Defined exactly in `docs/design/backend-agent-retrieval.md` §2.3. Event types:
`turn_start · token · tool_start · tool_end · turn_end · done (message + sources[]) · error (code)`
plus optional `thinking · retry_start · retry_end · queue_update`.
SDK→SSE mapping table lives in `api-streaming`'s `src/streaming/bridge.ts` (see design §3.4);
agent-runtime must NOT emit SSE — only raw SDK events via `subscribe()`.

### Env vars (all tracks)

From `.env.example` (each worktree copies it to `.env`):
`PI_MODEL · DATABASE_URL (postgresql://acs:acs@localhost:5432/acs) · REDIS_URL (redis://localhost:6379)
· OPENAI_API_KEY · EMBEDDING_MODEL (text-embedding-3-small) · TAVILY_API_KEY (optional) · COHERE_API_KEY (optional)
· PORT=8000 · HOST=0.0.0.0 · LOG_LEVEL=info`

### Data model

`retrieval-core` owns `src/db/schema.sql` per `docs/design/data-management.md` §2 (tickets +
documents + document_chunks; HNSW vector index, GIN tsvector, btree). All tracks treat it as
read-only reference until integration.

## Rules

1. Work **only** in your worktree dir (`/Users/harshsingh/Documents/personal/agentic-customer-support/.worktrees/<track>`).
   `cd` there first. Never touch the main checkout or other worktrees.
2. Commit on your branch (`retrieval-core` / `agent-runtime` / `api-streaming`) with clear messages.
   Do **not** merge or push.
3. `npm install` your own worktree deps. If you need a NEW package, add it to your worktree's
   package.json AND append one line to `DEPS.md` (create it) — orchestrator reconciles at integration.
4. Keep `tsc --noEmit` green before finishing.
5. Do not change files owned by another track (per the exports above). If a change to a contract
   is needed, note it in `DEPS.md` / a `CONTRACT-NOTES.md` instead of changing the interface.
6. Deliverables per track:

| Track | Deliverable |
|---|---|
| retrieval-core | schema.sql applied to the running Postgres; `npm run ingest` loads suraj520 tickets + 2–3 manuals (config/data/); `npm run query -- "lg tv wifi reset"` returns hybrid results; all retrieval tests pass |
| agent-runtime | `npm run chat` REPL runs the support agent (kb_search wired to real retrieval when available, else mock); route_to_agent + guardrails active; tools tested |
| api-streaming | `npm run dev` boots Fastify on :8000; /health 200; POST /api/chat creates a chat + SSE /api/chat/:id/events streams mock events end-to-end (real runtime after integration); BullMQ + MCP scaffolded; rate limiting registered |

## Verification at integration (orchestrator)

1. Merge `retrieval-core` → `agent-runtime` → `api-streaming` into `main` (in that order).
2. `npm install` (reconciled lockfile), `npm run typecheck` green.
3. Smoke: DB up → ingest → `npm run chat` answers from KB → `npm run dev` streams over SSE → UI (later phase).
