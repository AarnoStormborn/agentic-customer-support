<<<<<<< HEAD
# CONTRACT-NOTES — retrieval-core track deviations & notes

Orchestrator + other tracks: read before integration. The **interface** in
`docs/design/integration-contract.md` is unchanged; these are implementation notes.

## 1. Embedding fallback (NO OPENAI_API_KEY found in env / .env / ~/.pi/agent/auth.json)

- `src/retrieval/embed.ts` uses the **OpenAI API when `OPENAI_API_KEY` is set**
  (batched 100, retry ×5 backoff on 429/5xx). Otherwise it emits **deterministic
  feature-hash embeddings** (FNV-1a hashing trick, word+bigram+trigram tokens,
  L2-normalized, dim 1536). Deterministic across runs/machines → ingest stays
  idempotent; cosine similarity approximates token overlap.
- All verification in this track used the fallback (`embeddingMode: "hash"`).
  To switch to real embeddings: set `OPENAI_API_KEY`, `npm run ingest -- --all`
  (upserts converge; HNSW index rebuild not needed at this scale).
- `EMBEDDING_MODEL=text-embedding-3-small` → `vector(1536)` pinned in schema.sql.
  Changing model → new dim → re-embed (data-management §6).

## 2. LG manual URL

The task-specified `https://gscs-b2c.lge.com/downloadFile?fileId=uzd9l7qf1exIqd8TH2Yvbg`
returned an **HTML error page, not a PDF** (verified 2026-08-11). Used the dustin.eu
retail mirror from docs/data-research.md §4 / data-management §3.2
(`lg_oled_55b9pla.pdf`, 770 KB, parses to 19 pages). Noted in
`scripts/provision-data.sh`.

## 3. searchHybrid merge semantics (kb + sql)

- `"kb"` = document_chunks: FTS (GIN tsvector) + pgvector cosine (HNSW) fused by
  RRF in one SQL statement (`1/(60+rank)`, k=60, 50 candidates each).
- `"sql"` = tickets FTS over `complaint_narrative || ticket_subject || product_purchased`,
  scored RRF-style `1/(60+rank)` for scale consistency with kb.
- When both sources enabled, each contributes `ceil(topK/2)` results; **kb results
  come first**, then sql. `topK` counts per source, not total (documented here, not
  in the contract — the agent/tools layer can re-order by score or per-source caps).
- `filter` currently allowlists `{ docName, section }` only.

## 4. Schema field names (data-management §2 canonical over backend-agent-retrieval §4.5)

Used `docs/design/data-management.md` §2 DDL exactly (per track instructions):
`tickets.status` (NOT `ticket_status`), `documents.file_path` unique,
`document_chunks.section / heading_path / page_start / page_end / chunk_text`.
`HybridResult.source` maps: `section`/`heading_path` → `sectionPath`, `page_start`
→ `page`, `chunk_text` → `text`. `url` is always `null` (documents table has no URL
column yet — add provenance URLs later if needed).

## 5. ingestTickets scope

Only `"suraj520"` implemented (parquet → CSV → map → upsert). `"cfpb"`/`"comcast"`
throw a clear NotImplemented error until their raw files are provisioned + mappers
written — the contract signature is preserved.

## 6. Extra exports beyond the contract

- `embedTexts` (contract) + `embeddingsEnabled()`, `embeddingDim()`, `EMBEDDING_MODEL`
  (helpers for agent tools / UI to display which backend is active).
- `closePool()` for CLI exit hygiene; `parsePdf`/`chunkDocument` exported for the
  BullMQ ingest worker (api-streaming phase).

## 7. Dependencies

No new npm packages added — `pg`, `openai`, `pdf-parse`, `dotenv` were already in
package.json. suraj520 parquet → CSV conversion uses `python3` + `pyarrow` (on PATH;
see DEPS.md).

## 8. Verification commands (all run in this worktree)

```
npm run db:migrate            # schema applied; vector@0.5.1 + pg_trgm@1.6
npm run ingest -- --all       # 8,469 tickets (idempotent upsert) + 3 manuals → 282 chunks
npm run query -- "lg tv wifi reset"   # hybrid results (see FINISH report)
```
=======
# CONTRACT-NOTES — deviations & integration notes (agent-runtime track)

Notes for the orchestrator and the api-streaming track. The public exports
match `docs/design/integration-contract.md`; everything here is additive.

## Export surface (contract-compliant)

```ts
// src/runtime/index.ts
createSupportRuntime(opts?: { model?, chatId?, sessionDir? }): Promise<SupportRuntime>
type SupportRuntime = { prompt, steer, abort, subscribe, getLastMessages, dispose }
// src/guardrails/extension.ts
guardrailsExtension(pi: unknown): void        // exact contract signature
supportGuardrails: InlineExtension             // named extension used by the runtime loader
```

- `SupportRuntimeImpl` (the concrete class) is exported from
  `src/runtime/session.ts` too — the chat CLI uses it. It adds one method beyond
  the contract: `promptWithBudget(text, budgetMs?)` (per-turn timeout + abort).
- `subscribe` emits **raw pi SDK `AgentSessionEvent`s** — agent-runtime emits no
  SSE; api-streaming's bridge owns the SDK→SSE mapping (design §3.4).

## Tool names (the four supervisor tools)

`kb_search` (rag) · `tickets_query` (sql) · `web_search` (web) · `route_to_agent`.
`route_to_agent` accepts `{ agent: "rag"|"sql"|"web", query }` and returns
`content` (specialist answer) + `details: { sources, childToolCalls, model,
turnCount, tokens }`. Guardrail hook validates `agent` before spawn.

## Decisions worth knowing

1. **`guardrailsExtension(pi: unknown)`** — internally casts to `ExtensionAPI`
   and shares the same factory as `supportGuardrails`. Kept `unknown` exactly as
   the contract states so api-streaming can import it without SDK types.
2. **`model?`/`PI_MODEL` matching is lenient** — accepts `"provider/id"` or a bare
   id (`"deepseek-v4-flash"` → `opencode-go/deepseek-v4-flash`); ambiguous bare
   ids throw. Falls back to a preferred list, then first available.
3. **Child sessions reuse the parent's `ModelRuntime`** — `createSupportRuntime`
   calls `configureRouteToAgent({ modelRuntime, supervisorModel })`; the
   `route_to_agent` tool lazily creates one only if used standalone.
4. **Context-hook safety note is a user-role message** — the SDK's in-session
   `Message` union has no `"system"` role (`UserMessage | AssistantMessage |
   ToolResultMessage`), so the note is prepended as a clearly-marked user
   message rather than a system message.
5. **Mock SQL is deliberately looser than real SQL** — `ILIKE '%lg tv%'` matches
   per-token ("LG OLED TV"), `'%wifi%'` matches "Wi-Fi" (hyphen-normalized).
   Real mode (`SQL_MODE=real`) runs real Postgres semantics; see DEPS.md.
6. **`sessionDir`** → `SessionManager.create(dir)` (JSONL); default in-memory.
7. **SDK event nuance** — session-level `turn_start` carries **no** `turnIndex`
   (only the extension-level `TurnStartEvent` does). Bridges mapping `turn_start`
   should not rely on `turnIndex` from `subscribe()` events.

## Untouched (owned by other tracks)

`src/db/`, `src/retrieval/` (real impl), `src/server/`, `src/streaming/`,
`src/queue/`, `src/mcp/`, `schema.sql`, provision scripts.
>>>>>>> agent-runtime
# CONTRACT-NOTES — deviations & integration notes (agent-runtime track)

Notes for the orchestrator and the api-streaming track. The public exports
match `docs/design/integration-contract.md`; everything here is additive.

## Export surface (contract-compliant)

```ts
// src/runtime/index.ts
createSupportRuntime(opts?: { model?, chatId?, sessionDir? }): Promise<SupportRuntime>
type SupportRuntime = { prompt, steer, abort, subscribe, getLastMessages, dispose }
// src/guardrails/extension.ts
guardrailsExtension(pi: unknown): void        // exact contract signature
supportGuardrails: InlineExtension             // named extension used by the runtime loader
```

- `SupportRuntimeImpl` (the concrete class) is exported from
  `src/runtime/session.ts` too — the chat CLI uses it. It adds one method beyond
  the contract: `promptWithBudget(text, budgetMs?)` (per-turn timeout + abort).
- `subscribe` emits **raw pi SDK `AgentSessionEvent`s** — agent-runtime emits no
  SSE; api-streaming's bridge owns the SDK→SSE mapping (design §3.4).

## Tool names (the four supervisor tools)

`kb_search` (rag) · `tickets_query` (sql) · `web_search` (web) · `route_to_agent`.
`route_to_agent` accepts `{ agent: "rag"|"sql"|"web", query }` and returns
`content` (specialist answer) + `details: { sources, childToolCalls, model,
turnCount, tokens }`. Guardrail hook validates `agent` before spawn.

## Decisions worth knowing

1. **`guardrailsExtension(pi: unknown)`** — internally casts to `ExtensionAPI`
   and shares the same factory as `supportGuardrails`. Kept `unknown` exactly as
   the contract states so api-streaming can import it without SDK types.
2. **`model?`/`PI_MODEL` matching is lenient** — accepts `"provider/id"` or a bare
   id (`"deepseek-v4-flash"` → `opencode-go/deepseek-v4-flash`); ambiguous bare
   ids throw. Falls back to a preferred list, then first available.
3. **Child sessions reuse the parent's `ModelRuntime`** — `createSupportRuntime`
   calls `configureRouteToAgent({ modelRuntime, supervisorModel })`; the
   `route_to_agent` tool lazily creates one only if used standalone.
4. **Context-hook safety note is a user-role message** — the SDK's in-session
   `Message` union has no `"system"` role (`UserMessage | AssistantMessage |
   ToolResultMessage`), so the note is prepended as a clearly-marked user
   message rather than a system message.
5. **Mock SQL is deliberately looser than real SQL** — `ILIKE '%lg tv%'` matches
   per-token ("LG OLED TV"), `'%wifi%'` matches "Wi-Fi" (hyphen-normalized).
   Real mode (`SQL_MODE=real`) runs real Postgres semantics; see DEPS.md.
6. **`sessionDir`** → `SessionManager.create(dir)` (JSONL); default in-memory.
7. **SDK event nuance** — session-level `turn_start` carries **no** `turnIndex`
   (only the extension-level `TurnStartEvent` does). Bridges mapping `turn_start`
   should not rely on `turnIndex` from `subscribe()` events.

## Untouched (owned by other tracks)

`src/db/`, `src/retrieval/` (real impl), `src/server/`, `src/streaming/`,
`src/queue/`, `src/mcp/`, `schema.sql`, provision scripts.
# CONTRACT-NOTES.md — api-streaming track deviations / integration notes

Notes for the orchestrator when merging `api-streaming` into `main`. Nothing here
changes the *interfaces* owned by other tracks — it documents where this track made
local choices so integration is mechanical.

## 1. Runtime import swap (the only real wiring change)

`src/server/routes/chat.ts` and `src/server/app.ts` import the runtime from
`../../runtime/mock.js` (this track's local mock, exports `createSupportRuntime` with
the contract signature). At integration, change that import to `../../runtime/index.js`
(the agent-runtime implementation). Same signature, verified by `tsc --noEmit`.

`buildApp({ createRuntime })` also accepts the factory as an option — zero-code swap.

## 2. Sources on `agent_settled` (bridge contract addition)

The bridge (`src/streaming/bridge.ts`) builds the `done` event's `sources[]` from
`event.sources` on the `agent_settled` SDK event, falling back to `[]`. The mock emits
`agent_settled { ..., sources: [...] }`. **The real runtime should also attach
`sources` to `agent_settled`** (its `tool_result` guardrail hook already collects them
per design §3.3). If the runtime prefers not to, the bridge can be switched to read
`session.getLastMessages()` — flagging here so it's a one-line change.

## 3. `src/config/env.ts` ownership

`src/config/` is not claimed by any track in the integration contract; this track
created `src/config/env.ts` (env parsing + limits). If another track also created it,
reconcile the two (keep all keys both sides read; none are conflicting).

## 4. `src/runtime/mock.ts` coexistence

The mock lives at `src/runtime/mock.ts` and does NOT create `src/runtime/index.ts`
(agent-runtime owns that file). Keep both; the import swap in note 1 is the only touch
point. The mock also re-declares the `SupportRuntime` interface locally so this track
typechecks standalone — delete the local copy after integration if desired.

## 5. Health check DB/Redis probes

`src/server/routes/health.ts` uses its own `pg` Pool + throwaway ioredis ping (both
with ~1.5s timeouts) because `src/db/pool.ts` is owned by retrieval-core. Integration
may switch to `getPool()` from `src/db/pool.ts` for a single shared pool.

## 6. Rate limiting specifics

- Global default: 30 req/min/IP (@fastify/rate-limit).
- Override to 10/min on: `POST /api/chat`, `POST /api/chat/:id/steer`,
  `POST /api/chat/:id/cancel`, `POST /api/tasks` (per §2.5).
- Disabled on: `GET /health`, SSE + WS routes. Long-lived sockets instead get a per-IP
  **connection cap** (default 5, `env.MAX_CONNECTIONS_PER_IP`) enforced in
  `src/streaming/limits.ts` (shared by SSE + WS). 429 on excess.

## 7. API shape deltas vs design §2.2

- `POST /api/chat` accepts both `conversationId` (§2.2) and `sessionId` (task spec) —
  `sessionId` is an alias. Response is `{ chatId, conversationId, eventsUrl, status }` (201).
- `POST /api/chat/:id/steer` returns `202 { queued: true }`; cancel returns
  `202 { cancelled: true }` (matching §2.2).
- Unknown chat id on steer/cancel/SSE → `404 { error: "chat_not_found" }`.

## 8. SSE stream lifecycle

- `sse: "only"` route; `X-Accel-Buffering: no` header set; plugin heartbeat 15s.
- On connect: `reply.sse.replay()` serves the registry ring buffer (last 200 events,
  `env.RING_BUFFER_SIZE`) after `Last-Event-ID`, then live subscription.
- Stream closes after a terminal `done` / `error` event.

## 9. Queue + MCP scaffolds

- `src/queue/jobs.ts`: `QUEUE_NAME="acs-tasks"`, job types
  `ingest.document | ingest.tickets | reembed`, `createTaskQueue()` + shared
  `redisConnection()` helper (ioredis with `maxRetriesPerRequest: null` — BullMQ rule).
- `src/queue/worker.ts`: `startWorker(logger, handlers?)` — stubs log + ack; real
  ingest handlers plug in at integration (retrieval-core exports
  `ingestTickets` / `ingestManuals`).
- `src/mcp/server.ts`: `buildMcpServer()` registers `kb_search` + `tickets_query` with
  zod input schemas; `src/mcp/index.ts` runs it over stdio (`npm run mcp`). Tool
  handlers return placeholders until integration.

## 10. Environment vars read by this track

`PORT`, `HOST`, `LOG_LEVEL`, `DATABASE_URL`, `REDIS_URL`, `PI_MODEL`,
`EMBEDDING_MODEL` (all in `.env.example`), plus optional tuning vars:
`RING_BUFFER_SIZE` (200), `MAX_CONNECTIONS_PER_IP` (5), `RATE_CHAT_MAX` (10),
`RATE_READ_MAX` (30), `TURN_BUDGET_MS` (120000). These extra tunables are additive;
no contract var is repurposed.
