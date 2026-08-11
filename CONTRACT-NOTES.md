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
