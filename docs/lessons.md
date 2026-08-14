# lessons.md — Learning Log

> Beginner-friendly knowledge log for the owner (knows React, learning the rest of the v2 stack).
> Each entry: what it is, why we picked it, how it works in ~5 minutes, tiny example.

---

## 1. Why the rebuild is TypeScript, not Python

The old project was Python (Google ADK + LiteLLM + FastAPI-planned). The v2 rebuild runs on the
**pi agents SDK**, which is a **Node.js / TypeScript library** — pi (this assistant) is itself a
Node CLI. If the agent runtime is pi, the whole app naturally becomes a Node project.

**Mental model (Python → Node):**
| Python | Node/TS equivalent |
|---|---|
| `pip` + `requirements.txt` / `uv` | `npm` + `package.json` |
| `venv` | `node_modules` (no venv needed) |
| `async def` / `await` | `async` / `await` (same idea) |
| `fastapi` | `fastify` (or Express/NestJS — TBD, see §4) |
| type hints (optional) | TypeScript types (enforced at compile time) |
| `python main.py` | `npm run dev` (runs `tsx src/...`) |

**Why TypeScript and not plain JS:** types catch bugs before runtime — the compiler (like a
linter + type checker in one) validates that you never pass a string where a number is expected.
The pi SDK ships its own types, so you get autocomplete and compile-time checks for free.

---

## 2. What the pi agents SDK actually is

- It's the same library that powers this coding assistant, exposed as a **library** you can
  `import` in your own Node app.
- Core idea: `createAgentSession()` gives you an **agent session** — a model + tools + message
  history. You `prompt()` it, it calls tools, streams events.
- You define **custom tools** with `defineTool()` (name, description, JSON-schema parameters,
  an `execute` function). The model decides when to call them.

```ts
import { createAgentSession, defineTool, ModelRuntime } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

const myTool = defineTool({
  name: "echo",
  description: "Echo text",
  parameters: Type.Object({ text: Type.String() }),
  execute: async (_id, { text }) => ({ content: [{ type: "text", text }], details: {} }),
});

const runtime = await ModelRuntime.create();
const { session } = await createAgentSession({ customTools: [myTool], modelRuntime: runtime });
session.subscribe((e) => { /* stream tokens / tool events here */ });
await session.prompt("Say hello");
```

**Key facts:**
- Sessions are isolated: in-memory or JSONL files (tree branching).
- Streaming events: `message_update` (token deltas), `tool_execution_start/end`, `turn_*`, `agent_*`.
- Model-agnostic: uses your existing `~/.pi/agent/auth.json` credentials (15+ providers).
- **No native multi-agent**: no handoffs like ADK's `transfer_to_agent`. We build sub-agents as a
  `route_to_agent` **custom tool** that spawns a child session with a specialist prompt + one tool.
- **MCP**: pi *consumes* MCP servers (via `pi-mcp-adapter`) but doesn't natively serve as one.
  We'll export our retrieval tools as a standalone MCP server instead.

---

## 3. The three retrieval modes (the core of this project)

| Mode | What it answers | Store | Technique |
|---|---|---|---|
| SQL | "What's the status of ticket #42?" | Postgres `tickets` table | LLM writes SELECT, we execute read-only |
| Vector | "How do I reset my LG TV Wi-Fi?" | pgvector over manual chunks | embed query → nearest neighbors |
| Web | "Is this a known issue in 2026?" | live internet | Tavily/DuckDuckGo API |

v2 upgrades vector to **hybrid search**: keyword (Postgres full-text search) + semantic (pgvector),
fused with **RRF** (Reciprocal Rank Fusion — just combines two rankings by their positions), then
optionally **reranks** with a cross-encoder model (Cohere or BGE) for precision. Cross-encoder =
looks at query+result *together* (slow but accurate), vs bi-encoder (embeddings) which encodes
them separately (fast but coarser).

---

## 4. Backend framework candidates (decision in progress)

Fastify, Express, NestJS, Hono — all Node web frameworks; you pick one and add routes/handlers.

| | Express | **Fastify** | NestJS | Hono |
|---|---|---|---|---|
| Style | Minimal, most popular, oldest | Minimal + fast (JSON schema validation) | Structured (DI, decorators, Angular-like) | Ultra-light, edge-friendly |
| Learning curve | Low | Low–medium | High | Low |
| Good for | Anything | API servers, streaming | Big teams, enterprise | Edge/workers, minimal APIs |
| Streaming/SSE | manual | `@fastify/sse` plugin | manual/modules | built-in helpers |

**Current lean (to confirm in design):** **Fastify** — fast, first-class SSE plugin, TS-friendly,
good for the real-time chat API we need. NestJS is overkill for a learning project; Hono is great
but less mainstream.

**Decision (design doc §1, `docs/design/backend-agent-retrieval.md`): Fastify.** Beginner-friendly
"why" is in §8 below.

---

## 5. UI framework candidates (decision in progress)

You know React. The real question is **how the app is served**:

| | React + Vite (SPA) | Next.js |
|---|---|---|
| What | Pure client app; a separate API server (Node) | Full-stack framework (API routes + React pages in one) |
| Data fetching | fetch/axios to your API | Server Components / RSC, server actions |
| When to pick | API already separate; you want to learn React properly | You want one deployable app; SEO; less plumbing |
| Downside | You manage two apps (API + UI) | More magic; bigger learning surface |

**Current lean (to confirm in design):** depends on how we ship the backend. If Fastify API is
separate, **React + Vite SPA** keeps a clean line. If we want one deployable thing, **Next.js**.
The UI design agent will make a concrete recommendation (Phase 4.4).

---

## 6. Glossary (quick lookup)

- **ESM** (`"type": "module"`) — modern JS module system; `import`/`export` instead of `require`.
- **tsx** — runs TypeScript directly (no build step) during dev.
- **SSE** — Server-Sent Events: server pushes to browser over one HTTP connection (one-way). Simpler than WebSocket.
- **WebSocket** — two-way persistent connection (chat needs this for e.g. steer/cancel).
- **pgvector** — Postgres extension adding vector columns + similarity search (HNSW index).
- **HNSW** — index structure for fast approximate nearest-neighbor search.
- **RRF** — Reciprocal Rank Fusion: merge two ranked lists by position (`1/(k+rank)`).
- **BullMQ** — Redis-backed job queue (like Celery in Python).
- **Guardrails** — validation layers that block unsafe input/output before the model or tools act.

---

## 7. Log (append as the project moves)

- **[v2 kickoff]** Chose pi SDK in-process (option A) → Node/TS project. Fastify + React/Next.js
  pending design validation. `AGENTS.md` created so rules persist across sessions
  (removed 2026-08 by owner; the decisions live in `docs/plan.md` + `docs/design/consolidated.md`).
- **[backend design]** Phase 3.1–3.3 design landed: **Fastify** chosen over Express/NestJS/Hono
  (SSE plugin + speed + TS). Agent loop: `route_to_agent` custom tool spawning child sessions;
  guardrails via `input`/`context`/`tool_call`/`tool_result` hooks. Retrieval: pgvector `<=>` +
  tsvector GIN + RRF, Cohere rerank, structural chunking, read-only SQL role.
- **[retrieval-core impl]** Phase 5 foundation built: `src/db/` (schema + async pool) and
  `src/retrieval/` (embed / chunk / hybrid / ingest / query CLI) landed. 8,469 suraj520 tickets +
  3 manuals (282 chunks) ingested; `npm run query` works end-to-end. OpenAPI key absent →
  deterministic hash-embedding fallback (see §10.6, CONTRACT-NOTES.md).

---

## 8. Retrieval-core implementation lessons (what you just learned by building it)

### 10.1 Reciprocal Rank Fusion (RRF) — fuse ranks, not scores

Dense (vector) and lexical (keyword) retrievers return **incomparable scores**: cosine
similarity is ~0–1, BM25/ts_rank is unbounded. RRF ignores the scores entirely and fuses
**rank positions**: `score = 1/(k + rank)` for each list, then sums per document. `k=60`
is the standard smoothing constant. A document ranked #1 by both retrievers gets
`1/61 + 1/61 ≈ 0.033`; one ranked #1 by only one gets `1/61 ≈ 0.016`. It's robust to
outliers (a wild cosine score can't dominate) and needs zero calibration:

```sql
-- two ranked lists, fused:
SELECT id, COALESCE(1.0/(60+fts.rank),0) + COALESCE(1.0/(60+vec.rank),0) AS score
FROM fts FULL OUTER JOIN vec USING (id);
```

### 10.2 Why we need BOTH a GIN and an HNSW index

| | GIN (tsvector) | HNSW (pgvector) |
|---|---|---|
| Finds | exact words/prefixes | semantically similar vectors |
| Index type | inverted list of tokens | approximate nearest-neighbour graph |
| Great at | model numbers, ticket IDs, "wi-fi" | "TV won't turn on" even if the manual says "blank screen" |
| Weak at | synonyms, typos, paraphrase | exact IDs, rare tokens |

Hybrid search runs both in ONE query and RRF-fuses the candidates — each index covers the
other's blind spot (this is the whole point of §4.1 in the design).

### 10.3 Cosine distance `<=>` vs inner product `<#>` (v1 bug)

v1 ranked by negative inner product `<#>`: `similarity = -a·b`, which is **magnitude-
sensitive** — a long text embedding scores higher than a short one even when less relevant.
pgvector's cosine distance `embedding <=> query` normalizes for you; the HNSW index must
match: `USING hnsw (embedding vector_cosine_ops)`.

### 10.4 Parameterized SQL — why the v1 injection bug can't come back

v1 did `f"LIMIT {top_k}"` (f-string interpolation into SQL — the same class of bug as SQL
injection). In v2 every value travels as a bound parameter: the SQL text is a static
string, values go in `$1, $2, …` and are passed separately to the driver:

```ts
await pool.query("SELECT … WHERE x @@ websearch_to_tsquery('english', $1) LIMIT $2", [query, 10]);
```

`websearch_to_tsquery` also *parses* user input into a tsquery safely (AND by default,
quotes for phrases) — no string-built query text anywhere. (This also means the user query
`!!!` produces an empty tsquery → the FTS branch just returns nothing instead of erroring.)

### 10.5 `pg.Pool` vs a sync engine inside async code (v1 lesson #3)

v1 created a *synchronous* SQLAlchemy engine inside async tools — each query blocked the
entire event loop. `pg.Pool` is async: you `await pool.query(...)`, the pool hands the query
to a free connection and the event loop keeps serving other work meanwhile. One module-scope
singleton (`getPool()` in `src/db/pool.ts`) is shared by ingest + search + migrate, and
`pool.on('error')` stops an idle-connection failure from crashing the process silently.

### 10.6 Deterministic hash embeddings (the no-API-key fallback)

No OpenAI key was available in this environment, so `embed.ts` falls back to a
**feature-hashing** embedding: split text into word/bigram/trigram tokens, hash each token
to a vector index with a random sign (FNV-1a), accumulate, L2-normalize. It's deterministic
across runs/machines (idempotent ingest) and cosine similarity roughly tracks token overlap
— enough to run the whole pipeline and demo hybrid search without paying for embeddings.
Real semantic quality requires the OpenAI key (set `OPENAI_API_KEY`, re-run ingest; upserts
converge).

### 10.7 What we learned from the running system (pg quirks)

- **`numeric` columns come back as strings** from node-pg — our RRF score `1/(60+rank)` is
  numeric, so `r.score.toFixed()` exploded until we cast `::float8` in SQL.
- **`BIGSERIAL` ids are strings too** (`row_number()` → bigint) — `60 + rank` was string
  concatenation server-side before the cast.
- **Idempotent upsert trick**: `INSERT … ON CONFLICT … RETURNING (xmax = 0) AS inserted` —
  `xmax` is 0 for freshly-inserted rows, non-zero for updated ones → you get insert/update
  counts for free.
- **Batching without string-building**: `INSERT … SELECT * FROM unnest($1::text[], …)`
  lets you pass 500 rows as parallel arrays in one statement (pg's 65535-param limit →
  ~4,000+ rows per batch at 15 cols).
- **`vector(n)` is dimension-pinned**: passing a 2-dim vector into a 1536-dim column is a
  hard error (`different vector dimensions 1536 and 2`) — always check the model's dim.

---

## 9. Why Fastify won (backend framework decision)

A web framework is the Node equivalent of Flask/FastAPI in Python: it routes URLs to handler
functions and gives you middleware (code that runs on every request). We compared four:

| | Express | **Fastify** | NestJS | Hono |
|---|---|---|---|---|
| Age/style | 2010 default, minimal | Fast, modern, built-in validation | Angular-style, strict | Ultra-light, edge-first |
| Speed | 1× baseline | **2–3× Express** [betterstack] | ~Express speed | fast, cold-start optimized |
| SSE streaming | hand-rolled | **official plugin** | manual | built-in, edge-focused |
| TypeScript | bolt-on types | **built-in** | first-class | first-class |
| Learning curve | lowest | low–medium | **high** | low |
| Ecosystem | biggest (107M dl/wk) | curated, official plugins | enterprise (69K stars) | newer (31K stars) |

**Why Fastify for our chat API:** (1) it's the only one with an official SSE plugin that does
what we need — stream tokens, auto-reconnect with `Last-Event-ID` replay, heartbeats
[@fastify/sse]; (2) our hot path is thousands of tiny JSON events per answer, so its 2–3×
serialization speed matters; (3) TypeScript types are built in, not patched on; (4) plugins
map 1:1 onto our `src/` modules (cors, ws, rate-limit, logging). NestJS adds too much
learning overhead for one team; Hono's edge portability buys nothing since we run one
containerized Node server.

```ts
// hello world — 10 lines
import Fastify from "fastify";

const app = Fastify({ logger: true });   // logger: true → pino logging for free

app.get("/hello", async () => ({ hello: "world" }));  // return value = JSON response

app.get("/events", { sse: true }, async (_req, reply) => {   // SSE: add @fastify/sse
  await reply.sse.send({ event: "token", data: { delta: "hi" } });
});

app.listen({ port: 3000 });
```

Sources: [betterstack](https://betterstack.com/community/guides/scaling-nodejs/fastify-vs-express-vs-hono/), [@fastify/sse](https://github.com/fastify/sse), [npmtrends](https://npmtrends.com/express-vs-fastify-vs-hono). Full evaluation: `docs/design/backend-agent-retrieval.md` §1.
- **[Phase 3.5 — UI]** Design doc `docs/design/ui.md`: **React + Vite (SPA) over Next.js** —
  chat tool has no SEO/SSR need, backend is a separate Fastify API, and App Router's server/client
  complexity isn't worth it for a pure client-streaming UI. zustand for state, Tailwind v4 for
  styling. See §8 for the beginner explanation.

---

## 10. React + Vite vs Next.js (the UI decision)

You know React. The question was never "which UI library" — it's **how the app gets built and
served**. Two ways to run React in production:

| | React + Vite (SPA) | Next.js |
|---|---|---|
| What it is | Vite is just a **dev server + bundler**. You write normal React; the browser renders everything; your API lives on another server | A **full-stack React framework**: routing, server-side rendering (SSR), Server Components, API routes, caching — one tool tries to do frontend *and* backend |
| SSR / SEO | None — browser renders after JS loads | Built-in server rendering (good for SEO / public pages) |
| Mental model | "React renders, my API is somewhere else" — everything you already know still applies | New concepts: `'use client'` boundaries, Server Components, two caches, Server Actions |
| Best for | Interactive apps (chat, dashboards) with a separate API | Public-facing sites needing SEO, or when you want one deployable app |

**Which won and why: React + Vite (SPA).**

1. **This app is a logged-in support console** — no SEO, no public pages, so Next.js's main
   superpower (SSR) is irrelevant. Choosing Vite when "you already have a separate backend and just
   need a great UI setup" is the documented sweet spot (rollbar.com/blog/nextjs-vs-vitejs).
2. **Streaming works the same in both** — our token stream is SSE into the *browser*. In Next.js
   you'd still consume it in a client component ("use client"), so App Router adds ceremony, not value.
3. **You're already learning Node/TS/Fastify/BullMQ.** Next.js App Router is famously tricky even
   for experienced React devs (server/client boundary, two caches). Vite adds exactly one new
   concept: it's a bundler. That's the whole learning surface.

**Minimal hello world — React + Vite** (a Vite SPA is just React + a config file):

```tsx
// main.tsx
import { createRoot } from "react-dom/client";
createRoot(document.getElementById("root")!).render(<h1>Hello, agentic support</h1>);
```

```bash
npm create vite@latest ui -- --template react-ts   # scaffold
cd ui && npm install && npm run dev                 # http://localhost:5173
```

**Minimal hello world — Next.js** (same component, but Next adds file-based routing + SSR by default):

```tsx
// app/page.tsx  (App Router: file = route)
export default function Page() {
  return <h1>Hello, agentic support</h1>;
}
```

```bash
npx create-next-app@latest                          # asks about TS, Tailwind, App Router…
npm run dev                                         # http://localhost:3000
```

The takeaway: both run React; Vite leaves the backend to Fastify (our plan) while Next.js would
pull a second server-rendering layer into the project for benefits this app doesn't need. Full
evaluation + comparison table: `docs/design/ui.md` §1.

---

## 11. Custom tools with `defineTool` (and what they return)

A custom tool is just an object with a name, a description the model reads, a
TypeBox parameter schema, and an `execute` function. Pass them to a session with
`customTools: [...]`.

```ts
import { defineTool } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

const kbSearch = defineTool({
  name: "kb_search",
  description: "Search the product manuals KB",
  parameters: Type.Object({ query: Type.String({ description: "The query" }) }),
  execute: async (_toolCallId, params, signal, _onUpdate, _ctx) => {
    // params is type-safe from the TypeBox schema
    signal?.throwIfAborted();                       // abort-aware
    return {
      content: [{ type: "text", text: "results…" }], // ← what the MODEL sees
      details: { sources: [...], count: 3 },         // ← structured data for YOUR UI
    };
  },
});
```

**Key facts (we verified against the installed SDK 0.84.1):**
- `execute(toolCallId, params, signal, onUpdate, ctx)` — `signal` is the agent's
  AbortSignal (propagate it to `fetch`/DB calls); `ctx` is the extension context.
- Return shape is `AgentToolResult`: `content` (text/image parts, fed back to the
  model) + `details` (arbitrary JSON — we put `sources` there so the SSE bridge
  can build the `done` event's sources list).
- **There is no `isError` field on the return value** — throw from `execute` and
  the SDK marks the tool result as an error (`tool_execution_end.isError`).
- `noTools: "builtin"` disables `read/bash/edit/write` while keeping custom +
  extension tools — this is the support-session sandbox (locked rule: no
  filesystem tools). `tools: [name]` on top of that is an explicit allowlist.

## 12. Guardrails via extension hooks (interception layer)

Extensions register `pi.on(event, handler)` hooks. Four of them form our
guardrail layer — the source of truth we never trust the model/tool output over:

| Hook | Fires | We do |
|---|---|---|
| `input` | user text arrives, before skill/template expansion | block prompt-injection patterns (`{ action: "handled" }` skips the LLM entirely); truncate oversized input (`{ action: "transform" }`) |
| `context` | before **every** LLM call | prepend a safety system note + clamp oversized tool results |
| `tool_call` | before a tool executes | block non-SELECT `tickets_query` args, unknown `route_to_agent` agents, suspicious web queries (`{ block: true, reason }`) |
| `tool_result` | after a tool runs, before the model sees it | scrub PII (emails, phones, SSNs) |

Return-value shapes (from `docs/extensions.md`): input → `{action:
"continue"|"transform"|"handled"}`; context → `{messages}`; tool_call →
`{block?, reason?}`; tool_result → `{content?, details?, isError?}`. Use
`isToolCallEventType<"my_tool", MyInput>("my_tool", event)` to narrow custom
tool events.

**Why the safety note is a user-role message:** the in-session message union is
`UserMessage | AssistantMessage | ToolResultMessage` — there is no `"system"`
role in-session, so we prepend a clearly-marked user message instead.

## 13. Sub-agents: `route_to_agent` spawns a child session per call

pi has **no native handoffs** (ADK's `transfer_to_agent` doesn't exist here), so
sub-agents are a custom tool that spawns a fresh `AgentSession` per call:

1. `route_to_agent` tool runs → child session created with the specialist's
   system prompt (`systemPromptOverride`) and **exactly one** custom tool.
2. `child.prompt(query)` streams; we collect tokens, tool calls, and the final
   answer + `details.sources`.
3. `child.dispose()` in a `finally` — children are cheap (in-memory, no files).

Guards we added: bounded concurrency (3 children max, tiny semaphore), per-child
timeout (60s → abort), parent AbortSignal propagation, and the `tool_call` hook
validates `agent ∈ {rag,sql,web}` **before** a child spawns.

**Mock-first development:** the contract lets each track carry a local copy of
the interface it consumes + a mock. Our `src/retrieval/index.ts` is the local
copy (same signatures as retrieval-core's); `RETRIEVAL_MODE=mock` (default) uses
a keyword-overlap mock KB so `npm run chat` works with zero DB. At integration
the real module replaces it — tools don't change at all.

## 14. Model selection and auth (`ModelRuntime`)

`ModelRuntime.create()` resolves credentials in priority order: runtime
overrides → `~/.pi/agent/auth.json` → env API keys → fallback resolver. Then
`modelRuntime.getAvailable()` returns only models with valid auth. Our resolver
picks: explicit `model` option → `PI_MODEL` env (accepts bare ids like
`deepseek-v4-flash`) → preferred list (`claude-sonnet-4-5`, `claude-haiku-4-5`,
`gpt-5`, …) → first available. Children reuse the same `ModelRuntime` (creating
one per child would re-resolve auth + re-fetch the catalog each time) and prefer
a cheap model (`PI_SPECIALIST_MODEL` or haiku/flash).

---

## Log (append as the project moves)

- **[agent-runtime track]** Phase 6 built: `src/runtime/` (`createSupportRuntime`,
  `noTools:"builtin"`, model resolution), `src/agent/` (supervisor + specialist
  prompts, `route_to_agent` child-session tool with semaphore + timeout),
  `src/tools/` (kb_search / tickets_query / web_search, mock-first),
  `src/guardrails/` (input/context/tool_call/tool_result hooks + PII scrub).
  `npm run chat` runs end-to-end on mock retrieval (no DB). Learned: SDK has no
  in-session `system` role; session-level `turn_start` has no `turnIndex`;
  DDG Lite HTML uses single-quoted attributes; throwing from `execute` marks the
  tool result as error.

---

## 15. API + streaming implementation (api-streaming track)

**Fastify plugins = small modular apps.** Everything in Fastify is a plugin — CORS, SSE,
WebSocket, rate-limit are each one `app.register(plugin, opts)` call. A plugin registered
with `register` gets its own encapsulated scope: hooks/decorators inside it only affect
routes registered under it. That's why `fastify-plugin` wraps most official plugins —
it marks them "apply at root" so e.g. the SSE `onRoute` hook sees every route.

**`@fastify/sse` in three facts** (this bit us — read these before using it):
1. `sse: "only"` on a route makes the handler stream-only; `reply.sse.send({id,event,data})`
   writes one SSE frame. `id` is what the browser echoes back as `Last-Event-ID` on reconnect.
2. The connection only stays open **once the response is committed as SSE** — call
   `reply.sse.sendHeaders()` (+ `keepAlive()`) first thing, or a handler that hasn't written
   anything returns normally and Fastify sends an empty 200 and closes.
3. The plugin's `reply.sse.replay(cb)` only fires **when a `Last-Event-ID` header exists**
   (i.e. reconnects). First-time clients get nothing — so replay the ring buffer manually.
   Heartbeat comments (`: ping`) are built in via `heartbeatInterval` — keeps proxies from
   killing idle streams.

**Ring buffer for replay.** Keep the last N events per chat in an array (cap at 200); on
connect, send everything with `id > Last-Event-ID`. It's how a client that dropped mid-turn
re-catches up without missing tokens. Cost: O(N) memory per chat — trivial at this scale.

**Rate limiting vs connection caps.** `@fastify/rate-limit` counts *requests per minute per
IP* — perfect for POST /api/chat (10/min) and reads (30/min). It's wrong for SSE/WebSocket:
those are one long-lived request, so you cap *concurrent connections per IP* (we use 5) with
your own counter instead, and set `config: { rateLimit: false }` on those routes.

**BullMQ needs `maxRetriesPerRequest: null`.** BullMQ uses Redis blocking commands
(BRPOPLPUSH). ioredis by default retries failed commands and would throw "Reached the max
retries" on blocked calls — setting `maxRetriesPerRequest: null` on the connection is the
documented BullMQ requirement. Queue (producer) and Worker (consumer) each get their own
ioredis connection.

**MCP in 5 lines.** The Model Context Protocol lets an LLM call your tools over JSON-RPC.
With `@modelcontextprotocol/sdk`: create `McpServer`, `server.registerTool(name, {description,
inputSchema: { query: z.string() }}, handler)`, connect a `StdioServerTransport` — done.
`npm run mcp` runs it; logs go to stderr so stdout stays protocol-clean.

**CJS/ESM interop gotcha (`verbatimModuleSyntax`).** Many Fastify plugins are CommonJS
packages whose types use `export default`. Under `verbatimModuleSyntax` + NodeNext, TS types
the default import as the *module namespace* (`typeof import(...)`) — not the plugin function
— so `app.register(sse, …)` fails to typecheck even though it works at runtime. Fix: cast
`const sse = sseModule as unknown as FastifyPluginAsync<…>`. One-liner, but confusing the
first time.

**Mock-driven parallel development.** Three tracks build against a written contract
(`docs/design/integration-contract.md`). This track ships a local mock of the runtime
(`src/runtime/mock.ts`) that emits a believable SDK event sequence — so the whole SSE/WS
pipeline is testable end-to-end before the real agent exists. Integration = swap one import.
That's the pattern: interface + mock + one swap point, verified by `tsc --noEmit` after merge.

---

### 7b. Log — api-streaming track

- **[api-streaming impl]** Built `src/server/` (Fastify: cors, sse, websocket, rate-limit,
  pino), `src/streaming/` (registry + ring buffer, SDK→SSE bridge, SSE + WS handlers),
  `src/queue/` (BullMQ jobs + stub worker), `src/mcp/` (kb_search/tickets_query scaffold),
  `src/runtime/mock.ts` (mock SupportRuntime). `npm run dev` on :8000; /health reports
  Postgres+Redis; POST /api/chat → SSE streams the full mock turn; rate limit 429s after
  burst; WS steer/cancel frames work; MCP initializes over stdio. Details:
  `CONTRACT-NOTES.md` (integration notes) + `DEPS.md` (new deps).

---

## 16. Integration lessons (parallel worktrees + merge)

- **Worktrees make true parallelism safe**: `git worktree add -b <branch> .worktrees/<name>`
  gives each agent its own checkout + branch; main stays clean. Rule that saved us: an
  **integration contract** (module exports + SSE schema + env vars) written before the split,
  with each track carrying a *local copy + mock* of the interface it consumes.
- **Merge conflicts were exactly the seams we predicted**: `src/retrieval/index.ts`
  (real vs local copy), `DEPS.md`, `CONTRACT-NOTES.md`, `lessons.md` (both agents appended
  "section 10"). All resolved by keeping the real impl + concatenating notes.
- **Two real integration bugs** (both contract violations the notes *did* flag):
  1. Sources never reached `done` — the agent loop fires a **second `turn_start`** after a
     tool call (turn 1: tool call, turn 2: answer), and our collector reset on every
     `turn_start`. Fix: reset only on `agent_start`.
  2. Mock SQL schema leaked into the tool description — the agent wrote `SELECT id, product,
     status` against a real table with `ticket_id, product_purchased, ticket_type`. Fix:
     real schema in the tool prompt + sources in tool `details`.
- **Dynamic `import()` of a relative path resolves against the importing module**, not cwd —
  resolve against `process.cwd()` with `pathToFileURL` for env-configured module paths.
- **Silent guardrail blocks hang SSE streams** — when the `input` hook returns `handled`,
  the agent never runs and no terminal event fires. Detect no-op (message count unchanged)
  and emit `error: guardrail_blocked`.

## 17. Building the web UI (Vite + React + Tailwind v4 + zustand)

### 17.1 Vite dev proxy — why the UI never needs CORS

The Fastify API runs on `:8000`, Vite on `:5173`. Instead of enabling CORS and
hard-coding an absolute API URL in every `fetch`, the dev server **forwards
same-origin paths** to the backend:

```ts
// ui/vite.config.ts
server: {
  port: 5173,
  proxy: { "/api": { target: "http://localhost:8000", changeOrigin: true },
           "/health": { target: "http://localhost:8000", changeOrigin: true } },
}
```

The browser calls `/api/...` on the Vite origin; Vite relays it to Fastify.
Production can either serve the built `dist/` from Fastify (`@fastify/static`)
or set `VITE_API_URL` for a remote API. Mind the **exact route prefix**: the
backend's health route is `/health`, not `/api/health` — `/api/health` 404s
(we hit that during verification).

### 17.2 Tailwind v4 — CSS-first theming (no tailwind.config.js)

v4 moved configuration into CSS. Design tokens become CSS custom properties in
a `@theme` block, and every utility is generated from them:

```css
@theme {
  --color-canvas: #0f1115;   /* → bg-canvas, text-canvas, border-canvas… */
  --color-accent: #6366f1;   /* → bg-accent, text-accent … */
}
```

Dark-first + light mode = **swap the variables** on a class, no `dark:` variants:

```css
.light { --color-canvas: #f6f7f9; --color-surface: #ffffff; … }
```

Most online tutorials show v3's `tailwind.config.js` — for v4 read the
[CSS-first docs](https://tailwindcss.com/docs/styling-with-utility-classes).

### 17.3 EventSource (SSE) vs WebSocket — and why tokens use SSE

- **SSE (EventSource)** is one-way server→client over plain HTTP with **typed
  `event:` names**, **auto-reconnect**, and **`Last-Event-ID` replay** — if the
  connection drops, the browser reconnects and the server re-sends missed
  events. Perfect for token/tool streaming (the chat's hot path).
- **WebSocket** is full-duplex (client→server steering, presence) but you hand-
  roll reconnect, backoff, and message framing.
- This backend exposes cancel/steer over plain REST (`POST .../cancel`,
  `.../steer`) — so the UI uses **zero WebSockets** (the design doc's `/ws`
  never landed). One wrapper (`ui/src/lib/sse.ts`) registers a listener per
  event type and dispatches parsed JSON into the store. Testing trick: the
  `EventSource` constructor is injectable, so tests pass a fake class.

### 17.4 zustand — stores without providers, updates without re-render storms

`create<State>()((set, get) => ({ … }))` gives a store readable anywhere via a
hook. Two habits that mattered:

- **Selectors** — `useChatStore((s) => s.messages)` subscribes a component to
  only that slice, so 60 Hz token updates re-render just the bubble, not the
  whole tree.
- **Pure reducer core** — the SSE event handling is a plain function
  `reduceSseEvent(state, event, data) => nextState`, unit-tested directly with
  no store/network; the zustand actions are one-line wrappers around it.
- `persist` middleware saves settings to localStorage automatically.

### 17.5 rAF batching — don't setState per token

Tokens arrive many times per second; one React update each would thrash the
renderer. The hook buffers deltas and flushes once per animation frame:

```ts
tokenBuf.current += delta;
if (rafId.current === null) rafId.current = requestAnimationFrame(flush);
// flush(): one appendTokens() call per frame
```

Same idea as coalescing scroll/resize events.

### 17.6 react-markdown + remark-gfm, and testing with vitest + happy-dom

Assistant answers are markdown: `react-markdown` renders it (GFM tables via
`remark-gfm`), with a `components` override to style code blocks and turn
inline `[1]` citations into clickable superscripts.

Tests run under **happy-dom** (a fast DOM implementation) with
`@testing-library/react`. Two gotchas worth remembering:
- RTL's auto-cleanup between tests needs a **global `afterEach`** — enable
  `test.globals: true` in `vitest.config.ts`, or DOM from earlier tests leaks
  into later ones.
- `vi.fn(class)` is **not constructible** with `new` — pass a real subclass
  that records its instances instead.

---

## 18. Scaling a Postgres search table to 3.76M rows (CFPB full dump)

The CFPB full-dump ingest (3.76M rows, 8.7 GB CSV, streaming csv-parse + batched
`INSERT ... SELECT unnest(...) ON CONFLICT`) was a real scale test. It broke things
in exactly the order you'd expect:

1. **Stale planner stats.** After a 3.76M-row insert, the planner still thought the
   table had 8k rows and chose sequential scans. `ANALYZE tickets;` fixed estimates.
2. **ILIKE `%...%` can't use btree.** Four OR'd `ILIKE` columns forced full scans
   (18.7 s for `q=lg oled`). `pg_trgm` GIN indexes
   (`gin_trgm_ops`) enable substring search via BitmapOr.
3. **Lossy bitmaps at scale.** With default `work_mem` (4 MB) the bitmap for a
   125k-match query exceeded memory → lossy pages → heap rechecks. `work_mem=128MB`
   made the bitmap exact (5.2 s → 1.9 s).
4. **Exact COUNT(*) is the wrong tool.** `count(*)` over 3.76M rows = full BitmapOr
   heap scan. Standard fix: cap the count (`SELECT count(*) FROM (SELECT 1 ... LIMIT 10001)`),
   return `totalCapped: true`, UI shows "10,000+".
5. **Inline `to_tsvector(...)` in WHERE = seq scan.** The hybrid retriever computed
   `to_tsvector('english', narrative || subject || product)` per row → 46 s. Fix: a
   **generated column** (`search_tsv tsvector GENERATED ALWAYS AS (to_tsvector(...)) STORED`)
   + GIN index → index scans. 46 s → 1.9 s worst case, ~45 ms typical.
6. **Big generated-column backfills are slow** (3.76M rows × long narratives ≈ 5+ min
   in one transaction) — run them with patience or `CREATE INDEX CONCURRENTLY`.

Also learned: `websearch_to_tsquery` gives you Google-ish query syntax (`"exact phrase"`,
`-exclude`) for free, and switches search semantics from substring to word-based — a UX
decision, not just a perf one.

**Net:** endpoint worst case 18.7 s → 0.03 s; hybrid retrieval 46 s → 1.9 s; all 65 tests
green. The Postgres-only stack (no Elasticsearch) holds up fine at this scale.

---

## 19. MCP servers (what we learned wiring ours)

- **MCP = a protocol, not a framework.** Any process can be an MCP *server*: it speaks JSON-RPC
  over stdio (or SSE), announces its tools via `tools/list`, and executes them via `tools/call`.
  We exported our retrieval tools (`kb_search`, `tickets_query`) as an MCP server so **any** agent
  framework (pi, Claude, other tools) can use them — the retrieval layer becomes portable.
- **`@modelcontextprotocol/sdk`** does the protocol for you: `new McpServer()` +
  `server.registerTool(name, { title, description, inputSchema }, handler)` + connect a
  `StdioServerTransport`. Logs must go to **stderr** (stdout is the protocol channel).
- **Schemas = LLM ergonomics.** The tool description and zod input schema are what a model sees —
  writing a precise description ("Returns ranked chunks with manual, section and page references")
  is as important as the implementation.
- **Never trust a raw WHERE clause.** Our `tickets_query` takes SQL from the model; the full
  statement is run through the same SELECT-only allowlist as the agent tool, then executed in a
  read-only transaction with a statement timeout. Defense in depth: allowlist + role + timeout.
- **Test handlers without a transport.** Keep the handler logic in plain exported functions
  (mocked retrieval/DB) and let `server.ts` be a thin wiring layer — unit tests never touch stdio.

---

## 20. Containerizing a Node/TS app (Dockerfile lessons)

- **Multi-stage**: `node:24-alpine` build stage (`npm ci` → `tsc`), then a slim runtime stage
  with `npm ci --omit=dev` + compiled `dist/`. Result: small image, no dev deps, no source.
- **`npm ci` needs `package-lock.json`** — copy both package files BEFORE `src/` so Docker layer
  caching keeps `npm ci` from re-running on every code change.
- **tsc does NOT copy non-TS assets.** `schema.sql` is read at runtime by `dist/db/migrate.js`,
  so the Dockerfile must `COPY src/db/schema.sql dist/db/schema.sql` after the build — easy to
  miss (the classic "works locally, breaks in Docker" because tsx runs from source).
- **Compose networking**: inside the network, `localhost` is the container itself — DB/Redis are
  reached by service name (`postgres:5432`, `redis:6379`). `DATABASE_URL` must be rewritten for
  the container context (via compose env, not `.env` — dotenv won't override set vars).
- **Healthchecks make ordering real**: `depends_on: postgres: { condition: service_healthy }`
  with `pg_isready` beats guessing with `sleep`.
- **Boot = migrate + serve**: `CMD ["sh","-c","node dist/db/migrate.js && node dist/server/index.js"]`
  since the schema is idempotent (IF NOT EXISTS) — the container self-provisions its tables.

---

## 21. A retrieval eval harness (and what it caught)

- **Golden-set eval without an LLM judge.** Hand-curate (query → expected sources) pairs and
  score retrieval deterministically: recall@k, precision@k, MRR, hit rate. Fast, reproducible,
  CI-friendly. (Faithfulness of *answers* needs an LLM judge — separate, heavier step.)
- **It immediately found a real bug.** `sql-lg-refund` scored 0: `search_tsv` didn't include
  `ticket_type`, so "refund tickets" couldn't match by type. Fix: add ticket_type to the
  generated column + backfill → eval went 67% → 100%.
- **`websearch_to_tsquery` ANDs all terms.** A 5-word query with one unmatched word
  ("television" — tickets say "LG Smart TV") returns nothing. That's correct FTS behavior, but
  golden queries must be realistic, and it's a reason to consider query relaxation later.
- **Metrics math is easy to get wrong.** First version scored against a set of *hit ids* instead
  of *expected items* — produced recall 4.0. Score per expected item; test the metrics in
  isolation.
- **Baseline now:** 6/6 cases, recall@k 1.00, MRR 1.00, precision@k 0.93 (kb + sql over the
  3.76M-row DB). Run anytime: `npm run eval`.

---

## 22. Pure-router supervisor (fixing double retrieval)

The supervisor originally exposed BOTH direct tools (`kb_search`, `tickets_query`,
`web_search`) AND `route_to_agent`. The model often did both for one question —
the captured event trace showed `kb_search` *and* `route_to_agent(rag)` for the
same "LG TV wifi" query: doubled latency + cost.

Fix: make the supervisor a **pure router** — `customTools: [routeToAgent]` +
`tools: ["route_to_agent"]` allowlist, and the prompt rewritten to "you never
retrieve yourself, always dispatch". Verified live: a two-part question now emits
exactly two `route_to_agent` calls (rag + sql), zero direct tool calls.

Lessons:
- **Exposing both a tool and a router-to-same-tool invites redundancy** — models
  hedge by calling both. Remove the ambiguous path structurally (allowlist), don't
  rely on prompt wording alone ("prefer route_to_agent" was ignored).
- The `tools: [...]` allowlist is the hard guarantee; the prompt is the soft one.
- Child sessions still expose exactly one tool each (rag→kb_search, sql→tickets_query,
  web→web_search) — the router layer is now the only place with routing logic.

---

## 23. Chat session persistence (Postgres-backed)

The registry was in-memory: sessions vanished on restart and the UI sidebar
depended on it. Now every finished turn is written to `chats` + `chat_messages`
(best-effort — a DB failure logs and is swallowed so the chat API keeps working),
and on boot the registry rehydrates the last 100 chats.

Key design points:
- **Live vs historical turns**: live turns hold a pi `AgentSession`; rehydrated
  turns hold `session: null` + stored messages. History/preview serve from
  `turn.messages`, never from the session.
- **Resume = seed the agent**: a follow-up with `conversationId` creates a NEW
  live turn and seeds `agent.state.messages` with the prior history (the SDK's
  sanctioned replacement point) — the model gets full context without replaying.
- **Replace, don't append**: re-saving a chat deletes + reinserts its messages
  (idempotent per chatId), so retries converge.
- **Side effect discovered**: DB-guarded tests persist their mock-runtime chats
  into the real store — harmless but noisy; a test-scoped schema/truncate is a
  future nicety.

---

## 24. Query relaxation (FTS auto-drop)

`websearch_to_tsquery` ANDs every term — ONE unmatched word zeroes the result
set ("refund request lg oled **television**" → 0, because tickets say
"LG Smart TV"). Query relaxation fixes it: generate variants from strictest
(ALL terms) down to a single term, run each until rows come back, and report
`relaxed: true` when a drop happened.

- `queryTerms` keeps quoted phrases (`"credit card"`) as single units;
  `tsQueryVariants` drops trailing terms strictest-first.
- `relaxedSearch` is the generic loop (returns first non-empty variant + attempts).
- Threaded through BOTH the hybrid retrievers (kb + sql) and the `/api/tickets`
  endpoint; `relaxed` surfaces in responses so the UI can hint "broader results".
- The eval's golden case now *requires* relaxation ("refund request lg oled
  television") — and scores 100%. Endpoint: 0 → 49 results with relaxed:true.
- Gotchas hit: param-slot collision (LIMIT reused the q's $1 — the count-cap
  refactor broke it), and a rewrite that accidentally deleted the /:id route
  (tests caught it immediately).

---

## 25. Answer-level faithfulness eval (LLM judge)

Retrieval metrics (recall@k, MRR) can't see whether the ANSWER is good — so we
added a judge: run a real agent turn per golden case, capture the final answer
+ sources, and have a cheap model (haiku-class) score faithfulness 1-5
(`npm run eval:answer`).

First run's honest findings (avg 3.3/5, 50% pass):
- **Faithfulness ≠ correctness**: the judge rewards an honest "not found". The
  Sony case — retrieval finds the Xperia manual (retrieval eval: 100%) but the
  answer said "not in catalog" — passed as faithful but was factually WRONG.
  Retrieval eval and answer eval measure different things; you need both.
- **Clarifying-question answers pass** (no fabrication) but aren't answers — the
  router's "ask one short clarifying question" rule fires too eagerly on
  answerable queries.
- Judge scores can look harsh on real-content answers (kenmore got 1 with a
  plausible answer) — verdicts are directional, not gospel; inspect the report.
- `parseVerdict` is the testable seam: strict-JSON prompt + tolerant parser
  (code fences, clamping). Judge calls use a dedicated no-tools session —
  reuse the model-picking pattern instead of pi-ai/compat imports.

---

## 26. Configurable retrieval strategies (HYDE, RRF-k, multiQuery, expansion, rerank)

Exploration phase: the retrieval layer is now a **strategy object** that flows
UI → POST /api/chat → runtime → the kb_search tool (built per-session via
`buildKbTool(strategy)`, so no shared mutable state) → searchHybrid.

Techniques:
- **Mode**: hybrid (FTS+vector+RRF) / vector / keyword / **hyde** (embed an
  LLM-generated hypothetical answer instead of the query) / **hyde-hybrid**.
- **rrfK** slider (10-120): the RRF fusion constant, k=60 default.
- **relax** (query relaxation, §24), **multiQuery** (LLM paraphrases → retrieve
  each → RRF-fuse), **queryExpansion** (rule-based synonyms, free), **rerank**
  (Cohere cross-encoder, key-gated passthrough).

Measured on the golden set (`npm run eval -- --strategy <mode>`):
| mode | recall@k | MRR | note |
|---|---|---|---|
| hybrid | 1.00 | 1.00 | default |
| vector | 1.00 | 1.00 | matches hybrid on this tiny corpus |
| keyword | 0.83 | 0.83 | misses a case |
| hyde | 1.00 | 0.92 | HYDE doesn't beat plain vector here (3 manuals — HYDE shines on big/noisy corpora) |
| hyde-hybrid | 1.00 | 0.92 | same as hyde |

Lessons:
- **Technique value is corpus-dependent**: HYDE adds LLM latency+cost and didn't
  beat plain vector on 3 manuals. The eval is how you know — always measure.
- **Strategy must be per-session, not global**: thread it via a tool factory
  (closure), never a module singleton (concurrent chats would race).
- The conditional-SQL bug: the RRF score expression referenced `fts.rank` even
  in vector-only mode — build the FROM/score expressions per mode.
- `hybrid` vs `vector` identical here because hash-embeddings ≈ token overlap;
  with real embeddings + a bigger corpus the difference will show.
