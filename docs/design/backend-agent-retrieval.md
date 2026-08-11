# Design: Backend, Agent Loop, Retrieval (Phase 3.1–3.3)

**Status:** design proposal (owner review before Phase 5+ implementation)
**Applies to:** v2 rebuild on `@earendil-works/pi-coding-agent` (Node/TypeScript)
**Sections:** 1 backend framework → 2 API design → 3 agent loop (pi SDK) → 4 retrieval → 5 project structure

---

## 1. Backend framework evaluation (Fastify vs Express vs NestJS vs Hono)

### 1.1 The decision context

We need a Node.js HTTP framework for a **real-time chat API**: an agent that streams
tokens/tool events over SSE, a WebSocket channel, REST control endpoints (steer/cancel),
a BullMQ task queue, and health checks. The owner knows React but is **new to Node**, so
this section is written for a beginner.

**What a web framework does (30 seconds):** your server needs to answer HTTP requests
("give me `/api/chat`"). A framework gives you the plumbing: routing (URL → function),
request parsing, response serialization, and middleware (functions that run for every
request, e.g. auth). It's the Node equivalent of Flask/FastAPI in Python or a router in React.

**The four candidates in one sentence each:**

| Framework | One sentence | React analogy |
|---|---|---|
| **Express** | The 2010-era default: tiny core, you bolt everything on yourself | Vanilla React with no helpers |
| **Fastify** | Express's faster, more modern cousin with validation/serialization built in | React + built-in hooks/rules |
| **NestJS** | Angular-style "enterprise" framework: dependency injection, modules, decorators | React + Redux + strict architecture rules |
| **Hono** | Ultra-light, TypeScript-first, runs anywhere incl. edge (Cloudflare Workers) | Preact/Svelte — small, modern, edge-friendly |

### 1.2 Comparison table

| Dimension | **Express** | **Fastify** ⭐ | **NestJS** | **Hono** |
|---|---|---|---|---|
| **Performance (req/s)** | Baseline (1×) | **2–3× Express** for JSON APIs — radix-tree router + schema-based serialization [1] | Express-speed by default; ≈ Fastify speed only via the Fastify adapter | Fast, low overhead; optimizes cold start/bundle size [1] |
| **Streaming / SSE support** | Manual (`res.write`); you assemble SSE yourself | **First-class `@fastify/sse` plugin**: async-iterator streams, `Last-Event-ID` replay, heartbeats, `sse: 'only'/'dual'/'manual'` route modes, TS types [2] | Manual per-provider (SSE via `@nestjs/sse`, WS via gateway decorators) | `streamSSE()` / `stream()` helpers built in, but edge-focused [3] |
| **WebSocket support** | `ws` glued manually | `@fastify/websocket` official plugin | `@nestjs/websockets` gateway decorators | `@hono/node-ws` helper |
| **TypeScript ergonomics** | Community types (`@types/express`); quality varies across middleware [1] | **Built-in TS**: request/response types inferred from JSON-schema; plugin types included [1] | First-class TS, decorators + DI | **TypeScript-first by design**; excellent inference [1] |
| **Learning curve** | Lowest (minimal API) | Low–medium: familiar routing, one new idea (plugins + schemas) | **High**: DI, modules, decorators — Angular-shaped [1] | Low; modern TS idioms |
| **Ecosystem** | **Largest**: 15+ years, 107M weekly downloads, 69.2K stars [3] | Growing, **quality-curated** official plugins; 8.7M weekly downloads, 36.6K stars; used by Microsoft, Walmart, Discord [3][4] | Large, enterprise-heavy; 69.2K stars (most of the four) [4] | Newer but fast-growing; 31.2K stars; 46.8M weekly downloads counted across runtimes [3] |
| **Best fit** | Simple APIs, legacy, huge middleware pool [1] | **High-throughput services, real-time APIs, containers** [1] | Big teams, strict conventions, enterprise [1] | Serverless/edge (Workers, Deno, Bun), minimal APIs [1] |

### 1.3 Recommendation: **Fastify**

Concrete reasons, tied to *this* project:

1. **It is the only candidate with a first-class, officially maintained SSE plugin** that
   does what our chat API needs out of the box: streaming async iterators, client reconnect
   via `Last-Event-ID` replay, heartbeats, and typed route modes (`sse: 'only'` for
   `/api/chat/:id/events`) [2]. Express would force us to hand-roll SSE framing and replay;
   NestJS adds an architecture we don't need for a learning project.
2. **2–3× Express throughput** for JSON APIs [1]. We stream many small JSON events per turn
   (token deltas), so serialization speed is not cosmetic — it's the hot path.
3. **TypeScript built in, not bolted on.** Fastify types flow from route schemas into
   handlers; Express relies on separate `@types/express` and inconsistent middleware typing [1].
4. **Plugin model = clean modularity that matches our structure.** Logging (pino), CORS,
   rate-limit, SSE, WebSocket are all official plugins — one `register()` each. This maps
   1:1 onto our `src/server/`, `src/streaming/` modules.
5. **Right-sized learning curve.** Hono is tempting (TS-first, tiny), but its home turf is
   edge/serverless [1] — we run one containerized Node service; Hono's edge portability buys
   nothing here and its ecosystem is younger. NestJS is the opposite error: its DI/decorator
   machinery is real learning overhead that doesn't serve a single-agent-team API.
6. **No lock-in risk.** Fastify is a proven, mature, MIT-licensed core maintained by a real
   org (4.8M downloads, Fortune-500 users) [4]. The API is small enough that a future swap
   is contained in `src/server/`.

**Deployment note:** SSE through Nginx needs buffering disabled (`X-Accel-Buffering: no` or
`proxy_buffering off`) — Fastify lets us set response headers per route easily; we'll do this
in the Docker/nginx config during Phase 10.

Sources: [1] Better Stack framework guide (https://betterstack.com/community/guides/scaling-nodejs/fastify-vs-express-vs-hono/) · [2] `@fastify/sse` README (https://github.com/fastify/sse) · [3] npm trends (https://npmtrends.com/express-vs-fastify-vs-hono) · [4] HireNodeJS 2026 comparison (https://www.hirenodejs.com/blog/nodejs-frameworks-compared-2026)

---

## 2. API design

### 2.1 Endpoints

| Method | Path | Purpose | Response |
|---|---|---|---|
| `POST` | `/api/chat` | Create a chat turn (accept user message, start agent run in background) | `201` + `{ chatId, conversationId, eventsUrl }` |
| `GET` | `/api/chat/:id/events` | **SSE stream** for a turn (replays buffered events, then live) | `text/event-stream` |
| `POST` | `/api/chat/:id/steer` | Queue a mid-stream steering message (`session.steer`) | `202` |
| `POST` | `/api/chat/:id/cancel` | Abort the running turn (`session.abort()`) | `202` |
| `WS` | `/api/chat/:id` | Full-duplex channel: receive the same events, send `steer`/`cancel` control frames | WebSocket |
| `POST` | `/api/tasks` | Enqueue a background BullMQ job (ingest doc, re-embed, eval) | `202` + `{ taskId }` |
| `GET` | `/health` | Liveness/readiness (DB + Redis ping) | `200` `{ status: "ok", deps: {...} }` |

**Why a separate `POST /api/chat` + `GET .../events` pair (not `POST` streaming the response)?**
The agent turn outlives a single HTTP request (tool calls, sub-agents). By separating
*start* from *stream*, clients can reconnect to the SSE stream after a drop and replay
missed events via `Last-Event-ID` — a property we get free from `@fastify/sse` [2].

### 2.2 Request / response shapes

**POST /api/chat** — body:

```jsonc
{
  "message": "My LG Smart TV won't connect to Wi-Fi, ticket #10293",
  "conversationId": "conv_01JQX2A8K9",   // optional — omit to create a new conversation
  "ticketId": 10293,                     // optional — attach ticket context
  "metadata": { "channel": "web_chat" }  // optional, free-form
}
```

**201 Created** — body:

```jsonc
{
  "chatId": "chat_01JQX2B4M1",           // this turn's id (the :id in event URLs)
  "conversationId": "conv_01JQX2A8K9",
  "eventsUrl": "/api/chat/chat_01JQX2B4M1/events",
  "status": "started"
}
```

**POST /api/chat/:id/steer** — body: `{ "text": "Actually, also check the router settings" }` → `202 { "queued": true }`
**POST /api/chat/:id/cancel** — body: `{}` → `202 { "cancelled": true }`
**POST /api/tasks** — body: `{ "type": "ingest.document", "payload": { "path": "manuals/lg-oled.pdf" } }` → `202 { "taskId": "bull_01J..." }`
**GET /health** → `200 { "status": "ok", "uptime": 1234.5, "deps": { "postgres": "ok", "redis": "ok" } }`

### 2.3 SSE event schema

Transport framing (standard SSE): each event is `id: <seq>\nevent: <type>\ndata: <json>\n\n`.
The browser `EventSource` auto-reconnects; we send `id` so `@fastify/sse` can replay from
`Last-Event-ID` [2]. `data` is always a JSON object.

Event types and payloads (ordered roughly as they occur in a turn):

**`turn_start`** — a new LLM turn (response + tool calls) begins.

```jsonc
// event: turn_start
{ "chatId": "chat_01JQX2B4M1", "turnIndex": 1, "ts": 1769800000000 }
```

**`token`** — a text token delta (from SDK `message_update` / `text_delta`).

```jsonc
// event: token
{ "chatId": "chat_01JQX2B4M1", "turnIndex": 1, "delta": "Let me check" }
```

**`tool_start`** — a tool call is about to execute (SDK `tool_execution_start`).

```jsonc
// event: tool_start
{
  "chatId": "chat_01JQX2B4M1",
  "turnIndex": 1,
  "toolCallId": "call_01",
  "toolName": "route_to_agent",
  "args": { "agent": "rag", "query": "LG Smart TV Wi-Fi reset instructions" }
}
```

**`tool_end`** — a tool call finished (SDK `tool_execution_end`).

```jsonc
// event: tool_end
{
  "chatId": "chat_01JQX2B4M1",
  "turnIndex": 1,
  "toolCallId": "call_01",
  "toolName": "route_to_agent",
  "isError": false,
  "durationMs": 412,
  "summary": "3 chunks retrieved from knowledge base"
}
```

**`turn_end`** — an LLM turn completed (SDK `turn_end`).

```jsonc
// event: turn_end
{ "chatId": "chat_01JQX2B4M1", "turnIndex": 1, "ts": 1769800001200 }
```

**`done`** — the whole turn finished; carries the final answer and **sources** (SDK
`agent_settled` + message state).

```jsonc
// event: done
{
  "chatId": "chat_01JQX2B4M1",
  "conversationId": "conv_01JQX2A8K9",
  "turnIndex": 2,
  "message": "To reset Wi-Fi on your LG Smart TV: Settings → General → Network → …",
  "sources": [
    {
      "type": "kb",                      // "kb" | "sql" | "web"
      "title": "LG OLED TV User Guide — Network Settings",
      "docName": "lg-oled-user-guide.pdf",
      "sectionPath": "4.2 Wi-Fi Connection",
      "page": 42,
      "score": 0.91,
      "url": null
    },
    {
      "type": "sql",
      "title": "ticket #10293",
      "row": { "id": 10293, "ticket_type": "Technical issue", "ticket_priority": "High", "ticket_status": "Open" },
      "score": null
    }
  ],
  "usage": { "inputTokens": 2140, "outputTokens": 318, "totalCostUsd": 0.0042 }
}
```

**`error`** — a failure (SDK error event / `prompt()` rejection / our timeout guard).

```jsonc
// event: error
{
  "chatId": "chat_01JQX2B4M1",
  "code": "turn_timeout",                // "turn_timeout" | "canceled" | "provider_error" | "guardrail_blocked" | "internal"
  "message": "Agent turn exceeded 120s budget and was aborted",
  "retryable": true
}
```

Optional extras we'll emit when present (cheap, aids the UI): `thinking` (thinking deltas),
`retry_start` / `retry_end` (SDK auto-retry), `queue_update` (steer/follow-up queue depth).

### 2.4 Control-flow semantics

- **Start**: `POST /api/chat` returns immediately; the turn runs in the background. Events
  buffer in the streaming registry (ring buffer, last ~200) until an SSE client connects;
  `@fastify/sse` `replay()` serves the gap on reconnect via `Last-Event-ID`.
- **Steer**: `session.steer(text)` queues a steering message delivered after the current
  turn's tool calls (SDK semantics — it interrupts mid-stream cleanly).
- **Cancel**: `session.abort()` + registry marks the turn canceled; SSE clients get an
  `error` event with `code: "canceled"` and the stream closes.
- **WS**: same event feed as SSE plus inbound `{ type: "steer"|"cancel", ... }` frames —
  one transport for clients that want full duplex (mobile app, advanced UI).

### 2.5 Security posture (owner decisions — locked)

- **No authentication.** This is a learner project, not a user platform — no login, no tokens,
  no user accounts. API is open on the dev host.
- **Solid rate limiting is REQUIRED** (owner). Per-IP limits on all mutating/chat endpoints:
  - `@fastify/rate-limit` on `POST /api/chat`, `POST /api/chat/:id/steer|cancel`, `POST /api/tasks`
    — e.g. 10 req/min/IP for chat, 30 req/min/IP for reads (`GET /api/tickets*`, `GET /api/manuals*`).
  - Per-session guard: max tokens per turn / max turns per session enforced in the agent runtime
    (protects against runaway loops and large-context abuse).
  - SSE/WS connections are cheap to open but long-lived: cap concurrent connections per IP
    (e.g. 5) via `@fastify/websocket` options + an SSE connection counter.
- **Never expose filesystem tools** to the support agent (locked project decision).
- Database access stays read-only for the app role + separate `acs_readonly` role for the SQL tool
  (§4.6).

---

## 3. Agent loop design (pi agents SDK)

All SDK imports live in `src/runtime/` (project rule). API verified against installed
`@earendil-works/pi-coding-agent@0.84.1` (`docs/sdk.md`, `docs/extensions.md`,
`examples/sdk/03-custom-prompt.ts`, `05-tools.ts`, `13-session-runtime.ts`).

### 3.1 Wiring: `createAgentSession`

```ts
// src/runtime/session-factory.ts
import { createAgentSession, DefaultResourceLoader, getAgentDir, ModelRuntime,
         SessionManager, SettingsManager } from "@earendil-works/pi-coding-agent";
import { getModel } from "@earendil-works/pi-ai";
import { supportGuardrails } from "../guardrails/extension.js";
import { routeToAgent } from "../agent/route-to-agent.js";
import { SUPPORT_SYSTEM_PROMPT } from "../agent/support-prompt.js";

export async function createSupportSession(opts: { chatId: string }) {
  const modelRuntime = await ModelRuntime.create();          // auth from env + ~/.pi/agent/auth.json
  const settingsManager = SettingsManager.inMemory({         // no file I/O server-side
    compaction: { enabled: true },                           // keep long conversations bounded
    retry: { enabled: true, maxRetries: 2 },
  });

  // Resource loader: custom system prompt + our guardrail extension.
  const loader = new DefaultResourceLoader({
    cwd: process.cwd(),
    agentDir: getAgentDir(),
    settingsManager,
    systemPromptOverride: () => SUPPORT_SYSTEM_PROMPT,       // replaces pi's coding prompt entirely
    appendSystemPromptOverride: () => [],                    // suppress stray AGENTS.md/appends
    extensionFactories: [supportGuardrails],                 // input/context/tool_call/tool_result hooks
  });
  await loader.reload();

  const model = getModel("anthropic", "claude-opus-4-5");    // from @earendil-works/pi-ai
  if (!model) throw new Error("model not found");

  const { session } = await createAgentSession({
    modelRuntime,
    settingsManager,
    resourceLoader: loader,
    sessionManager: SessionManager.inMemory(),               // registry holds state; see §3.5
    model,
    thinkingLevel: "low",
    noTools: "builtin",                                      // ← architecture rule 1: NO fs tools
    customTools: [routeToAgent],                             // the only tool the supervisor sees
  });

  return { session, modelRuntime, loader };
}
```

Key SDK facts this leans on (from `docs/sdk.md`):
- `noTools: "builtin"` disables default `read/bash/edit/write` while keeping extension and
  custom tools enabled — exactly the support-session sandbox we want.
- `DefaultResourceLoader({ systemPromptOverride })` swaps the ~1,000-token coding prompt for
  our customer-support prompt; `appendSystemPromptOverride` stops `AGENTS.md` files from
  leaking into product sessions.
- `ModelRuntime.create()` resolves credentials (env vars → `~/.pi/agent/auth.json`).
- `SettingsManager.inMemory()` avoids touching user files; `compaction` keeps turns bounded.

### 3.2 The `route_to_agent` custom tool (sub-agents)

pi has **no native handoffs** (verified: no supervisor primitive in `docs/sdk.md`), so we
implement routing as a custom tool that spawns a **child `AgentSession`**, runs one
specialist prompt + one tool, and disposes the child. Children are cheap (in-memory, no
files), so spawn-per-call is the right model.

```ts
// src/agent/route-to-agent.ts
import { defineTool, createAgentSession, DefaultResourceLoader, getAgentDir,
         ModelRuntime, SessionManager, SettingsManager } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { ragTool } from "../tools/rag-tool.js";
import { sqlTool } from "../tools/sql-tool.js";
import { webTool } from "../tools/web-tool.js";
import { SPECIALIST_PROMPTS, type AgentKind } from "./specialists.js";

const AGENTS: Record<AgentKind, { tool: ReturnType<typeof defineTool>; model: string }> = {
  rag: { tool: ragTool,  model: "anthropic/claude-haiku-4-5" },   // cheap model for extraction
  sql: { tool: sqlTool,  model: "anthropic/claude-haiku-4-5" },
  web: { tool: webTool,  model: "anthropic/claude-haiku-4-5" },
};

export const routeToAgent = defineTool({
  name: "route_to_agent",
  label: "Route to Specialist Agent",
  description:
    "Route a query to a specialist sub-agent. agent must be one of: " +
    "'rag' (knowledge base / manuals), 'sql' (tickets database), 'web' (live web search). " +
    "Call it with the user's exact question; it returns the specialist's answer with sources. " +
    "Never call it for small talk or to confirm your own knowledge.",
  parameters: Type.Object({
    agent: Type.String({ description: "rag | sql | web" }),
    query: Type.String({ description: "The query for the specialist" }),
  }),
  execute: async (_toolCallId, params, signal) => {
    const kind = params.agent as AgentKind;
    const child = await spawnChildSession(kind, signal);
    try {
      const events = await runAndCollect(child, params.query, signal); // tokens + final content
      return {
        content: [{ type: "text", text: events.finalText }],
        details: { sources: events.sources, turnCount: events.turnCount },
      };
    } finally {
      child.dispose();                                               // ← dispose after use
    }
  },
});

async function spawnChildSession(kind: AgentKind, signal?: AbortSignal) {
  const { tool } = AGENTS[kind];
  const modelRuntime = await ModelRuntime.create();
  const settingsManager = SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: true, maxRetries: 1 } });
  const loader = new DefaultResourceLoader({
    cwd: process.cwd(), agentDir: getAgentDir(), settingsManager,
    systemPromptOverride: () => SPECIALIST_PROMPTS[kind],
    appendSystemPromptOverride: () => [],
  });
  await loader.reload();
  const { session } = await createAgentSession({
    modelRuntime, settingsManager, resourceLoader: loader,
    sessionManager: SessionManager.inMemory(),
    noTools: "builtin",                // child has NO filesystem tools
    customTools: [tool],
    tools: [tool.name],                // allowlist: only the specialist tool is callable
    thinkingLevel: "off",              // specialists answer fast; no hidden reasoning
  });
  return session;
}
```

**Guardrail bonus:** because routing is a tool call, the `tool_call` extension hook can
validate `agent ∈ {rag, sql, web}` before the child spawns — the interception layer, not the
model, is the source of truth (architecture rule 3).

### 3.3 Guardrails via extension hooks

We attach one inline extension to the **supervisor** session (and a slimmed variant to
children) using the four interception hooks from `docs/extensions.md`:

```ts
// src/guardrails/extension.ts
import type { ExtensionAPI, InlineExtension } from "@earendil-works/pi-coding-agent";
import { ALLOWED_AGENTS, MAX_INPUT_CHARS, MAX_TOOL_RESULT_CHARS } from "../config/limits.js";

export const supportGuardrails: InlineExtension = {
  name: "acs-guardrails",
  factory: (pi: ExtensionAPI) => {
    // 1) INPUT — validate/sanitize what the user sends (before template expansion)
    pi.on("input", async (event) => {
      if (event.text.length > MAX_INPUT_CHARS) {
        return { action: "transform", text: event.text.slice(0, MAX_INPUT_CHARS) };
      }
      return { action: "continue" }; // default: pass through
    });

    // 2) CONTEXT — trim oversized tool results / prune PII before each LLM call
    pi.on("context", async (event) => {
      const messages = event.messages.map((m) => clampContent(m, MAX_TOOL_RESULT_CHARS));
      return { messages };
    });

    // 3) TOOL_CALL — block invalid routing targets; never trust the model blindly
    pi.on("tool_call", async (event) => {
      if (event.toolName === "route_to_agent") {
        const input = event.input as { agent?: string };
        if (!input.agent || !ALLOWED_AGENTS.has(input.agent)) {
          return { block: true, reason: `Unknown sub-agent: ${input.agent}`, terminate: false };
        }
      }
    });

    // 4) TOOL_RESULT — post-process results (attach source ids for SSE 'done', cap size)
    pi.on("tool_result", async (event) => {
      return { details: { ...event.details, sources: event.details?.sources ?? [] } };
    });
  },
};
```

Why these four (from `docs/extensions.md` lifecycle): `input` fires before skill/template
expansion and can transform text; `context` fires before **every** LLM call and can rewrite
messages; `tool_call` can **block** a tool (return `{ block: true, reason }`); `tool_result`
can **modify** a result before it reaches the model. The SQL read-only enforcement itself
lives one layer deeper — in the `sql_tool` (see §4.6) — so it holds even if hooks are
misconfigured.

### 3.4 Event → SSE mapping (the streaming bridge)

One module (`src/streaming/bridge.ts`) subscribes to the SDK event stream and pushes typed
SSE events into the registry (architecture rule 4: all event→client streaming through one
bridge).

```ts
// src/streaming/bridge.ts
export function attachBridge(session: AgentSession, sink: ChatEventSink) {
  return session.subscribe((event) => {
    switch (event.type) {
      case "message_update":
        if (event.assistantMessageEvent.type === "text_delta")
          sink.emit("token", { delta: event.assistantMessageEvent.delta });
        else if (event.assistantMessageEvent.type === "thinking_delta")
          sink.emit("thinking", { delta: event.assistantMessageEvent.delta });
        break;
      case "tool_execution_start":
        sink.emit("tool_start", { toolCallId: event.toolCallId, toolName: event.toolName, args: event.args });
        break;
      case "tool_execution_update":
        sink.emit("tool_update", { toolCallId: event.toolCallId, partial: event.partialResult });
        break;
      case "tool_execution_end":
        sink.emit("tool_end", { toolCallId: event.toolCallId, toolName: event.toolName, isError: event.isError });
        break;
      case "turn_start":  sink.emit("turn_start", { turnIndex: event.turnIndex }); break;
      case "turn_end":    sink.emit("turn_end",   { turnIndex: event.turnIndex }); break;
      case "agent_settled": // whole run done — bridge emits 'done' with sources from registry
        sink.emit("done", buildDonePayload(sink.chatId, session));
        break;
      case "queue_update": sink.emit("queue_update", { steering: event.steering.length, followUp: event.followUp.length }); break;
      case "auto_retry_start": sink.emit("retry_start", {}); break;
      case "auto_retry_end":   sink.emit("retry_end", {}); break;
    }
  });
}
```

### 3.5 Abort / timeout / error handling

| Failure mode | Mechanism |
|---|---|
| User cancel | `POST /api/chat/:id/cancel` → `session.abort()` (SDK `AgentSession.abort()`); bridge emits `error { code: "canceled" }`; stream closes |
| Turn timeout | Per-turn `AbortSignal.timeout(TURN_BUDGET_MS)` (e.g. 120s) wired into `prompt()`; on abort, `session.abort()` + `error { code: "turn_timeout" }`. Custom tool `execute` receives the signal (`signal` param) and should pass it to `fetch`/DB calls (`ctx.signal` equivalent for tools) |
| Provider error | SDK surfaces failures through the event/message stream and `prompt()` rejection; bridge catches `prompt()` rejection and emits `error { code: "provider_error", retryable: true }`; `SettingsManager` retry (maxRetries: 2) already covers transient failures |
| Guardrail block | `tool_call` `{ block: true, reason }` → SDK emits the blocked tool result as an error; we surface `error { code: "guardrail_blocked" }` with the reason |
| Client disconnects | SSE handler `onClose` removes the subscriber; the agent turn continues (events buffered) so a reconnect can replay from `Last-Event-ID` |
| Server crash | Turns are in-process; on restart, active turns are lost — acceptable for v2 (documented; durable orchestration is a later phase). Conversations survive in Postgres (§3.6) |

### 3.6 Session registry + persistence

- **Registry** (`src/streaming/registry.ts`): `Map<chatId, ChatTurn>` where `ChatTurn =
  { session: AgentSession, conversationId, subscribers: Set<Subscriber>, ringBuffer: Event[],
  status, startedAt }`. One `AgentSession` per turn (fresh in-memory session per turn keeps
  context cheap; conversation continuity comes from Postgres, below).
- **Conversation persistence** (`chat_sessions` + `chat_messages` tables): every user
  message and final assistant answer (with sources JSONB) is appended by the bridge. On a
  new turn for the same conversation, we hydrate the agent with the recent history
  (last N messages injected via `session.prompt` preface or `context` hook) — this gives
  multi-turn memory without keeping pi JSONL session files server-side.
- **Why not `SessionManager.create(cwd)` (JSONL files)?** The SDK's file sessions are
  designed for the interactive harness (tree branching, `/resume`). For a web product we
  want our own storage anyway (multi-tenant, queryable, PII-controlled), so
  `SessionManager.inMemory()` + Postgres is the documented, lighter path.

---

## 4. Retrieval design

### 4.1 Hybrid search (pgvector + Postgres FTS + RRF)

Dense-only retrieval loses lexical precision (model numbers, ticket IDs — exactly this
domain); hybrid BM25+dense consistently beats either alone (WANDS 0.7497 vs 0.6953/0.6983
[tech-stack-research §4.2]). We fuse **rank-based** with **Reciprocal Rank Fusion (RRF)** —
`1/(k + rank)` — because it is robust to score outliers (better than min-max score fusion).

One SQL query, no new services (pgvector ≥ 0.9, Postgres `tsvector` GIN + HNSW indexes):

```sql
-- retrieval/hybrid.sql — parameterized ($1 query, $2 embedding, $3 embedding_model, $4 top_n)
WITH fts AS (                                     -- lexical: tsvector @@ websearch_to_tsquery
  SELECT id, row_number() OVER (ORDER BY ts_rank_cd(content_tsv, q) DESC) AS rank
  FROM document_chunks, websearch_to_tsquery('english', $1) AS q
  WHERE content_tsv @@ q
  LIMIT 50
),
vec AS (                                          -- semantic: cosine distance (<=>)
  SELECT id, row_number() OVER (ORDER BY embedding <=> $2::vector) AS rank
  FROM document_chunks
  WHERE embedding_model = $3
  LIMIT 50
),
rrf AS (                                          -- fuse by rank position
  SELECT id,
         COALESCE(1.0 / (60 + fts.rank), 0.0) + COALESCE(1.0 / (60 + vec.rank), 0.0) AS score
  FROM fts FULL OUTER JOIN vec USING (id)
)
SELECT c.id, c.doc_id, c.section_path, c.page, c.content, rrf.score
FROM rrf JOIN document_chunks c ON c.id = rrf.id
ORDER BY rrf.score DESC
LIMIT $4;
```

Notes:
- **`<=>` cosine**, not the old `<#>` inner-product — the Python review flagged `<#>` as
  magnitude-sensitive (project-analysis §4.6); HNSW index uses `vector_cosine_ops`.
- `websearch_to_tsquery` gives BM25-flavored ranking with safe user-query parsing
  (no injection: everything parameterized via `$1..$4` — fixing the old f-string
  injection bug S1).
- Upgrade path if FTS quality disappoints: ParadeDB `pg_search` for true BM25 (§4.2 in
  tech-stack-research).

### 4.2 Reranker: **Cohere Rerank (start) → BGE-reranker-v2-m3 (optional self-host)**

Cross-encoder reranking is the single highest-leverage retrieval upgrade (~40% RAG accuracy
improvement reported; BEIR NDCG@10 41.7 → 51.0 → ~60 [tech-stack-research §4.3]).

- **Start with Cohere Rerank v4** — hosted, zero ops, strong multilingual.
- **Swap to BGE-reranker-v2-m3** (self-host via sentence-transformers/TEI) if
  data-residency or per-query cost matters — same call shape, one module swap.

Call sketch (top-50 candidates → top 8–10):

```ts
// src/retrieval/rerank.ts
import { CohereClient } from "cohere-ai";

const cohere = new CohereClient({ token: process.env.COHERE_API_KEY });

export async function rerank(query: string, docs: Chunk[], topN = 8) {
  if (!process.env.COHERE_API_KEY) return docs.slice(0, topN); // degrade gracefully
  const res = await cohere.v2.rerank({
    model: "rerank-v3.5",
    query,
    documents: docs.map((d) => d.content),
    topN,
    returnDocuments: true,
  });
  return res.results.map((r) => ({
    ...docs[r.index],
    rerankScore: r.relevanceScore,
  }));
}
```

### 4.3 Embedding strategy

- **Model:** `text-embedding-3-large` (3072-dim) to start — same zero-migration baseline as
  the old project; **`embedding_model` is stored per chunk** so we can re-embed
  incrementally (re-embed job scans `WHERE embedding_model <> target`).
- **Upgrade path:** bge-m3 (dense + sparse in one model, 1024-dim, multilingual) if lexical
  recall or non-English manuals become priorities [tech-stack-research §4.4].
- **Batching:** async batches of 64–128 embeddings per request with retry + backoff
  (fixes the old silent-failure `except:` bug, S4 — no swallowed errors; a failed batch
  fails the ingest job).

### 4.4 Structural chunking

Technical manuals → **structural/recursive chunking by headings**, 10–20% overlap
[tech-stack-research §4.5]:

```
PDF → parse pages → heading tree (1., 1.2, 1.2.3…) → section units
   → if section > ~1200 tokens: recursive split (paragraph/sentence boundaries, 15% overlap)
   → chunk = { content, section_path: "4.2 Wi-Fi Connection", page, doc_id }
```

Chunk metadata (`doc_id`, `section_path`, `page`) is what powers source citations in the
SSE `done` payload and RAGAS context-precision evals. Late chunking is a documented eval
candidate, not v2 scope.

### 4.5 Schema sketches

```sql
-- 4.5.1 tickets (SQL retrieval target; from suraj520 CSV + CFPB enrichment)
CREATE TABLE tickets (
  id               BIGSERIAL PRIMARY KEY,
  customer_name    TEXT,
  customer_email   TEXT,
  customer_age     INT,
  customer_gender  TEXT,
  product_purchased TEXT,          -- 'LG Smart TV', 'iPhone', ...
  date_of_purchase TIMESTAMPTZ,
  ticket_type      TEXT,           -- 'Refund request' | 'Technical issue' | ...
  ticket_priority  TEXT,           -- 'Critical' | 'High' | 'Medium' | 'Low'
  ticket_channel   TEXT,           -- 'Email' | 'Chat' | 'Social Media' | ...
  ticket_subject   TEXT,
  ticket_body      TEXT,
  ticket_status    TEXT DEFAULT 'Open',
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_tickets_product ON tickets (product_purchased);
CREATE INDEX idx_tickets_type    ON tickets (ticket_type);
CREATE INDEX idx_tickets_status  ON tickets (ticket_status);

-- 4.5.2 document_chunks (vector/lexical retrieval target)
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
  id         BIGSERIAL PRIMARY KEY,
  doc_name   TEXT NOT NULL UNIQUE,     -- upsert key (fixes old manual-id bug B5)
  doc_source TEXT NOT NULL,            -- 'pdf' | 'csv'
  doc_meta   JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE document_chunks (
  id             BIGSERIAL PRIMARY KEY,
  doc_id         BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index    INT NOT NULL,
  section_path   TEXT,                          -- '4.2 Wi-Fi Connection'
  page           INT,
  content        TEXT NOT NULL,
  content_tsv    tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
  embedding      vector(3072),
  embedding_model TEXT NOT NULL DEFAULT 'text-embedding-3-large',
  UNIQUE (doc_id, chunk_index)
);
CREATE INDEX idx_chunks_tsv        ON document_chunks USING GIN (content_tsv);
CREATE INDEX idx_chunks_embedding  ON document_chunks USING hnsw (embedding vector_cosine_ops);
```

### 4.6 Read-only SQL policy for the SQL agent

Three independent layers (defense in depth — never trust the model):

1. **DB role** — a dedicated `acs_readonly` role; the app's SQL-tool pool connects as it:
```sql
CREATE ROLE acs_readonly LOGIN PASSWORD 'change-me';
GRANT CONNECT ON DATABASE acs TO acs_readonly;
GRANT USAGE ON SCHEMA public TO acs_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO acs_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO acs_readonly;
-- connection side: default_transaction_read_only = on, statement_timeout = 15s
```
2. **Statement allowlist** — the `sql_tool` validates: single statement only; must start
   with `SELECT`/`WITH` (case-insensitive, after stripping comments); parameterized (no
   string-concatenated literals); `LIMIT` clamped (e.g. ≤ 200); executed via `pg` with
   parameter binding.
3. **Result caps** — truncate returned rows/text to fit the context hook's budget; row
   count reported to the model.

### 4.7 Ingest pipeline (BullMQ jobs)

```
POST /api/tasks { type: "ingest.document" }
  → BullMQ job "ingest.document" (worker in src/queue/worker.ts)
  → PDF: pdf-parse → structural chunker (§4.4) → embed (batch) → upsert documents+chunks
     (upsert by doc_name → BIGSERIAL ids; idempotent re-runs — fixes B5)
POST /api/tasks { type: "ingest.tickets" }
  → CSV: parse → column mapping (suraj520 → tickets schema; CFPB → enriched cols)
  → COPY/Bulk insert (idempotent by (customer_email, ticket_subject, created_at) hash)
POST /api/tasks { type: "reembed" }
  → chunks WHERE embedding_model <> target → re-embed → update (incremental)
```

---

## 5. Project structure

### 5.1 `src/` layout

```
src/
├── config/            env parsing + limits (TURN_BUDGET_MS, MAX_INPUT_CHARS, top_k defaults)
├── runtime/           ← ONLY module that imports @earendil-works/pi-coding-agent
│   ├── session-factory.ts     createSupportSession() / spawnChildSession()
│   ├── models.ts              model selection (getModel), ModelRuntime creation
│   └── loader.ts              DefaultResourceLoader builders (system prompt, extensions)
├── agent/
│   ├── support-prompt.ts      supervisor system prompt (routing rules, tone, cite-sources)
│   ├── specialists.ts         rag/sql/web specialist prompts + model mapping
│   └── route-to-agent.ts      the route_to_agent custom tool (defineTool)
├── tools/
│   ├── rag-tool.ts            hybrid search + rerank (calls retrieval/)
│   ├── sql-tool.ts            read-only SQL allowlist executor
│   └── web-tool.ts            Tavily search (DDG fallback behind flag)
├── guardrails/
│   └── extension.ts           input / context / tool_call / tool_result hooks
├── streaming/
│   ├── bridge.ts              SDK event → SSE event mapping (attachBridge)
│   ├── registry.ts            Map<chatId, ChatTurn> + ring buffer + subscribers
│   ├── sse.ts                 GET /api/chat/:id/events route (@fastify/sse)
│   └── websocket.ts           WS /api/chat/:id route (@fastify/websocket)
├── server/
│   ├── app.ts                 Fastify instance + plugin registration (cors, sse, ws, rate-limit, pino)
│   └── routes/
│       ├── chat.ts            POST /api/chat, steer, cancel
│       ├── tasks.ts           POST /api/tasks (BullMQ enqueue)
│       └── health.ts          GET /health
├── queue/
│   ├── jobs.ts                job types + payloads
│   └── worker.ts              BullMQ worker (ingest.document, ingest.tickets, reembed)
├── retrieval/
│   ├── hybrid.ts              RRF SQL execution (parameterized)
│   ├── embed.ts               OpenAI embeddings (batched, retried)
│   ├── rerank.ts              Cohere (BGE behind interface)
│   ├── chunk.ts               structural chunker (pdf-parse → sections → chunks)
│   └── ingest.ts              pipeline orchestration (called by queue worker)
├── db/
│   ├── pool.ts                pg Pool (app role + acs_readonly pool)
│   └── schema.sql             tickets / documents / document_chunks / chat tables
├── mcp/
│   └── server.ts              exports retrieval tools as an MCP server (future phase)
└── index.ts                   boot: fastify.listen + worker start
```

### 5.2 Dependencies (versions verified against npm registry today)

| Package | Version | Why |
|---|---|---|
| `@earendil-works/pi-coding-agent` | **0.84.1 (exact pin)** | agent runtime (project rule: pin exact; SDK pre-1.0) |
| `@earendil-works/pi-ai` | pin to match 0.84.1 | `getModel`, model catalogs |
| `fastify` | ^5.11.3 | recommended framework |
| `@fastify/sse` | ^0.6.0 | SSE streaming + Last-Event-ID replay |
| `@fastify/websocket` | ^11.3.0 | WS channel |
| `@fastify/cors` | ^11.3.0 | CORS for the React SPA |
| `@fastify/rate-limit` | ^11.2.0 | per-IP limits on /api/chat |
| `typebox` | ^1.3.12 | tool parameter schemas (SDK uses TypeBox) |
| `pg` | ^8.23.0 | Postgres driver |
| `pgvector` | ^0.3.0 | typed vector params |
| `bullmq` | ^6.0.11 | task queue (Redis) |
| `ioredis` | ^6.0.0 | Redis client for BullMQ |
| `openai` | ^7.4.0 | embeddings (text-embedding-3-large) |
| `cohere-ai` | ^8.0.0 | reranker |
| `@tavily/core` | ^0.7.7 | web search (primary) |
| `pdf-parse` | ^2.4.5 | PDF text extraction (pypdf equivalent) |
| `pino` | ^10.3.1 | logging (Fastify-native) |
| `dotenv` | ^17.4.2 | .env loading |
| dev: `typescript` | ^7.0.2 | compiler |
| dev: `tsx` | ^4.23.12 | run TS directly (npm run dev) |
| dev: `vitest` | ^4.1.10 | test suite (owner's call, after migration) |
| dev: `@types/node` | ^26.2.0 | Node types |

### 5.3 Open items (owner review)

- **Resolved (owner, 2026-08):** no auth, rate limiting required (§2.5); UI framework =
  React+Vite SPA (`docs/design/ui.md`); CFPB = full dump; tickets schema rebuilt from scratch
  (`docs/design/data-management.md`).
- MCP server export of retrieval tools (Phase 8) — scaffolded in `src/mcp/`, not built now.
- Durable turn orchestration across server restarts — deferred; documented in §3.5.
- Late chunking / bge-m3 / pg_search: eval-gated upgrades, not v2 scope.
