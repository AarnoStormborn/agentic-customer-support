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
  pending design validation. `AGENTS.md` created so rules persist across sessions.
- **[backend design]** Phase 3.1–3.3 design landed: **Fastify** chosen over Express/NestJS/Hono
  (SSE plugin + speed + TS). Agent loop: `route_to_agent` custom tool spawning child sessions;
  guardrails via `input`/`context`/`tool_call`/`tool_result` hooks. Retrieval: pgvector `<=>` +
  tsvector GIN + RRF, Cohere rerank, structural chunking, read-only SQL role.

---

## 8. Why Fastify won (backend framework decision)

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

## 9. React + Vite vs Next.js (the UI decision)

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
