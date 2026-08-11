# Agentic Customer Support

A multi-agent customer-support system built on the **pi agents SDK** (Node/TypeScript).
Agents retrieve answers from three sources: **SQL (support tickets)**, **vector (technical
manuals / knowledge base)**, and **web search** — streamed live to a chat UI over SSE + WebSocket.

> **v2 status:** full rebuild in progress. The v1 prototype (Python, Google ADK) lives in git
> history (`legacy/` + prior commits) for reference. See `docs/plan.md` for the phased plan,
> `AGENTS.md` for project rules, `docs/design/` for the design docs.

## Stack

| Layer | Choice |
|---|---|
| Agent runtime | pi agents SDK (`@earendil-works/pi-coding-agent`, pinned) — in-process `createAgentSession()` |
| Backend | Fastify + `@fastify/sse` + `@fastify/websocket` |
| Retrieval | Postgres + pgvector (hybrid vector + FTS + RRF), Cohere/BGE reranker |
| Web search | Tavily (fallback DuckDuckGo) |
| Queue | BullMQ + Redis |
| UI | React + Vite SPA (zustand, Tailwind) — see `docs/design/ui.md` |

## Layout

```
AGENTS.md            pi project rules (auto-loaded)
lessons.md           owner learning log
docs/                research + design docs
src/runtime/         pi SDK wiring (isolated)
src/tools/           custom tools (kb search, tickets SQL, web search, route_to_agent)
src/guardrails/      input/context/tool_call/tool_result hooks
src/streaming/       SSE + WS bridges
src/server/          Fastify API
src/queue/           BullMQ jobs (ingest, evals)
src/retrieval/       hybrid search, embeddings, ingest
src/mcp/             MCP server exposing retrieval tools
legacy/              v1 Python code (reference only)
```

## Getting started

```bash
npm install
cp .env.example .env   # fill in DATABASE_URL, OPENAI_API_KEY, etc.
docker compose up -d   # Postgres+pgvector, Redis
npm run spike          # verify the pi SDK agent loop works in-process
npm run dev            # Fastify API (SSE/WS)
```

See `docs/plan.md` Phase 5+ for the implementation order.
