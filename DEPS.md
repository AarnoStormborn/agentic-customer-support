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
