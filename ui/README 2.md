# ACS Web UI — Agentic Customer Support Console

React 19 SPA (Vite 7 + Tailwind v4 + zustand) that talks to the Fastify backend
on **port 8000**. This is a sibling of `src/` — its own npm project (the backend
stays the deployable core; the UI is a thin client).

## Quick start

```bash
# 1. Backend (repo root, separate terminal) — needs Postgres + Redis up
npm run dev            # Fastify on :8000 (PORT=8000 in .env)

# 2. UI (this folder)
cd ui
npm install
cp .env.example .env   # optional — VITE_API_URL is empty by default
npm run dev            # Vite on :5173, proxies /api + /health → :8000
```

Open http://localhost:5173.

## Scripts

| Script | What it does |
|---|---|
| `npm run dev` | Vite dev server (port 5173, HMR, proxy → :8000) |
| `npm run build` | `tsc --noEmit` + `vite build` → `dist/` |
| `npm run preview` | Serve the production build |
| `npm run test` | vitest (happy-dom, no network) |
| `npm run typecheck` | `tsc --noEmit` |

## Structure

```
ui/
├── vite.config.ts        # /api + /health proxy → http://localhost:8000
├── vitest.config.ts      # happy-dom, globals on (RTL auto-cleanup)
├── src/
│   ├── lib/              # api.ts (typed fetch), sse.ts (EventSource wrapper),
│   │                     # types.ts (mirrors the REAL backend payloads), format.ts
│   ├── stores/           # chatStore (SSE reducer), sessionStore, settingsStore
│   ├── hooks/            # useChatStream (SSE + rAF token batching), useDebounce, useHealth
│   ├── components/
│   │   ├── layout/       # AppShell, SessionSidebar, TopBar, ContextPanel
│   │   ├── chat/         # MessageList/Bubble, Composer, AgentStatusLine,
│   │   │                 # ToolActivityFeed, SourcesPanel
│   │   ├── tickets/      # TicketTable, TicketDrawer
│   │   ├── manuals/      # (routes) ManualBrowser / ManualDetail
│   │   └── common/       # Button, Badge, Spinner, Skeleton, EmptyState, ErrorBanner, Markdown
│   └── routes/           # chat, tickets, manuals, manual-detail, settings
└── index.html
```

## API contract this UI is built against

Real endpoints (`src/server/routes/*`), **not** the pre-backend sketch in
`docs/design/ui.md §6.2`:

| Endpoint | Used by |
|---|---|
| `POST /api/chat` `{message, conversationId?}` → `{chatId, conversationId, eventsUrl}` | Composer send (one chat per message; follow-ups reuse `conversationId`) |
| `GET /api/chat/:id/events` — SSE | `useChatStream` (event types: `turn_start, token, thinking, tool_start, tool_update, tool_end, turn_end, done, error, retry_start, retry_end, queue_update`) |
| `POST /api/chat/:id/cancel` / `steer` | Stop button (REST — **no WebSocket**; the design's `/ws` doesn't exist) |
| `GET /api/chat/:id/history` | Session reopen hydration |
| `GET /api/sessions` · `DELETE /api/sessions/:id` | Sidebar list + delete |
| `GET /api/tickets?q=&status=&page=&pageSize=` · `GET /api/tickets/:id` | Tickets route + TicketDrawer |
| `GET /api/manuals` · `GET /api/manuals/:id/chunks` | Manuals route + chunk viewer |
| `GET /api/models` · `GET /health` | TopBar model pill + connection pill |

## Design notes / known gaps (v1)

- **Streaming model:** one optimistic user bubble + one assistant bubble per
  send; `token` deltas are rAF-batched into a single store update per frame;
  `done` replaces the bubble text with the authoritative final message and
  attaches `sources[]`. An `error` event keeps the partial text and tags the
  bubble (code `canceled` → `cancelled`).
- **Model pill is UI-only.** The backend chooses the runtime model itself
  (`PI_MODEL` env); the chat POST body has no model field, so the picker is a
  preference for now (stored in settings).
- **Settings toggles are local-only** (zustand persist → localStorage). Wiring
  them into the agent's tool config is a future phase.
- **Sessions are in-memory on the backend** (ChatRegistry) — they reset on
  server restart, and the sidebar reflects that.
- **Reconnect caveat:** if a session is mid-stream when you reload, the SSE
  stream reconnects via `Last-Event-ID` replay but history hydration can double
  the tail of the last message. Rare and cosmetic.
- **EventSource** auto-reconnects with backoff; the wrapper dispatches typed
  events into the store. If auth headers are ever needed, swap the internals
  for `fetch` + `ReadableStream` + `eventsource-parser` behind the same
  `connectSse` interface (`src/lib/sse.ts`).

## Tests

`src/**/*.test.{ts,tsx}` — fast, no network:

- `lib/sse.test.ts` — dispatch by event type, malformed JSON, state transitions
  (fake EventSource injected via the ctor param)
- `lib/api.test.ts` — mocked `fetch`: query-string building, ApiError shape
- `stores/chatStore.test.ts` — the pure `reduceSseEvent` reducer (token append,
  tool feed, done finalize, error/cancel)
- `components/chat/MessageBubble.test.tsx` — markdown rendering, caret, sources,
  error/cancelled states (happy-dom + @testing-library/react)
