# UI Design — Agentic Customer Support v2

> Status: **Phase 3.5 design** (owner review before Phase 9 implementation)
> Scope: UI framework selection, page/screen design, state & data flow, styling system,
> project layout, and the API contract the UI expects from the Fastify backend.
> Companion docs: `docs/plan.md`, `docs/tech-stack-research.md`, `docs/design/backend-agent-retrieval.md` (Phase 4.3 — reconcile §6 when it lands).

---

## 1. UI framework evaluation (React + Vite vs Next.js vs alternatives)

### The question, restated

You know React. The app is a **chat-heavy, real-time support console** that talks to a
**separate Fastify API** over SSE (token/tool/source events) + WebSocket (steer/cancel/presence) +
REST (sessions, tickets, manuals). So the real question isn't "React or not" — it's **how the app
is built and served**: a plain client-side SPA, or a full-stack framework like Next.js.

### Candidates

| | **React + Vite (SPA)** ⭐ | **Next.js (App Router)** | React + Webpack/CRA | SvelteKit / Vue-Nuxt |
|---|---|---|---|---|
| What it is | Build tool + dev server; you write pure React, browser renders everything | Full-stack React framework: routing, SSR, Server Components, API routes, caching | Old-school React tooling | Other frameworks (different component model) |
| Dev experience | Near-instant dev server + HMR; zero magic | Great but heavier: App Router, Turbopack, file conventions | CRA is **deprecated**; slow, unmaintained | Good, but you must leave React |
| SSR / SEO | None (client-rendered) | First-class SSR/SSG — the main reason teams pick it | None | Varies |
| Streaming (SSE/WS) | Trivial: `EventSource`/`fetch` + `WebSocket` directly in the browser | Works, but App Router pushes you toward Server Components — and SSE must be consumed in **client** components anyway ("use client"), so SSR buys you nothing here | Same as Vite but slower tooling | Same as Vite |
| Learning curve for a React-only dev | **Lowest** — Vite adds ~1 new concept (it's just a bundler/dev server) | **Steep** — new mental models: server vs client boundary, `'use client'`, two caches, Server Actions | Low, but dead end | Medium+ (new framework) |
| Deployment | Static files (nginx or Fastify `@fastify/static`) — one less server | Node server (or serverless) running the whole framework | Static files | Static or server |
| Ecosystem | Huge, and Vite is the default for new React tooling (SvelteKit, Astro, Nuxt all use it) | Huge, Vercel-backed | Shrinking | Smaller (for you) |

### How the decision was made (evidence)

1. **SSR/SEO is irrelevant here.** This is an internal, authenticated tool — no public pages, no
   web crawlers. The main reason to choose Next.js (server rendering for "fast initial load times
   and SEO") doesn't apply ([Rollbar — Next.js vs Vite.js](https://rollbar.com/blog/nextjs-vs-vitejs/)).
2. **The backend already exists separately.** Vite's sweet spot is exactly "you already have a
   separate backend and just need a great UI setup" — same source:
   > "Choose Vite.js if… you are building a single-page app (SPA), want near-instant dev feedback
   > loops, or **already have a separate backend** and just need a great UI setup."
3. **Streaming is the core feature, and it's client-side regardless.** Token streaming is
   one-directional (server → client); SSE's `EventSource` gives auto-reconnect and typed events,
   which is why it beats WebSockets for ~80% of AI streaming UIs
   ([streaming-ai-web-apps](https://webdeveloper.com/learn/guides/streaming-ai-web-apps/),
   [SSE vs WS analysis](https://www.linkedin.com/posts/sarmalinux_nextjs-typescript-webdevelopment-activity-7469659572631216128-72wk)).
   In Next.js you'd still consume that stream in a client component; App Router adds server/client
   ceremony around a feature that never touches the server render path.
4. **Learning budget.** You're learning Node/TS/pi-SDK/Fastify/BullMQ at the same time. Next.js's
   App Router introduces genuinely new concepts (React Server Components, `'use client'`, two
   caches, Server Actions) that experienced React devs routinely trip over
   ([Contentful — Next.js vs React learning curve](https://www.contentful.com/blog/next-js-vs-react/),
   [Inngest — 5 lessons from App Router in production](https://www.inngest.com/blog/5-lessons-learned-from-taking-next-js-app-router-to-production)).
   A Vite SPA lets you keep 100% of your mental model: "React renders; my API is somewhere else."
5. **Counter-argument checked.** "Next.js beats React+Vite even for SPAs" (e.g.
   [dev.to — Why Next.js Beats React + Vite for SPAs](https://dev.to/axibord/why-nextjs-beats-react-vite-for-spas-its-not-just-about-seo-b9g))
   argues about built-in routing/data fetching and one deployable. For us, routing is one file
   (react-router), data fetching is a handful of `fetch` calls, and "one deployable" isn't a goal —
   the Fastify API is the deployable core and the UI is a thin client for it.

### Recommendation

> **React 19 + Vite 7, single-page app.** Learn one new tool (Vite — and it's just a dev server +
> bundler), keep React Router for pages, consume the Fastify API with `fetch` + `EventSource` +
> `WebSocket`. Revisit Next.js later *only if* we ever need SEO, server rendering, or a merged
> deployment — none of which are on the roadmap (AGENTS.md locked stack: backend is Fastify,
> streaming is SSE+WS, per `docs/plan.md` Phase 3.5).

### Non-goals / explicitly rejected

- **CRA (create-react-app)**: deprecated and unmaintained; Vite is its successor.
- **SvelteKit / Vue+Nuxt**: you'd have to learn a new component model for no benefit.
- **Next.js now**: adds a server-rendering layer the product doesn't need, for a chat UI whose
  interactivity lives entirely in the browser.

---

## 2. UI design

### 2.1 Application shell (three regions)

```
┌─────────────────────────────────────────────────────────────────────┐
│ SessionSidebar (240px, collapsible) │ TopBar: model pill, conn. status│
│ ┌──────────────────────────┐        ├────────────────────────────────┤
│ │ + New session            │        │                                │
│ │ Search sessions…         │        │  ChatArea (main)  │ ContextPanel│
│ │ ▸ Today                  │        │  message list     │ (320px,    │
│ │   • LG TV wifi reset    │        │  streaming cursor │ collapsible)│
│ │   • Ticket #402 refund  │        │  tool activity    │ sources /   │
│ │ ▸ Yesterday             │        │  composer         │ citations   │
│ │   • CFPB complaint…     │        │                   │             │
│ └──────────────────────────┘        └────────────────────────────────┘
└─────────────────────────────────────────────────────────────────────┘
```

- Left: **SessionSidebar** — new session, searchable history grouped by day, active-session
  highlight, model selector at the bottom.
- Center: **ChatArea** — the primary surface (see 2.2).
- Right: **ContextPanel** — live tool-activity feed + sources/citations for the active message;
  collapsible (auto-collapses on small screens).
- **TopBar** — model pill, connection indicator (SSE/WS health), settings gear, user avatar.

### 2.2 Chat interface (the core screen)

| Region | Behavior |
|---|---|
| **MessageList** | Vertical list of user/assistant turns; newest at bottom; auto-scroll with "pin to bottom" (pause scroll if user scrolls up); virtualize later if history grows. |
| **MessageBubble** | User: right-aligned, primary-tinted. Assistant: left-aligned, neutral card, rendered with `react-markdown` (GFM + code highlighting); inline citation markers `[1]` `[2]` are clickable superscripts that open the source in the ContextPanel. |
| **Streaming token display** | Assistant bubble shows a blinking `▍` caret while tokens arrive; text appends via SSE `token` events. Streaming is incremental — no flash-of-full-text. |
| **Typing / agent indicators** | Above the streaming bubble, a status line driven by SSE: `RAG agent: searching manuals…`, `SQL agent: querying tickets…`, `Web agent: fetching results…` (from `tool.started` / `agent.started` events), with per-agent spinner. |
| **ToolActivityFeed** | Collapsible stack of tool cards (icon, tool name, status `running → done`, elapsed ms, one-line result summary). Lives in ContextPanel; lets the user *see the agent think* — the key differentiator of an "agentic" UI. |
| **Sources / citations panel** | Each assistant message that used retrieval gets a "Sources" section: manual chunks (title, doc name, similarity/relevance badge) + matched tickets (id, subject, status). Clicking a **manual chunk** opens a read-only chunk viewer (full text + link to the manual browser). Clicking a **ticket** opens the TicketDrawer. |
| **Composer** | Auto-growing textarea, Enter to send / Shift+Enter newline, Stop button while streaming (POST cancel), model selector inline, message actions: regenerate, copy, thumbs. |

### 2.3 Tickets view

- Searchable, sortable **table** (id, subject, status, priority, created, source): server-side
  search (`/api/tickets?q=…&status=…&page=…`), debounced input, skeleton rows while loading.
- Clicking a row opens a **TicketDrawer** (right-side panel): full description, status history,
  "Ask the agent about this ticket" (pre-fills composer with ticket context), link to related
  chat sessions.

### 2.4 Knowledge / manual browser

- List of manuals (title, source, chunk count, ingestion date), searchable.
- Manual detail → ordered **chunk list**; each chunk shows its text excerpt + embedding metadata
  (vector id, position) for debugging retrieval quality.
- Chunk view: full text, "Ask about this chunk" (prefills composer with doc context), citation
  link target for the chat Sources panel.

### 2.5 Settings

- Default model + provider; retrieval toggles (SQL / vector / web, reranker on/off, top-K);
- Theme (dark/light — dark default); appearance (font size); connection info (API base URL).

### 2.6 Error / empty / loading / offline states

| State | Design |
|---|---|
| Empty (no session) | Welcome hero: what this tool does, suggested prompts, quick-start tiles. |
| Loading | Skeletons (rows for tables, bubbles for chat); never spinners-only. |
| Error | Inline `ErrorBanner` (title, message, Retry); streaming errors stop the stream gracefully and keep partial text with a "resume/cancel" choice. |
| Offline / disconnected | TopBar connection pill turns amber + auto-reconnect with backoff (EventSource does this natively); WS shows "reconnecting…". |
| Cancelled | Message bubble tagged `cancelled` (dimmed), composer re-enabled. |

### 2.7 Responsive behavior

- `≥1280px`: three columns (sidebar / chat / context).
- `768–1280px`: context panel becomes a drawer toggled by a "Sources" button in the chat header.
- `<768px`: sidebar off-canvas (hamburger); chat full-width; composer fixed at bottom.

### 2.8 Component tree (sketch)

```
App
└── Router (react-router)
    ├── ChatRoute
    │   └── AppShell
    │       ├── SessionSidebar (history, model select, + new)
    │       ├── ChatArea
    │       │   ├── ChatHeader (session title, model pill, cancel)
    │       │   ├── MessageList
    │       │   │   └── MessageBubble (user | assistant)
    │       │   │       ├── Markdown (+ citation links [1][2])
    │       │   │       └── StreamingCaret
    │       │   ├── AgentStatusLine (typing / per-agent indicator)
    │       │   └── Composer (textarea, send/stop, model picker)
    │       └── ContextPanel
    │           ├── ToolActivityFeed
    │           │   └── ToolActivityCard
    │           └── SourcesPanel
    │               ├── SourceChunkCard → ChunkDrawer
    │               └── SourceTicketCard → TicketDrawer
    ├── TicketsRoute → TicketTable (+ TicketDrawer)
    ├── ManualsRoute → ManualBrowser → ManualDetail → ChunkView
    └── SettingsRoute
common/: Button, Badge, Spinner, Skeleton, EmptyState, ErrorBanner, Modal, Drawer, Tooltip
```

---

## 3. State & data

### 3.1 State management: **zustand**

| | Plain React (useState/useReducer + Context) | **zustand** ⭐ | Redux Toolkit |
|---|---|---|---|
| Boilerplate | Low at first, grows with shared state | Tiny (~1 KB, no providers) | High (slices, thunks, middleware) |
| Re-render control | Context re-renders every consumer on any change | Selectors — only subscribed components re-render | Fine, but you pay for it in ceremony |
| Streaming-friendly (10–60 Hz token updates) | Workable but easy to thrash the tree | Best-in-class for high-frequency updates | OK but overkill |
| Learning curve | You know it | ~15 min (one `create()`) | New paradigm (actions, reducers, selectors) |

**Why:** chat state (messages, streaming text, tool activity, sources) updates many times per
second and is read by many components (bubbles, status line, activity feed, sources panel).
zustand lets each of those subscribe to only the slice it needs, with zero provider nesting, and
ships a `persist` middleware for settings. Redux Toolkit would be the "enterprise" answer but adds
a whole paradigm to a project whose backend is already the complicated half. Plain React state is
fine for *forms* and local UI — we use it there.

### 3.2 Stores (all typed, single source of truth)

```
stores/
  chatStore.ts     — messages[], streaming text, agent status, activity[], sources[], send/cancel actions
  sessionStore.ts  — session list, activeSessionId, model, create/select/delete
  settingsStore.ts — model default, retrieval toggles, theme (persisted to localStorage)
```

### 3.3 Consuming SSE + WebSocket + REST

```
Browser ── REST (fetch) ──────────→ Fastify :3000/api/*      (sessions, tickets, manuals, settings)
Browser ── SSE (EventSource) ─────→ /api/chat/:id/events     (token, tool, source, turn, done, error)
Browser ── WebSocket ─────────────→ /ws                       (steer, cancel, presence, session updates)
```

- **SSE**: `EventSource` in `lib/sse.ts` — auto-reconnects, dispatches events by `event:` type
  into `chatStore`. Caveat: `EventSource` can't send custom headers/POST; if the backend later
  needs an auth header, swap to `fetch` + `ReadableStream` + `eventsource-parser` behind the same
  `lib/sse.ts` interface (the UI won't change).
- **WebSocket**: thin `useSocket` hook wrapping native `WebSocket` with reconnect + backoff;
  used for steer/cancel and presence, not token streaming.
- **REST**: `lib/api.ts` — typed `fetch` wrappers (`GET/POST/DELETE`), one `ApiError` shape,
  base URL from `import.meta.env.VITE_API_URL`. Optionally add `@tanstack/react-query` in Phase 9
  for tickets/manuals caching (debounced search, invalidation) — recommended but not required to start.
- **Dev proxy**: `vite.config.ts` forwards `/api` and `/ws` to `localhost:3000` (Fastify), so the
  UI never hard-codes CORS in dev.

### 3.4 Optimistic updates for chat

1. User hits Send → **immediately** append `{role:'user', status:'pending'}` + assistant
   `{role:'assistant', status:'streaming', text:''}`.
2. `POST /api/chat/:id/messages` succeeds → mark user message `sent`.
3. SSE `token` events append to the assistant text (immutable updates, batched via `requestAnimationFrame` for 60 fps rendering).
4. `tool.*` / `agent.*` events update status line + activity feed; `source.found` populates sources.
5. `message.completed` → status `done`; `error` → keep partial text, mark `error`; `cancel` → `cancelled`.
6. On failure before any event: roll back to a retryable error bubble.

### 3.5 Persistence

- `settingsStore` → `localStorage` (zustand `persist`): model, theme, toggles.
- Sessions/messages → server-side (session registry + Postgres, per plan) — the sidebar hydrates
  from `GET /api/sessions`; the UI is a stateless client.
- Active session + view location → URL params (`/chat/:sessionId`) so refresh/browser-back work.

---

## 4. Styling

### 4.1 Choice: **Tailwind CSS v4**

| | CSS Modules | styled-components | **Tailwind v4** ⭐ |
|---|---|---|---|
| Runtime cost | None | JS runtime (style injection) | None (compiled at build) |
| Learning | You already know CSS — but you write custom CSS per component | New API (tagged templates), theming via ThemeProvider | Utility classes — one-time vocabulary, huge payoff |
| Tokens/theme | Manual (`:root` vars) | `ThemeProvider` | First-class `@theme` in CSS (v4) |
| Fits a learning project | Yes, fine | Adds JS concepts for no win | Best ratio of speed → consistency |

**Why:** utility-first CSS compiles to static CSS (no runtime), keeps styles colocated with markup,
and v4's CSS-first `@theme` gives us a single token source. CSS Modules would work, but you'd write
a custom design system by hand; styled-components adds a JS runtime + new API for something CSS
already does. Tailwind is also the community default for new Vite+React apps, so tutorials abound.
(Note: Tailwind **v4** changed config to CSS-first — most online tutorials show v3's
`tailwind.config.js`; use the [v4 docs](https://tailwindcss.com/docs/styling-with-utility-classes).)

### 4.2 Minimal design language (dark-first)

```css
/* styles/index.css (Tailwind v4 @theme) */
@theme {
  /* Color — neutral canvas + one accent */
  --color-canvas:    #0f1115;   /* app background   */
  --color-surface:   #16181d;   /* panels/cards     */
  --color-surface-2: #1e2128;   /* hover/insets     */
  --color-border:    #2a2e37;
  --color-accent:    #6366f1;   /* indigo-500       */
  --color-accent-soft: #e0e7ff;
  --color-text:      #e4e7ec;
  --color-text-dim:  #8a919e;
  --color-ok:  #34d399;  --color-warn: #fbbf24;  --color-danger: #f87171;

  /* Type — system stack + mono for code/tokens */
  --font-sans: ui-sans-serif, system-ui, "Segoe UI", Roboto, sans-serif;
  --font-mono: ui-monospace, "Cascadia Code", "JetBrains Mono", monospace;

  /* Spacing — 4 px base grid (Tailwind default) */
  --spacing-1: 4px; … /* standard scale */

  /* Shape */
  --radius-md: 8px;  --radius-lg: 12px;
  --shadow-panel: 0 4px 24px rgb(0 0 0 / .35);
}
```

- **Type scale**: 12 / 14 / 16 / 18 / 24 / 32 px (`text-xs`…`text-3xl`); mono for IDs, code,
  tool names.
- **Rules**: one accent color only; semantic colors only for status; borders instead of shadows for
  structure; dark default + light mode via a `dark:`-free token swap (a `theme` class on `<html>`).
- **Icons**: `lucide-react` (tree-shaken, consistent).
- **Motion**: 150 ms ease-out for hover/focus; 250 ms for drawers/panels; respect
  `prefers-reduced-motion`.

---

## 5. Project structure

Sibling folder `ui/` (its own `package.json` — the repo stays one npm project per app, matching
the "Fastify backend = deployable core, UI = thin client" split; a monorepo workspace can come later).

```
ui/
├── package.json  vite.config.ts  tsconfig.json  index.html  .env.example (VITE_API_URL)
└── src/
    ├── main.tsx               # React root
    ├── App.tsx                # Router + layout routes
    ├── routes/                # one file per view
    │   ├── chat.tsx  tickets.tsx  manuals.tsx  settings.tsx
    ├── components/
    │   ├── layout/    AppShell.tsx  SessionSidebar.tsx  TopBar.tsx  ContextPanel.tsx
    │   ├── chat/      MessageList.tsx  MessageBubble.tsx  Composer.tsx
    │   │              AgentStatusLine.tsx  ToolActivityFeed.tsx  SourcesPanel.tsx
    │   ├── tickets/   TicketTable.tsx  TicketDrawer.tsx
    │   ├── manuals/   ManualBrowser.tsx  ManualDetail.tsx  ChunkView.tsx
    │   └── common/    Button.tsx  Badge.tsx  Spinner.tsx  Skeleton.tsx
    │                  EmptyState.tsx  ErrorBanner.tsx  Markdown.tsx  Drawer.tsx
    ├── hooks/         useSSE.ts  useSocket.ts  useChatStream.ts  useDebounce.ts
    ├── stores/        chatStore.ts  sessionStore.ts  settingsStore.ts
    ├── lib/           api.ts  sse.ts  ws.ts  types.ts  format.ts
    └── styles/        index.css   # Tailwind v4 @theme + base layer
```

### Dependency list (pinned exact versions at install; `^` ranges shown here for reference)

| Package | Version | Why |
|---|---|---|
| `react` / `react-dom` | `^19.2.8` | UI runtime |
| `vite` | `^7.3.6` | Dev server + bundler (v8 exists; stay on 7 for plugin stability) |
| `@vitejs/plugin-react` | `^5.2.0` | React fast-refresh (supports Vite 7) |
| `typescript` | `^5.9.3` | Types (TS 7 native compiler exists; not yet default) |
| `react-router` | `^7.18.2` | Routing (v7 merged `react-router-dom` into `react-router`) |
| `zustand` | `^5.0.14` | Chat/session/settings stores |
| `tailwindcss` + `@tailwindcss/vite` | `^4.3.3` | Styling + Vite plugin |
| `react-markdown` + `remark-gfm` | `^10.x` / `^4.x` | Assistant message rendering, GFM tables/links |
| `lucide-react` | latest | Icons |
| `eventsource-parser` | `^4.0.0` | Only if EventSource is replaced by fetch-stream (auth headers) |
| `@tanstack/react-query` | `^5.101.4` | Optional (Phase 9) — tickets/manuals caching |

Dev-only: `@types/react`, `@types/react-dom`, `@types/node`.

**Runs:** `npm run dev` (Fastify :3000) + `cd ui && npm run dev` (Vite :5173, proxying `/api`+`/ws`).

---

## 6. API contract (what the UI expects from the backend)

> ⚠️ Reconcile with `docs/design/backend-agent-retrieval.md` (Phase 4.3) when it lands — this is the
> UI-side requirement list, aligned with the `docs/plan.md` Phase 3.1 sketch and the pi SDK event
> stream (`message_update`, `tool_execution_*`, `turn_*`, `agent_*`).

### 6.1 REST endpoints

| Method & path | Purpose |
|---|---|
| `GET /api/health` | Liveness + SSE/WS capability flags (TopBar connection pill) |
| `GET /api/models` | Available models + streaming support (model picker) |
| `POST /api/chat` | Create session + first message → `{ sessionId, messageId }` |
| `POST /api/chat/:sessionId/messages` | Send a message → `{ messageId }` (stream follows on SSE) |
| `GET /api/chat/:sessionId/events` | **SSE** stream (below) |
| `POST /api/chat/:sessionId/cancel` | Stop current run (Composer Stop button) |
| `POST /api/chat/:sessionId/steer` | Mid-stream steer (via WS too; REST fallback) |
| `GET /api/chat/:sessionId` | Message history (rehydrate on session open) |
| `GET /api/sessions` · `DELETE /api/sessions/:id` | Sidebar history |
| `GET /api/tickets?q=&status=&page=&pageSize=` | Ticket table (server-side search/pagination) |
| `GET /api/tickets/:id` | TicketDrawer detail |
| `GET /api/manuals` · `GET /api/manuals/:id` | Manual browser |
| `GET /api/manuals/:id/chunks/:chunkId` | Single chunk (sources panel → chunk viewer) |
| `GET /api/settings` · `PUT /api/settings` | Defaults + retrieval toggles |

### 6.2 SSE event types (`event:` field on `/api/chat/:sessionId/events`)

| event | Payload (shape) | UI reaction |
|---|---|---|
| `message.started` | `{ messageId }` | Open assistant bubble |
| `token` | `{ messageId, text }` (delta) | Append to streaming bubble |
| `agent.started` / `agent.finished` | `{ agent: "rag"\|"sql"\|"web" }` | AgentStatusLine |
| `tool.started` / `tool.progress` / `tool.result` | `{ tool, status?, summary? }` | ToolActivityFeed cards |
| `source.found` | `{ type: "manual"\|"ticket", chunk?\|ticket?, score? }` | SourcesPanel entries |
| `turn.started` / `turn.finished` | `{ turnId }` | Turn boundary (status line reset) |
| `message.completed` | `{ messageId, sources: […] }` | Finalize bubble + citations |
| `error` | `{ message, recoverable }` | ErrorBanner / graceful partial stop |
| `done` | `{}` | Clear streaming state, re-enable composer |

### 6.3 WebSocket (`/ws`)

- Client → server: `steer { text }`, `cancel`, `ping`.
- Server → client: `presence` (user/agent online), `session.updated` (sidebar refresh from other
  tabs/users), `ack`.

### 6.4 Hard requirements (from AGENTS.md)

- Streams **must** carry typed `event:` names (not one generic `message`) — the UI branches on type.
- `token` deltas only — never full re-sends per token.
- `cancel`/`steer` must return quickly (< 200 ms) and take effect within one turn.
- Errors are **events on the SSE stream**, not just HTTP codes — the UI recovers mid-stream.

---

## 7. Open questions for the owner

1. Auth: is this a single-user local tool (no auth needed) or multi-user (then SSE needs a token
   header → `eventsource-parser` + fetch-stream path)? Default assumption: single-user, dev-only.
2. `@tanstack/react-query`: adopt now (caching discipline from day 1) or later? Default: later.
3. Dark-only vs dark+light theme for v1? Default: dark-only first, tokens designed so light is a
   config swap later.

## 8. Sources

- Rollbar — *Next.js or Vite.js: Which Framework is Better, and When?*: https://rollbar.com/blog/nextjs-vs-vitejs/
- Contentful — *Next.js vs. React: The difference and which framework to choose*: https://www.contentful.com/blog/next-js-vs-react/
- Web Developer — *Streaming AI in Web Apps (Vercel AI SDK 5, SSE, Edge Runtime)*: https://webdeveloper.com/learn/guides/streaming-ai-web-apps/
- Sarma Kaza (LinkedIn) — *Most AI chat apps use WebSockets. They shouldn't.*: https://www.linkedin.com/posts/sarmalinux_nextjs-typescript-webdevelopment-activity-7469659572631216128-72wk
- dev.to — *Why Next.js Beats React + Vite for SPAs (It's Not Just About SEO)*: https://dev.to/axibord/why-nextjs-beats-react-vite-for-spas-its-not-just-about-seo-b9g
- Inngest — *5 Lessons Learned From Taking Next.js App Router to Production*: https://www.inngest.com/blog/5-lessons-learned-from-taking-next-js-app-router-to-production
- Vite docs: https://vite.dev/guide/ · Tailwind v4 docs: https://tailwindcss.com/docs/styling-with-utility-classes
- zustand: https://github.com/pmndrs/zustand · react-router v7: https://reactrouter.com/
- pi SDK event stream (`message_update`, `tool_execution_*`, `turn_*`, `agent_*`): `docs/tech-stack-research.md` §sdk
