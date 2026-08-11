# Agentic Customer Support

An AI support assistant that answers customer questions by **retrieving the right information**
from three places at once:

1. **Support tickets** — a database of ~3.7M real consumer complaints (plus 8.4k synthetic tickets),
2. **Product manuals** — searchable document knowledge base, and
3. **The web** — live search when the other two don't have an answer.

Ask it something like *"How do I reset the Wi-Fi on my LG TV?"* or *"Are there refund complaints
about LG OLED TVs?"* and it retrieves from manuals and tickets, cites its sources, and streams
the answer live to a chat UI.

## Why this project exists

It's a personal learning project built to explore modern **retrieval techniques** end-to-end:

- **Hybrid search** — combining keyword matching with semantic (meaning-based) similarity, then
  merging both result sets so you get the best of each.
- **Ranking and relevance** — testing how to surface the most useful chunks/documents for a query.
- **Retrieval at scale** — the complaints database is 3.7M+ rows, which pushes every technique
  beyond toy-sized examples.
- **Agents that retrieve** — a small team of agents (tickets / knowledge base / web) that decide
  which source to consult and weave the results into an answer with citations.
- **Real-time streaming** — tokens and tool activity stream to the browser over SSE/WebSockets.

The whole system is built on the [pi agents SDK](https://pi.dev) with a TypeScript/Fastify backend
and a React UI.

## Usage

### Requirements

- Node.js ≥ 22, npm
- Docker (for Postgres + Redis)

### Setup

```bash
npm install
cp .env.example .env        # fill in at least OPENAI_API_KEY
docker compose up -d        # starts Postgres + Redis
npm run dev                 # API server on :8000
```

### UI (optional but recommended)

```bash
cd ui
npm install
npm run dev                 # open http://localhost:5173
```

### Load data (optional — the DB ships empty)

```bash
bash scripts/provision-data.sh   # downloads tickets + manuals (~10 MB, no CFPB)
npm run ingest                   # tickets + manuals → database
```

> The full CFPB complaints dump (1.4 GB) is optional — see `docs/design/data-management.md`.

### Try it

- **API:** `curl -X POST :8000/api/chat -d '{"message":"how do i reset wifi on my lg tv"}'`
  then follow the SSE stream at the returned `eventsUrl`.
- **Chat CLI:** `npm run chat` for a terminal REPL.
- **UI:** the chat view, a searchable tickets table, and a manual browser.

## Docs

All documentation lives in [`docs/`](docs/README.md): the phased plan, design docs, research,
and a learning log (`docs/lessons.md`) written for anyone picking up this stack for the first time.
