# Learnings from v1 (Python/ADK) — what to carry into v2

Distilled from `docs/project-analysis.md` (full review) + git history. The v1 code is preserved
in `legacy/` and git history; these are the operational lessons.

## What broke / what to avoid

1. **Dependency name collisions** — v1 pinned both `box==0.1.5` (an unrelated, metadata-less PyPI
   package) and `python-box==7.3.2`. `box` shadowed `python-box`, so `from box import ConfigBox`
   failed at import time and the whole app was un-runnable (P0). **Lesson:** pin one canonical
   package, verify `import` works in CI, never let a stray transitive dep shadow a real one.
2. **f-string SQL** — `rag_agent.py` interpolated `top_k` directly into SQL (`LIMIT {top_k}`) and
   the SQL agent executed arbitrary LLM-generated queries against a table with PII. **Lesson:**
   parameterize every query; enforce a read-only role + SELECT-only allowlist at the tool boundary.
3. **Sync DB engine inside async tools** — tools used `sqlalchemy.create_engine(...)` inside async
   functions, blocking the event loop. **Lesson:** async pools (`pg.Pool` in v2) at module scope.
4. **Hand-rolled framework bridge** — v1 built a 239-line LiteLLM↔ADK adapter; its streaming path
   dropped tool calls and never set `turn_complete`. **Lesson:** don't bridge two frameworks by
   hand; pick one runtime (v2: pi SDK) and use its native stream.
5. **Dead code with wrong imports** — `guardrails/input_rails.py` still imported the removed
   OpenAI Agents SDK (wrong path), was never wired, and `src/api/*` were docstring stubs.
   **Lesson:** delete or wire — no placeholder modules.
6. **No data, no env, no container** — `config/data/` was referenced but never committed
   (gitignored); no `.env.example`; Dockerfile was 0 bytes; docker-compose booted Postgres with
   blank credentials. **Lesson:** `.env.example` + provisioning scripts + a working Dockerfile
   are part of the deliverable, not an afterthought.
7. **Ad-hoc tests** — 5 print-based scripts, no pytest, no CI. **Lesson:** real test suite
   (vitest in v2) with mocked LLM/DB, plus CI.

## What worked (keep the pattern)

- **Router → sub-agent topology** — a supervisor that routes to RAG / SQL / web agents was the
  right shape for support queries; v2 keeps it (via `route_to_agent` custom tool).
- **pgvector + HNSW index with inner-product (`<#>`) queries** — fast, simple, in-Postgres.
  v2 upgrades to cosine + hybrid FTS + RRF.
- **Async batch embedding ingestion** — batching API calls (100/chunk batch) avoided timeouts.
- **Schema-documented SQL agent** — injecting the tickets schema into the prompt improved query
  quality; v2 keeps schema-aware prompting but adds a hard read-only guard.
- **YAML-driven agent prompts** (`config/agents.yml`) — separating prompts from code was clean;
  v2 moves these to prompt files / system-prompt overrides.

## Carry-over facts

- Tickets domain from `legacy/config/schema.yml`: products (LG Smart TV, iPhone, Sony PlayStation,
  Google Pixel, HP Pavilion, LG OLED, Dell XPS, Sony Xperia), ticket types (Refund request, Billing
  inquiry, Product inquiry, Cancellation request, Technical issue), priorities (Critical/High/
  Medium/Low), channels (Social Media/Email/Phone/Chat). v2 `tickets` DDL extends this
  (`docs/design/data-management.md` §2).
- RAG chunking baseline from v1: chunk_size=1024, overlap=50 → v2: structural/section-based with
  metadata (`docs/design/backend-agent-retrieval.md` §4.4).
