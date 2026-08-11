# Project Analysis

**Repo:** `/Users/harshsingh/Documents/personal/agentic-customer-support`
**Analyzed:** 2026-08-11 · read-only review (no code modified)
**Scope:** full repo — README, pyproject.toml, uv.lock, main.py, src/ (all files), config/ (all files), tests/ (all files), docker-compose.yml, Dockerfile, .gitignore, git history (14 commits, 2025-03-29 → 2026-01-04).
**Method:** static review + empirical verification in a throwaway venv (`/tmp/acs-verify`) using the exact versions resolved in `uv.lock` (python-box, box, google-adk 1.3.0, duckduckgo-search 8.1.1, litellm 1.80.11, etc.). Every claim marked **[verified]** was executed against those installed packages.

---

## 1. Executive Summary

This is a multi-agent customer-support prototype: a supervisor agent that routes queries to three sub-agents (RAG/knowledge-base, SQL/tickets, web search), running on **Google ADK** (migrated from the OpenAI Agents SDK in commit `916af94`) with a hand-written **LiteLLM model adapter** bridging ADK ↔ any LiteLLM-backed model. The RAG ingestion pipeline, pgvector schema, agent prompts, and five simulation scripts exist and are mostly coherent.

**The project is currently un-runnable.** A dependency conflict between `box==0.1.5` (an unrelated, metadata-less PyPI package) and `python-box==7.3.2` makes `from box import ConfigBox` fail, which breaks `src/utils.py:5` → `src/agent_team/rag_agent.py` → `main.py` **at import time** — verified empirically. This single line is the P0 blocker.

Beyond that, the state is roughly:

| Area | Status |
|---|---|
| Agents (supervisor + RAG + SQL + web) on google-adk 1.3.0 | Implemented; API surface verified compatible with the installed ADK version |
| LiteLLM model adapter (`src/models/litellm_model.py`, 239 lines) | Implemented; non-streaming function-calling path works; streaming path drops tool calls |
| RAG ingestion pipeline (`config/ingest.py`) | Implemented but **cannot run**: depends on `config/data/` (never committed, gitignored) and has fragility bugs |
| pgvector schema + HNSW index | Implemented, consistent with ORM |
| **OpenAI Agents SDK** (README claim) | **Not used** — migrated away; one stale file still imports it |
| **OpenAI MCP** (README claim) | **Not present anywhere** |
| **WebSockets real-time comms** (README claim) | **Not present** (websockets only transitive via ADK) |
| **SSE real-time logging** (README claim) | **Not present** (sse-starlette only transitive; ADK has native SSE streaming, unused) |
| **FastAPI + async task queue** (README claim) | **Not implemented** — `src/api/routes.py`/`dependencies.py` are docstring-only stubs; `redis` dep unused |
| Guardrails | Dead code — wrong import path, imports removed SDK, never wired |
| Dockerfile | **Empty (0 bytes)** |
| Tests | 5 ad-hoc print-based scripts, not pytest |

**One positive headline:** all Google-ADK API surface the code uses (`Agent`, `Runner`, `FunctionTool(transfer_to_agent)`, `BaseLlm(LlmRequest, LlmResponse)`, `InMemorySessionService`, `genai` types, `Runner.run(user_id, session_id, new_message)`) exists and behaves as expected in google-adk 1.3.0 **[verified]**. Fixing the `box` line is sufficient to get `main.py` importing again.

---

## 2. Architecture Overview

### 2.1 Components

```
config/agents.yml        → agent names + prompts (support_agent, web_agent, sql_agent, rag_agent, guardrail_agents)
config/schema.yml        → tickets-table documentation injected into the SQL agent's prompt
config/schemas.py        → SQLAlchemy ORM: Document (t_docs) + DocumentChunk (t_docs_chunks, Vector(1536))
config/ingest.py         → CSV→tickets table (pandas) + PDF→chunks→embeddings→pgvector (async)
config/sql/pgvector.sql  → DDL + HNSW index (vector_ip_ops, matches <#> query)
config/sql/retrieval.sql → unused parameterized retrieval template

main.py                  → CLI REPL: builds the 4 agents, Runner, InMemorySessionService, input() loop

src/models/litellm_model.py → LiteLLMModel(BaseLlm): converts ADK LlmRequest→LiteLLM messages,
                              converts Google genai Schema→JSON Schema, maps tool calls both ways
src/agent_team/support_agent.py → supervisor: Agent with FunctionTool(transfer_to_agent) + monkeypatched
                                  declaration listing valid sub-agent names; sub_agents=[rag, sql, web]
src/agent_team/rag_agent.py    → retriever_tool: embed query → pgvector <#> (negative inner product) top-k
src/agent_team/sql_agent.py    → run_sql_queries: executes LLM-generated SQL on DB_STRING
src/agent_team/web_agent.py    → web_search: DDGS().text(query, max_results=5)
src/agent_team/guardrails/     → dead: input_rails.py (OpenAI SDK style) + broken __init__ import path
src/logger.py, src/exception.py, src/utils.py (embeddings, ConfigBox), src/api/* (stubs)
```

### 2.2 Data flow (as designed)

```
User input (CLI REPL)                        [main.py:76-95]
  → InMemorySessionService (in-memory, per-process)
  → Runner → support_agent (supervisor)
      → LiteLLMModel.generate_content_async → litellm.acompletion(model=OPENAI_MODEL, tools=[...])
      → model decides: transfer_to_agent(agent_name)   [support_agent.py:24, 41-49]
      → sub-agent selected
          ├─ database_agent → run_sql_queries → Postgres (DB_STRING) → str(result.all())
          ├─ knowledge_base_agent → retriever_tool → openai embeddings → pgvector top-k
          └─ web_search_agent → DDGS().text() → DuckDuckGo
      → sub-agent answers, then transfer_to_agent('customer_support_agent') back (prompt-driven)
  → text streamed back to REPL

Offline pipeline [config/ingest.py]:
  tickets.csv → pandas.to_sql → "tickets" table
  manuals/*.pdf → pypdf → fixed-size chunks (1024/50 overlap) → OpenAI embeddings (text-embedding-3-small)
               → t_docs + t_docs_chunks (pgvector, HNSW)
```

Planned (README) but **absent**: FastAPI HTTP layer, SSE logging stream, WebSocket channel, async task queue, MCP tooling. ADK 1.3.0 already ships `StreamingMode.SSE` and the lock file already contains fastapi/uvicorn/sse-starlette/websockets **transitively** — so the scaffolding for the README's roadmap is accidentally present and mostly needs direct pinning + code.

### 2.3 Key design decisions found in code

- **LiteLLM adapter instead of ADK's built-in models** — gives provider flexibility (any LiteLLM route) but is ~200 lines of hand-rolled message/schema conversion that must track two evolving SDKs.
- **Routing via `transfer_to_agent`**, prompt-driven (config/agents.yml), with a monkeypatch that appends valid target names to the tool description (support_agent.py:41-49).
- **SQL agent executes arbitrary LLM-generated SQL** — intentional for the demo, but unconstrained (see 4.2).
- **In-memory sessions** — nothing persistent; restart loses conversation history.

---

## 3. Implementation Status vs Planned Scope

| Planned (README.md) | Status | Evidence |
|---|---|---|
| Multi-agent team (KB/RAG, DB/SQL, Web) + supervisor | ✅ Implemented (google-adk) | src/agent_team/*, main.py |
| OpenAI Agents SDK | ❌ Migrated away; one stale import remains | git `916af94`; src/agent_team/guardrails/input_rails.py:1-7 |
| OpenAI MCP | ❌ Not present | grep for mcp → no hits |
| Real-time comms with WebSockets | ❌ Not present | no ws/websocket code; websockets==15.0.1 only transitive |
| Real-time logging with Server-Sent Events | ❌ Not present | no SSE code; sse-starlette==2.2.1 only transitive; ADK `StreamingMode.SSE` unused |
| Async task queueing with FastAPI | ❌ Not implemented | src/api/routes.py:1 & dependencies.py:1 empty; redis==7.1.0 dep unused |
| — (not in README) pgvector RAG ingestion | ✅ Implemented, not runnable | config/ingest.py (needs config/data/) |
| — (not in README) guardrails | ❌ Dead code | config/agents.yml:75-83 defined, never wired |

**Implicitly promised by repo structure but empty:** `Dockerfile` (0 bytes), `src/core/__init__.py` (was `src/api/app.py`, emptied in commit `b9c5fd2`), `src/api/routes.py`, `src/api/dependencies.py`, `src/services/__init__.py`.

---

## 4. Code Review Findings

### 4.1 Bugs

| # | Finding | Location | Severity |
|---|---|---|---|
| B1 | **`from box import ConfigBox` fails — app cannot import.** `box==0.1.5` is an unrelated PyPI package (anonymous upload, 2024-11-27, no metadata) whose files clobber python-box's `box/` package. **[verified]** — clean venv with both installed: `ImportError: cannot import name 'ConfigBox' from 'box'`; with only `python-box==7.3.2`: works. This breaks `main.py`, every agent module, and every test at import time. | src/utils.py:5; pyproject.toml:13 + :23 | **Critical (P0)** |
| B2 | `retriever_tool`: `if result:` on a SQLAlchemy 2.0 `Result` is always truthy → the `else: "No results found"` branch (line 54) is unreachable; empty result sets return `"[]"`. | src/agent_team/rag_agent.py:49-55 | Medium |
| B3 | `retriever_tool` returns `None` (bare `return`) when embedding generation fails — the LLM receives an empty tool result instead of an error message. | src/agent_team/rag_agent.py:29-32 | Medium |
| B4 | `config/schemas.py`: `default=datetime.now()` is evaluated once at class-definition time → every row gets the import-time timestamp (should be `datetime.now` without parens). | config/schemas.py:21, 37 | Medium |
| B5 | Ingest assigns **manual IDs** (`last_chunk_id`, `doc_id=i+1` based on `os.listdir` order) while the tables use `BIGSERIAL`. Re-running `upsert_docs` (despite the name) hits duplicate-key errors — the sequence never advances and doc order is filesystem-dependent. | config/ingest.py:168-173, 212-222 | High |
| B6 | `run_sql_queries` on non-SELECT statements: `result.all()` is undefined behavior for DML and nothing commits (autobegin rolls back on close) — confusing for INSERT/UPDATE attempts. | src/agent_team/sql_agent.py:31-36 | Low |
| B7 | `LiteLLMModel` streaming path (`_convert_chunk_to_response`) only reads `delta.content` — tool-call deltas are dropped, so in stream mode the agents can never call tools. Not hit today (ADK defaults `streaming_mode=NONE` **[verified]**, run_config.py) but breaks the README's SSE plan. Also `turn_complete` is never set (commented out at line 228). | src/models/litellm_model.py:228, 230-239 | High (latent) |
| B8 | `support_agent` monkeypatch: `"Valid agent_name values" not in decl.description` would raise `TypeError` if `description` were `None`. Works today because ADK 1.3.0's `transfer_to_agent` declaration has a non-None description **[verified]**. | src/agent_team/support_agent.py:44 | Low |
| B9 | Tests use substring heuristics for pass/fail (e.g. "sql_answered" = response contains "total"/"tickets") — false positives/negatives likely; no asserts anywhere. | tests/test_back_transfer.py:60-66, 82-90 | Medium |
| B10 | `config/ingest.py` `from schemas import ...` (line 15) resolves only because Python adds the script's own dir to `sys.path` when run as `python config/ingest.py`; running as a module (`python -m config.ingest`) from the repo root fails. | config/ingest.py:15 | Medium |

### 4.2 Security

| # | Finding | Location | Severity |
|---|---|---|---|
| S1 | **SQL injection via f-strings** in RAG retrieval: `top_k` (LLM-controlled argument, despite `int` annotation) is interpolated directly into the SQL string, and the embedding is pasted as `'{str(embedding)}'::vector`. An LLM/attacker controlling `top_k` can inject arbitrary SQL. | src/agent_team/rag_agent.py:38-42 | **High** |
| S2 | **Arbitrary SQL execution by design, with no guardrails**: the SQL agent executes whatever query the model generates against `DB_STRING` — including DROP/TRUNCATE/DELETE, and the tickets table contains PII (customer_name, email — schema.yml:2-3). No read-only enforcement (transaction rollback, statement allowlist, or read-only DB role). | src/agent_team/sql_agent.py:29-36 | **High** |
| S3 | **SQL injection in parameterized-looking template**: `config/sql/retrieval.sql:3` mixes a named parameter with a cast (`:embedding::vector`) — even if it were wired up, SQLAlchemy `text()` would emit `:embedding::vector` which is not safely parameterized. (File is unused.) | config/sql/retrieval.sql:3 | Low |
| S4 | Broad `except:` swallowing in `generate_embeddings` (utils.py:24) and ingest methods hides failures (embeddings return `[]`/`None`, pipeline "completes" with empty data). | src/utils.py:24-25; config/ingest.py:88, 134, 193, 239 | Medium |
| S5 | `OPENAI_MODEL`/`OPENAI_EMBEDDINGS`/`DB_STRING` come from env with no validation; `DB_STRING=None` would reach `create_engine(None)` with an opaque traceback. | src/agent_team/sql_agent.py:29-30, rag_agent.py:36 | Low |
| S6 | `docker-compose.yml` interpolates `POSTGRES_*` from the environment with no `.env` — `docker compose config` warns **all three are blank** **[verified]**, so the DB container would start with empty credentials, then the app's `DB_STRING` can't match. | docker-compose.yml:6-9 | High (ops) |

### 4.3 Dead code

| # | Finding | Location | Severity |
|---|---|---|---|
| D1 | **Guardrails package is entirely dead**: `__init__.py` imports from `src.guardrails.input_rails` (module lives at `src/agent_team/guardrails/input_rails.py` — path is wrong); `config/agents.yml:75-83` defines `guardrail_agents` but nothing reads that key; `init_supervisor_guardrail` is never called. | src/agent_team/guardrails/__init__.py:1; config/agents.yml:75-83 | Medium |
| D2 | `redis>=6.0.0` dependency — **zero usages** anywhere in code (no queue, no cache). | pyproject.toml:25; grep → no hits | Low |
| D3 | `pyaml` dependency — never imported; only `yaml` (from PyYAML, a transitive dep) is used. | pyproject.toml:21 | Low |
| D4 | `config/sql/retrieval.sql` — never referenced by any code. | grep for "retrieval.sql" → no hits | Low |
| D5 | `src/utils.py:read_config` + `ConfigBox` — `read_config` has no callers (the only reason `box` is imported). | src/utils.py:7-11 | Low |
| D6 | Unused imports: `sys` (main.py:3), `tool_context` (support_agent.py:6), inner `import json` (litellm_model.py:68, 87 — already imported at line 6). | main.py:3; support_agent.py:6; litellm_model.py:68, 87 | Low |
| D7 | Empty modules: `src/core/__init__.py` (0 B — was `src/api/app.py`), `src/services/__init__.py`, `src/api/__init__.py`, `src/models/__init__.py`, `src/agent_team/guardrails/__init__.py` (1 line). | — | Low |

### 4.4 Inconsistencies

| # | Finding | Location | Severity |
|---|---|---|---|
| I1 | **SDK mismatch**: README says "designed using OpenAI Agents SDK"; the code migrated to google-adk (commit `916af94`), but `guardrails/input_rails.py` still imports the removed SDK (`from agents import Agent, Runner, ...`). `openai-agents` is **not** in the lock file (grep count 0) — this module can never import. | src/agent_team/guardrails/input_rails.py:1-7 | High |
| I2 | **Two packages named `box`**: `box>=0.1.5` + `python-box>=7.3.2` (same import name, different distributions). Root cause of B1. | pyproject.toml:13, 23 | Critical |
| I3 | Version pins are floors only (`>=`), so `uv.lock` is the sole reproducibility mechanism. google-adk resolved to **1.3.0**, litellm to **1.80.11**, openai to **2.14.0**, duckduckgo-search to **8.1.1** — all fast-moving APIs; the hand-rolled adapter is at risk of silent breakage on `uv sync --upgrade`. | pyproject.toml:11-26 | Medium |
| I4 | `.python-version` says 3.10; `requires-python = ">=3.10"`. No CI, no ruff/mypy/pytest config. | .python-version; pyproject.toml:10 | Low |
| I5 | README architecture image links to a raw GitHub URL at commit `d811efa` (assets/sys-arch-acs.png exists locally but README references the remote copy). | README.md:12-14 | Low |
| I6 | `.gitignore` line 13 `**/data/**` silently excludes `config/data/` — the exact directory `config/ingest.py` needs (tickets.csv, manuals/) — while nothing documents how to obtain the data. | .gitignore:13; config/ingest.py:255, 269 | High (ops) |

### 4.5 Stubs / scaffolding

| # | Finding | Location | Severity |
|---|---|---|---|
| ST1 | `src/api/routes.py` — only a docstring (`""" Endpoint Routes for API """`). No FastAPI app, no router, no endpoints anywhere. | src/api/routes.py:1 | High |
| ST2 | `src/api/dependencies.py` — only a docstring. | src/api/dependencies.py:1 | High |
| ST3 | `src/core/__init__.py` — empty; commit `b9c5fd2` "restructuring for API design" deleted `src/api/app.py` and `src/api/orchestrator.py` (40 lines) without replacing them. | src/core/__init__.py | Medium |
| ST4 | `Dockerfile` — 0 bytes, committed empty since `d811efa`. | Dockerfile | Medium |
| ST5 | `main.py` is a blocking `input()` REPL — the only interface. No HTTP, no task queue, no events/SSE exposure. | main.py:76-95 | High (vs README scope) |
| ST6 | `config/ingest.py` is the only "service" code; `src/services/` is empty. | src/services/__init__.py | Low |

### 4.6 Other observations

- `main.py` paths are cwd-dependent (`open("config/agents.yml")`, main.py:21) — must run from repo root; breaks when launched from elsewhere or inside a container.
- Logging: `src/logger.py:12-13` names the file by the date *at startup* and also rotates at midnight — rotation produces date-named files while the handler keeps writing to the original; harmless but confusing. `logs/` is gitignored.
- `src/exception.py` relies on `sys.exc_info()` — stale traceback if `CustomException` is raised outside an `except` block (it always is, in the current call sites).
- Retrieval correctness: `rag_agent` uses `embedding <#> query` (negative inner product) with HNSW `vector_ip_ops` — consistent choice **[verified by reading pgvector.sql]**; note `<#>` is *not* cosine similarity, so results are magnitude-sensitive — a semantic-relevance footgun for a KB retriever.
- `tests/simulate_conversations.py` writes `results.json` (gitignored) — the only artifact-producing test; requires live OpenAI keys + DB, so not repeatable offline.
- Session/state: `InMemorySessionService` means no persistence; fine for a demo, insufficient for the planned API.

---

## 5. Dependency & Tooling Review

### 5.1 Direct dependencies (pyproject.toml:11-26) vs actual usage

| Dep | Pinned (floor) | Resolved in uv.lock | Used? | Notes |
|---|---|---|---|---|
| `box` | >=0.1.5 | **0.1.5** | ❌ broken | Unrelated anonymous PyPI package; clobbers python-box → **breaks the app** |
| `python-box` | >=7.3.2 | 7.3.2 | ✅ (intended) | The real `ConfigBox` provider; sufficient alone |
| `google-adk` | >=0.0.1 | 1.3.0 | ✅ | All used API verified working on 1.3.0 |
| `litellm` | >=1.0.0 | 1.80.11 | ✅ | Backs every agent via LiteLLMModel |
| `openai` | >=1.69.0 | 2.14.0 | ✅ | Embeddings (utils.py, ingest.py) |
| `duckduckgo-search` | >=5.0.0 | 8.1.1 | ✅ | `from duckduckgo_search import DDGS` + `.text()` verified on 8.1.1 |
| `sqlalchemy` | >=2.0.40 | 2.0.40 | ✅ | Agents + ORM |
| `asyncpg` | >=0.29.0 | 0.31.0 | ✅ (ingest only) | async engine in ingest.py |
| `psycopg2-binary` | >=2.9.10 | 2.9.10 | ✅ | default driver for `create_engine(postgresql://)` |
| `pgvector` | >=0.4.0 | 0.4.0 | ✅ | Vector column type |
| `pandas` | >=2.2.3 | 2.2.3 | ✅ (ingest only) | CSV → tickets table |
| `pypdf` | >=5.4.0 | 5.4.0 | ✅ (ingest only) | PDF extraction |
| `pyaml` | >=25.1.0 | 25.1.0 | ❌ | unused (PyYAML provides `yaml`) |
| `redis` | >=6.0.0 | 7.1.0 | ❌ | unused (planned for task queue) |

### 5.2 Notable transitive deps (already in lock, not pinned directly)

- **fastapi==0.128.0, uvicorn==0.34.0, sse-starlette==2.2.1, websockets==15.0.1** — pulled in by google-adk. The entire README roadmap (FastAPI, SSE, WebSockets) is *accidentally* installed; only code is missing.
- `google-cloud-aiplatform`, `vertexai`, `grpcio`, `pysqlite3-binary` (linux marker), etc. — heavy AI-platform tree (508 packages total in lock) largely unnecessary for this usage; worth trimming with `google-adk[agents]`-style optional extras if available.

### 5.3 Missing / stale

- **`openai-agents`** — NOT in lock, but `guardrails/input_rails.py:1-7` imports `agents`. (Should be deleted, not added — the project moved to ADK.)
- **pytest** — absent; the 5 test scripts are plain executables, none discoverable by pytest (no `test_*` discovery failure, but no asserts; `test_adk_agents.py` has commented-out tests).
- **No .env / .env.example** — `OPENAI_MODEL`, `OPENAI_EMBEDDINGS`, `DB_STRING`, `LOG_LEVEL` read via `os.getenv` with defaults only for `OPENAI_MODEL` (support_agent.py:21); everything else required.
- **No lint/format/type tooling, no CI, no ruff config** — nothing enforces quality; the monkeypatch at support_agent.py:41-49 is the kind of thing linting would flag.
- **Packaging broken** — `[tool.setuptools] packages = ["src"]` (pyproject.toml:28-29) ships only the `src` package itself; **verified by building a wheel** in a scratch dir: `src/agent_team/` etc. are excluded. `uv.lock` uses `source = { editable = "." }`, so dev-from-root works — any wheel install would be broken.

### 5.4 Tooling

- uv 0.9.21 available; lock is uv-format. No `.venv` currently in the repo; system Python 3.11.6 has none of the project deps installed — a fresh `uv sync` is required before anything runs (and will currently install the broken `box==0.1.5`).

---

## 6. Blocking Gaps & Recommended Fixes

### P0 — project cannot start; fix first

1. **Remove `box>=0.1.5` from pyproject.toml:13** (keep `python-box>=7.3.2`), run `uv lock` + `uv sync`. [verified: python-box alone provides `from box import ConfigBox`]. This unblocks every import path (B1/S1 source).
2. **Provide environment config**: add `.env.example` with `OPENAI_MODEL`, `OPENAI_EMBEDDINGS`, `DB_STRING`, `POSTGRES_USER/PASSWORD/DB`, `LOG_LEVEL`; document `cp .env.example .env`. docker-compose.yml:6-9 currently boots Postgres with blank credentials (S6).
3. **Restore/obtain the dataset**: `config/data/tickets.csv` and `config/data/manuals/*.pdf` are referenced (ingest.py:255, 269) but never committed (`**/data/**` in .gitignore:13). Either add the data (with a deliberate exception to the gitignore), or add a provisioning script + documented source, and fail loudly in `ingest.py` when the dirs are missing.

### P1 — required to complete the README scope

4. **Implement the FastAPI layer** the repo already scaffolds: `src/api/routes.py` (endpoints: POST /chat, GET /events), `src/api/dependencies.py` (session/user DI), an `app.py`/`main.py` entrypoint. Pin fastapi/uvicorn/sse-starlette/websockets as **direct** deps (they're already in the lock transitively). Use google-adk's native `StreamingMode.SSE` for real-time logging/streaming — but first fix B7 (the LiteLLM adapter's stream path drops tool calls and never sets `turn_complete`).
5. **WebSockets + task queue**: add a WebSocket endpoint for real-time comms (README item); wire the already-present `redis` dep into a queue (arq/rq or a simple asyncio worker) for async task processing — or remove `redis` from deps until then.
6. **Fix or delete guardrails** (D1/I1): the module is broken three ways (wrong import path, removed SDK, never wired). Either rewrite as a google-adk callback (e.g. `before_model_callback`/`before_tool_callback` input validation) and attach to the support agent, or delete the package and the `guardrail_agents` config section.
7. **Harden the SQL agent** (S2): enforce read-only — wrap in a transaction that always rolls back, allowlist `SELECT`-only statements, or connect via a dedicated read-only Postgres role; never expose `DROP/TRUNCATE/UPDATE/DELETE` from the tickets table (PII).
8. **Fix RAG SQL injection** (S1): bind `top_k` as a parameter (`LIMIT :top_k`) and cast the embedding via `text()`-safe parameter passing instead of f-string interpolation; also fix B2 (empty-result branch) and B3 (None return) while in there.
9. **Make the test suite real**: convert the 5 scripts to pytest (fixtures + mocks for litellm/DB/DDGS), add asserts, and add a CI workflow (uv sync + pytest). Add ruff + mypy config to pyproject.
10. **Fix packaging + containerization**: `[tool.setuptools] packages = ["src"]` → `find` (or `src`-layout auto-discovery) so wheels include subpackages; write a real Dockerfile (currently 0 bytes) — multi-stage: `ghcr.io/astral-sh/uv` builder + slim runtime, `CMD ["uv", "run", "python", "main.py"]` (or the future FastAPI server).

### P2 — hygiene / robustness

11. Make ingest idempotent (B5): drop manual IDs, let `BIGSERIAL` assign, upsert by `doc_name`; resolve `from schemas import` (B10) via relative import or package-ification.
12. `config/schemas.py:21,37` — `default=datetime.now` (no parens) (B4).
13. Delete dead code: `pyaml`, unused `read_config`/ConfigBox import pattern, `retrieval.sql`, unused imports (`sys` main.py:3, `tool_context` support_agent.py:6, inner `import json`), `src/core`/`src/services` empties or give them real content.
14. Make paths root-relative (`Path(__file__).resolve().parent`) in main.py and ingest.py so the app runs from any cwd (4.6).
15. Add `.env.example` values validation at startup; surface a clear error if `DB_STRING`/`OPENAI_EMBEDDINGS` are missing.
16. Consider constraining fast-moving deps (`google-adk`, `litellm`, `duckduckgo-search`) with upper bounds or dependabot-style refresh cadence, given the hand-rolled adapter (I3).
17. Evaluate trimming the google-adk AI-platform transitive tree (5.2) if package size matters.
18. Keep `logs/` behavior sane: fix the date-vs-rotation naming in `src/logger.py`.

---

### Verification appendix (how claims were checked)

- Repo fully read (all files listed in scope); `git log` 14 commits reviewed with `--stat`.
- Temp venv `/tmp/acs-verify` with lock-resolved versions: **box==0.1.5 + python-box==7.3.2 → `ImportError: cannot import name 'ConfigBox' from 'box'`**; python-box alone → OK. `import main` / `import src.agent_team.*` fail with that ImportError.
- google-adk 1.3.0: `Agent(...)` with the project's exact kwargs OK; `FunctionTool(transfer_to_agent)._get_declaration()` OK (description non-None); `LlmRequest`/`LlmResponse` fields match usage; `Runner.run(user_id, session_id, new_message)` signature matches; `InMemorySessionService.create_session` is async; `StreamingMode` defaults to `NONE`.
- duckduckgo-search 8.1.1: `from duckduckgo_search import DDGS`, `.text()` present.
- Wheel build test: `packages = ["src"]` ships only `src/__init__.py`.
- `docker compose config`: POSTGRES_* all blank without `.env`.
