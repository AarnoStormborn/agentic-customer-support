# AGENTS.md — Project Rules (auto-loaded by pi every session)

> Pi loads this file into the system prompt automatically. It is the single source of truth
> for how to work in this repo. **Do not ask the owner to repeat anything stated here.**

## Project identity

- **agentic-customer-support** — a customer-support system with multi-agent retrieval:
  SQL (tickets) + vector (manuals/knowledge base) + web search.
- v2 is a **full rebuild** on the **pi agents SDK** (`@earendil-works/pi-coding-agent`, Node/TypeScript).
  The old Python implementation lives in git history (`main.py`, `src/`, `config/`, `tests/`) —
  **history must be preserved; never rewrite or force-push it away.**
- The owner's explicit goal: this project is a **learning vehicle** for new techniques and stacks.
  Every non-trivial concept the owner may not know must be explained in `lessons.md`.

## Locked stack decisions (do not revisit without owner)

| Area | Decision |
|---|---|
| Language/runtime | **TypeScript / Node.js** (pi SDK is Node-only) |
| Agent runtime | **pi agents SDK** `@earendil-works/pi-coding-agent` — in-process `createAgentSession()` (option A) |
| SDK isolation | All pi SDK imports live in `src/runtime/` only; pin exact version; upgrade touches one module |
| Data stores | Postgres + **pgvector** (hybrid search), **Redis** (BullMQ) |
| Streaming | **SSE** for token/tool events, **WebSocket** for real-time comms |
| Task queue | **BullMQ + Redis** |
| Guardrails | pi extension interception hooks: `input`, `context`, `tool_call`, `tool_result` |
| Security | SQL is **read-only allowlist** (SELECT-only, parameterized); no secrets in code (`.env` only); no bash/read/write tools in support sessions |

## Open decisions (currently being designed — see `docs/plan.md` Phase 3)

- Backend web framework: Fastify vs Express vs NestJS vs Hono (**evaluate, recommend one**)
- UI framework: React (owner knows it) vs Next.js vs Vite+React SPA (**evaluate, recommend one**)
- Retrieval specifics: hybrid fusion (RRF), reranker choice, embedding model, chunking strategy
- Dataset bundle: from `docs/data-research.md` (owner defers provisioning)

## Architecture rules (design constraints — honor them)

1. Support sessions get **no filesystem tools** (`noTools: "builtin"` or explicit `tools: [...]`).
2. Sub-agents (rag / sql / web) are implemented as a `route_to_agent` custom tool that spawns
   child `AgentSession`s (pi has no native handoffs). Child sessions: specialist system prompt,
   one tool each, disposed after use.
3. Guardrails fire at the interception layer — never trust model/tool output blindly.
4. All event→client streaming goes through one bridge module (`src/streaming/`).
5. Every phase produces/updates a doc in `docs/`. Design docs live in `docs/design/`.

## Learning rule (MANDATORY)

- Maintain **`lessons.md`** (repo root) — a beginner-friendly knowledge log for the owner.
- Whenever the owner might not know a concept (framework, TS/Node idiom, SDK behavior, DB
  technique), **add an entry to `lessons.md`** explaining it in plain terms with a tiny example.
- Keep entries small and factual; reference source docs where useful. Do not pad.
- If a design doc makes a framework/library choice, the "why" must also land in `lessons.md`.

## Workflow conventions

- Work is planned in phases (see `docs/plan.md`). Confirm phase mapping before big changes.
- **Scaffold/config changes:** use `npm`, TypeScript ESM (`"type": "module"`), `tsx` for running.
- **Env:** copy `.env.example` → `.env`; never commit `.env` (gitignored).
- **Running:** `npm run dev` (server), `npm run spike` (SDK check), `npm run typecheck`.
- **Testing:** test suite comes after migration (owner's call); prefer `vitest` when we build it.
- **Docs first, code second.** Design decisions land in `docs/design/` before implementation.

## Key docs map

| File | Contents |
|---|---|
| `docs/plan.md` | Phased plan (1 setup, 2 learnings, 3 design, 4 parallel agents, 5+ impl) |
| `docs/project-analysis.md` | Code review of the old Python project (bugs, gaps) |
| `docs/tech-stack-research.md` | Stack research incl. pi SDK deep-dive |
| `docs/data-research.md` | Verified datasets (CFPB, suraj520 tickets, manuals) |
| `docs/design/*.md` | Design docs (backend, agent loop, retrieval, data mgmt, UI) |
| `lessons.md` | Owner's learning log (new stack, frameworks, SDK) |

## House rules

- Be concise; show file paths clearly.
- Ask the owner only when a decision is genuinely open (see "Open decisions" above) —
  never re-ask anything already recorded in this file, `docs/plan.md`, or `lessons.md`.
- Keep the Python history untouched; work in the new TS structure.
