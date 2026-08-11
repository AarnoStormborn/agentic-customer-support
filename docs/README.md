# Agentic Customer Support — Research & Analysis Docs

This folder holds the findings of the research/analysis agents working on the project.

| File | Agent | Status |
|------|-------|--------|
| [project-analysis.md](project-analysis.md) | `project-analysis` — code review & current status | in progress |
| [tech-stack-research.md](tech-stack-research.md) | `stack-research` — tech stack + pi SDK research | in progress |
| [data-research.md](data-research.md) | `data-research` — dataset research | in progress |

## Mission

1. **Update the stack** — modernize dependencies/frameworks (evaluate pi agents SDK)
2. **Complete the project** to its planned scope (FastAPI API, real-time comms, SSE logging, task queue, MCP)
3. **Extend with a full-scale retrieval system** — SQL retrieval + vector retrieval + web search over a tech-support dataset (tickets + manuals)

## Current Stack (as found)

- Google ADK (google-adk) + LiteLLM model adapter
- Postgres + pgvector, SQLAlchemy, asyncpg
- OpenAI embeddings (text-embedding-3-small)
- duckduckgo-search
- FastAPI (planned, not implemented)
