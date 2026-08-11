# Tech Stack Research

> Deep research (web + local docs) into a modern 2025/2026 stack for the agentic customer-support / retrieval system.
> Current stack: Google ADK + LiteLLM, Postgres + pgvector, OpenAI embeddings, duckduckgo-search, FastAPI planned.
> **No project code was modified during this research.**

---

## 1. Executive Summary

1. **Agent framework → OpenAI Agents SDK (Python)**, not ADK and not pi SDK. The Agents SDK is the most natural fit for the three-agent supervisor (RAG / SQL / Web) design: first-class handoffs, guardrails, sessions, built-in tracing, native MCP support, and async streaming — all in-process with FastAPI. LangGraph is the alternative if durable, resumable workflows become a hard requirement ([openai-agents-python](https://openai.github.io/openai-agents-python/agents/), [MCP support](https://github.com/openai/openai-agents-python/blob/main/docs/mcp.md), [Langfuse framework comparison](https://langfuse.com/blog/2025-03-19-ai-agent-comparison)).
2. **pi SDK verdict**: technically capable of running the support system as **custom tools + extensions** (`pi.registerTool()`, event hooks), but it is a **Node.js/TypeScript, local-filesystem coding-agent harness** — no native multi-agent supervisor, no built-in MCP (needs `pi-mcp-adapter`), pre-1.0 API (v0.84.x), and web-hosted deployment requires reimplementing session storage and sandboxing. Keep pi as the **developer-side harness / internal ops agent (RPC mode)**, not as the product runtime ([aibuilderclub 2026 guide](https://www.aibuilderclub.com/blog/pi-agent-extensions-guide), [pi SDK docs](https://pi.dev/docs/latest/sdk)).
3. **Retrieval → keep Postgres + pgvector** (upgrade ≥0.8/0.9 for HNSW + iterative scans), add **hybrid search (Postgres FTS or ParadeDB BM25 + vector, fused with RRF)**, a **reranker** (Cohere Rerank hosted, or BGE-reranker-v2-m3 self-hosted), and consider **bge-m3 / snowflake-arctic-embed2** embeddings instead of OpenAI-only for multilingual + sparse support. Cross-encoder reranking is the single highest-leverage retrieval upgrade ([denser.ai hybrid search](https://denser.ai/blog/hybrid-search-for-rag/), [reranking study](https://app.ailog.fr/en/blog/news/reranking-cross-encoders-study)).
4. **Orchestration → FastAPI + SSE (`sse-starlette`)** for agent event streaming (token deltas, tool activity, sources) with REST for control; **ARQ + Redis** for durable background jobs (ingestion, evals). No Celery — its async story is bolted-on. Fix the current **blocking sync SQLAlchemy engine inside async tools** by moving to SQLAlchemy 2.0 async + asyncpg ([SSE vs WS](https://dev.to/raxxostudios/server-sent-events-beat-websockets-for-80-of-my-ai-streaming-uis-5-patterns-49ac), [ARQ vs BackgroundTasks](https://davidmuraya.com/blog/fastapi-background-tasks-arq-vs-built-in/), [task queue guide](https://dev.to/datanestdigital/background-jobs-in-python-celery-vs-rq-vs-dramatiq-vs-arq-2026-decision-guide-37m6)).
5. **Web search → Tavily as primary** (agent-native, 1,000 free credits/mo, pay-as-you-go), keep DuckDuckGo as free fallback, add Exa only if deep-research mode is desired; **observability → Langfuse** (self-hostable, ClickHouse-backed, native OpenAI-Agents tracing, RAGAS integration) + **RAGAS** for offline retrieval/faithfulness evals ([Tavily pricing](https://www.tavily.com/pricing), [Langfuse × OpenAI Agents](https://langfuse.com/integrations/frameworks/openai-agents), [RAGAS × Langfuse](https://langfuse.com/guides/cookbook/evaluation_of_rag_with_ragas)).

---

## 2. Pi Agents SDK Deep-Dive

Primary sources: local install `@earendil-works/pi-coding-agent` **v0.84.1** (`docs/sdk.md`, `docs/extensions.md`, `docs/rpc.md`), the [official SDK docs](https://pi.dev/docs/latest/sdk) / [GitHub docs](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/sdk.md), and an independent 2026 analysis ([aibuilderclub — "Pi Agent Extensions"](https://www.aibuilderclub.com/blog/pi-agent-extensions-guide)) that covers building web-hosted products on pi (it pegs pi at v0.80.7 / ~71.5k stars, July 2026; local install is newer).

### 2.1 What the SDK actually is

Pi is **not** a multi-agent framework like ADK/OpenAI-Agents/LangGraph. It is a **terminal coding-agent harness** (read/bash/edit/write + a ~1,000-token system prompt) with a layered TypeScript SDK:

| Layer | Package | Purpose |
|---|---|---|
| Unified LLM API | `@earendil-works/pi-ai` | Providers, models, OAuth, credential store (15+ providers) |
| Agent loop | `@earendil-works/pi-agent-core` | Tool-calling loop, message state |
| Harness/SDK | `@earendil-works/pi-coding-agent` | Sessions, tools, extensions, compaction, SDK entry points |
| UI | `@earendil-works/pi-tui` | Terminal rendering |
| Orchestrator | `@earendil-works/pi-orchestrator` | **Experimental** scheduled jobs / delegation |

The SDK exposes `createAgentSession()`, `AgentSession` (prompt/steer/followUp, `subscribe()` event stream), `SessionManager` (in-memory or JSONL session files with **tree branching**), `ModelRuntime` (model + auth resolution), `DefaultResourceLoader` (extensions/skills/prompts/context files), and run modes: `InteractiveMode`, `runPrintMode`, and **`runRpcMode`** (JSON-RPC over stdio, language-agnostic).

### 2.2 Capabilities — can the support system be built as pi extensions/custom tools?

**Yes, technically.** The extension API is genuinely powerful and in-process:

- **Custom tools** the LLM can call, with TypeBox schemas, streaming `onUpdate` progress, and optional `terminate` semantics:

```typescript
// From docs/sdk.md — custom tool via the SDK
import { Type } from "typebox";
import { createAgentSession, defineTool } from "@earendil-works/pi-coding-agent";

const myTool = defineTool({
  name: "my_tool",
  label: "My Tool",
  description: "Does something useful",
  parameters: Type.Object({
    input: Type.String({ description: "Input value" }),
  }),
  execute: async (_toolCallId, params) => ({
    content: [{ type: "text", text: `Result: ${params.input}` }],
    details: {},
  }),
});

const { session } = await createAgentSession({ customTools: [myTool] });
```

```typescript
// From docs/extensions.md — extension style (pi.registerTool)
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "search_kb",
    label: "Search Knowledge Base",
    description: "Vector + BM25 hybrid search over technical manuals",
    parameters: Type.Object({ query: Type.String(), top_k: Type.Optional(Type.Number()) }),
    async execute(toolCallId, params, signal, onUpdate, ctx) {
      onUpdate?.({ content: [{ type: "text", text: "Searching…" }] });
      return { content: [{ type: "text", text: "results…" }], details: { sources: [] } };
    },
  });
}
```

- **Event interception** (`tool_call` can block/mutate input, `tool_result` can modify output, `context` can rewrite messages before each LLM call, `before_agent_start` can inject context/system-prompt changes, `input` can route/handle user text) — this is where permission gates, citation injection, PII scrubbing, and streaming telemetry would live.
- **Streaming events** (`message_update` token deltas, `tool_execution_start/update/end`, `turn_start/end`, `agent_start/end`) — subscribable in the SDK:

```typescript
// From docs/sdk.md — event streaming
session.subscribe((event) => {
  if (event.type === "message_update" && event.assistantMessageEvent.type === "text_delta") {
    process.stdout.write(event.assistantMessageEvent.delta);
  }
});
await session.prompt("What files are in the current directory?");
```

- **Sessions**: `SessionManager.inMemory()` / `SessionManager.create(cwd)` / `open(file)`, tree-based branching (`branch`, `fork`, labels), compaction with extension hooks (`session_before_compact`), and queueing semantics (`steer` mid-stream, `followUp` after completion).
- **Skills & prompts**: SKILL.md instruction files and slash-command prompt templates loaded per-session via `DefaultResourceLoader`.
- **MCP**: **not built-in by design.** The `pi-mcp-adapter` package (v2.22.0 in local npm cache) adds MCP-server consumption with token-budget protection (single ~200-token proxy tool, lazy server startup); it reads `.mcp.json` / host configs and supports MCP OAuth. There is **no first-class "pi as an MCP server" export** — pi consumes MCP servers; it does not serve as one ([aibuilderclub FAQ](https://www.aibuilderclub.com/blog/pi-agent-extensions-guide)).
- **Sub-agents / multi-agent**: no native supervisor or handoff primitives. Parallelism exists at the **tool** level (parallel tool execution in one turn), and interleaving via `steer`/`followUp`. `pi-subagents` (community package) and the experimental `pi-orchestrator` add delegation; the founder's documented stance is "spawn Pi instances via tmux, build your own with extensions, or install a package" ([aibuilderclub](https://www.aibuilderclub.com/blog/pi-agent-extensions-guide)).
- **Non-TUI integration**: `pi --mode rpc` exposes a JSON-RPC protocol over stdio (`prompt`, `steer`, events as JSONL) — the sanctioned path for non-Node hosts ([docs/rpc.md](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/rpc.md)).

### 2.3 Fit for this project

**The honest assessment — it is the wrong primary runtime for a Python/FastAPI support backend, and a good secondary/dev tool:**

| Question | Answer |
|---|---|
| Can RAG + SQL + web-search live as pi custom tools? | Yes — `defineTool` / `pi.registerTool` map 1:1 onto the current `retriever_tool` / `run_sql_queries` / web-search tools. |
| Can the agent stream to a web client? | Yes, via SDK `subscribe()` events bridged to SSE, but you must build that bridge; `runRpcMode`/RPC gives a process boundary. |
| Multi-agent supervisor (support → rag/sql/web)? | **No native equivalent of ADK `transfer_to_agent` / OpenAI handoffs.** Would be hand-rolled via tools + message queueing, or community packages (young, unvetted). |
| MCP (README explicitly wants "OpenAI MCP")? | Only via `pi-mcp-adapter` extension; and pi **consumes** MCP rather than exposing a product-grade MCP server. |
| Python ecosystem fit? | **Node.js only.** Embedding pi in-process means running a Node service alongside FastAPI, or a subprocess over JSON-RPC. Python frameworks give in-process async tool calls, shared DB pools, and Langfuse instrumentation for free. |
| Production posture | Pre-1.0 (0.8x), breaking changes across versions, extension ecosystem young; extensions run with **full system permissions** (a real supply-chain consideration). Web-hosted deployment requires replacing session persistence (use `SessionManager.inMemory()` + your own DB) and wrapping the default `bash/read/write` tools per-user sandbox — real work, explicitly documented ([aibuilderclub hosted-products section](https://www.aibuilderclub.com/blog/pi-agent-extensions-guide)). |
| Where it genuinely shines | As the **developer harness** (this very research session is run inside it), and as an **internal "ops agent"** — e.g., a `pi --mode rpc` subprocess that a support engineer or the system itself invokes for deep-research / repo work, with custom extensions (permission gates, compaction control, event-driven tooling). This is the pattern proven by OpenClaw, which embeds `createAgentSession()` directly for a gateway product ([aibuilderclub](https://www.aibuilderclub.com/blog/pi-agent-extensions-guide)). |

**Verdict:** Do **not** rebuild the customer-support product on the pi SDK. Keep pi as the team's daily driver + optional RPC-based research agent. Build the product runtime on a Python agent framework (Section 3).

---

## 3. Framework Comparison

Sources: [Langfuse framework comparison (2025)](https://langfuse.com/blog/2025-03-19-ai-agent-comparison), [ZenML ADK vs LangGraph](https://www.zenml.io/blog/google-adk-vs-langgraph), [durable-state comparison](https://dev.to/zira125/durable-state-for-coding-agents-langgraph-vs-google-adk-vs-openai-agents-sdk-49c5), [2026 agent frameworks roundup](https://www.firecrawl.dev/blog/best-open-source-agent-frameworks), [ADK 2.0 announcement](https://developers.googleblog.com/en/why-we-built-adk-20/), [breakingcube ADK vs Agents SDK](https://tech.breakingcube.com/2026/03/15/google-adk-vs-openai-agents-sdk-comparison/), [openai-agents-python MCP docs](https://github.com/openai/openai-agents-python/blob/main/docs/mcp.md), [pi docs](https://pi.dev/docs/latest/sdk).

| Dimension | **pi SDK** (TS) | **Google ADK** (Python/Go/Java) | **OpenAI Agents SDK** (Python/TS) | **LangGraph** (Python/TS) |
|---|---|---|---|---|
| Multi-agent | ✗ None native (community `pi-subagents`, experimental orchestrator) | ✓ `transfer_to_agent`, sub-agents, workflow runtime (graph) in 2.0 | ✓ Handoffs + triage agent; minimal primitives | ✓ Graph workflows; arbitrary topologies |
| Tools | Custom tools via extension/SDK (`defineTool`) | `FunctionTool` / MCP | Function tools / MCP | Function tools / MCP |
| Streaming | Rich event stream (SDK/RPC) | Async generator events | Streaming events (`stream_events`/`run`), async | `stream_mode` (messages/updates/custom) |
| MCP | Via `pi-mcp-adapter` (client only) | First-class MCP client | **First-class MCP client** | First-class MCP client |
| Sessions/persistence | JSONL tree w/ branching + compaction; in-memory option | SessionService (in-memory/file/Vertex); limited | Sessions (in-memory; BYO persistence); no built-in DB | **Durable checkpointing**, time-travel, human-in-the-loop |
| Guardrails | Extension hooks (block/mutate) | Input/output rails (built-in) | Guardrails (tripwires, built-in) | Middleware/nodes |
| Observability | Extension events; no vendor tracing | Vertex + agent-engines hooks; LiteLLM logging | **Built-in tracing** (exportable), Langfuse/LangSmith adapters | LangSmith first-party; Langfuse adapter |
| Maturity | Pre-1.0 (0.8x), young ecosystem | 2.x, Google-backed, production on Vertex | Production, most-used for OpenAI-model shops (successor to Swarm) | **Most enterprise-adopted** (34.5M monthly downloads per [Firecrawl](https://www.firecrawl.dev/blog/best-open-source-agent-frameworks)) |
| Model-agnostic | ✓ 15+ providers, BYO keys | ✓ + LiteLLM adapter (current project already uses this) | ✓ OpenAI-compatible; other models via Chat Completions work but handoff/tracing optimized for OpenAI | ✓ Fully model-agnostic |
| Best for | Coding harness, hackable CLI agent | GCP/Vertex-centric enterprise, multi-language | Lightweight supervisor+handoff products on OpenAI models | Complex stateful workflows, durable execution |

**Recommendation for this project:** **OpenAI Agents SDK (Python)** — the architecture (a supervisor `support_agent` handing off to `rag_agent` / `sql_agent` / `web_agent`) is literally the SDK's canonical handoff/triage pattern, it is async-native (matches FastAPI), has built-in tracing that Langfuse instruments natively, and supports MCP for the web-search/tooling layer. **LangGraph** is the fallback if you need durable, resumable runs (checkpointing to Postgres, retries after crash) or complex branching beyond a star topology. **ADK** stays viable (LiteLLM adapter already exists) but its strengths (Vertex AI integration, multi-language) don't buy anything here, and its docs/lock-in lean Google Cloud. The current repo's README says "designed using OpenAI Agents SDK" while `pyproject.toml`/code use `google-adk` — an inconsistency worth resolving in the migration.

---

## 4. Retrieval Stack Recommendations

Sources: [digitalapplied vector DB matrix (2026)](https://www.digitalapplied.com/blog/vector-databases-for-ai-agents-pinecone-qdrant-2026), [Firecrawl vector DB guide](https://www.firecrawl.dev/blog/best-vector-databases), [pgvector 0.9.0 / pgvectorscale](https://thebuild.com/blog/on-pgvectorscale-and-hybrid-search-without-an-elasticsearch-sidecar), [ParadeDB hybrid-search manual](https://www.paradedb.com/blog/hybrid-search-in-postgresql-the-missing-manual), [Jonathan Katz hybrid pgvector](https://jkatz.github.io/post/postgres/hybrid-search-postgres-pgvector), [Qdrant vs pgvector](https://open-techstack.com/blog/pgvector-vs-qdrant-2026/), [kalviumlabs production retro](https://www.kalviumlabs.ai/blog/vector-databases-compared-pgvector-pinecone-qdrant-weaviate/), [denser.ai hybrid benchmark](https://denser.ai/blog/hybrid-search-for-rag/), [RRF rationale](https://blog.gopenai.com/hybrid-search-in-rag-dense-sparse-bm25-splade-reciprocal-rank-fusion-and-when-to-use-which-fafe4fd6156e), [rerank study](https://app.ailog.fr/en/blog/news/reranking-cross-encoders-study), [BGE/Cohere/Jina rerank guide](https://localaimaster.com/blog/reranking-cross-encoders-guide), [hosted vs self-hosted rerank](https://bigdataboutique.com/blog/rag-reranking-improving-retrieval-quality-with-cross-encoders), [embedding model guide](https://innovativeais.com/blog/best-embedding-models-for-rag-in-2026), [late chunking paper](https://arxiv.org/abs/2409.04701), [chunking strategies](https://www.firecrawl.dev/blog/best-chunking-strategies-rag).

### 4.1 Vector database — stay on pgvector

| DB | Type | Native hybrid (BM25+dense) | p99 @10M vectors | Scale ceiling | Ops | Fit here |
|---|---|---|---|---|---|---|
| **pgvector** | Postgres extension | Via FTS/tsvector + RRF (manual composition) | ~25–40 ms | 10–100M (pgvectorscale for more) | Zero new infra; ACID, backups, joins with tickets data | **✔ Default — already in stack** |
| Qdrant | Dedicated (Rust) | Yes (2024) + sparse vectors | ~12 ms | 100M+ | Extra service (self-host or cloud) | Only if latency/filter perf becomes critical |
| Weaviate | Dedicated | **Best OOTB** (BM25+vector+meta) | ~16 ms | 100M+ | Extra service | Attractive but adds infra |
| Chroma | Embedded | Partial | ~30 ms | Prototype-scale | In-process | Prototyping only |
| LanceDB | Embedded/local | Yes (hybrid) | n/a | Local-first | No server | Edge/datasets, not this |
| Milvus | Distributed | Via collections | ~18 ms | 1B+ | Heavy ops | Overkill here |

At this project's scale (support manuals: 10k–1M chunks), pgvector is the consensus default: "pgvector is what most teams should actually use, because most workloads are smaller than people think" ([digitalapplied](https://www.digitalapplied.com/blog/vector-databases-for-ai-agents-pinecone-qdrant-2026)); pgvector handled 2M vectors without special tuning in a production retro ([kalviumlabs](https://www.kalviumlabs.ai/blog/vector-databases-compared-pgvector-pinecone-qdrant-weaviate/)). Upgrade to **pgvector ≥ 0.8** (iterative index scans, HNSW improvements — [AWS blog](https://aws.amazon.com/blogs/database/supercharging-vector-search-performance-and-relevance-with-pgvector-0-8-0-on-amazon-aurora-postgresql)) / **0.9.0** (parallel index builds — [thebuild.com](https://thebuild.com/blog/on-pgvectorscale-and-hybrid-search-without-an-elasticsearch-sidecar)); optionally add **pgvectorscale** (StreamingDiskANN) if corpus grows.

### 4.2 Hybrid search (the biggest retrieval win)

- Dense-only retrieval loses lexical precision (product codes, ticket IDs, model numbers — exactly this domain). Hybrid BM25+dense consistently beats either alone: WANDS benchmark 0.7497 NDCG vs 0.6983 (BM25) / 0.6953 (vector) ([denser.ai](https://denser.ai/blog/hybrid-search-for-rag/)).
- **Implement in Postgres, no new services:** `tsvector` column + `websearch_to_tsquery` for BM25-ish FTS, vector index for dense, fuse with **Reciprocal Rank Fusion (RRF)** in SQL ([Jonathan Katz](https://jkatz.github.io/post/postgres/hybrid-search-postgres-pgvector), [ParadeDB](https://www.paradedb.com/blog/hybrid-search-in-postgresql-the-missing-manual)). RRF (rank-based) beats score-normalization fusion because min-max score normalization is fragile to outliers ([gopenai analysis](https://blog.gopenai.com/hybrid-search-in-rag-dense-sparse-bm25-splade-reciprocal-rank-fusion-and-when-to-use-which-fafe4fd6156e)).
- **Upgrade option:** [ParadeDB `pg_search`](https://www.paradedb.com/blog/hybrid-search-in-postgresql-the-missing-manual) extension for true BM25 scoring inside Postgres (used by pgvectorscale's hybrid story) — worth enabling if FTS quality disappoints.
- The current `retrieval.sql`/`pgvector.sql` in `config/sql/` should be reworked for this (embedding-only `ORDER BY embedding <#> …` today).

### 4.3 Reranking

- A cross-encoder rerank over top-20–50 candidates is the **highest-leverage single upgrade**: ~40% RAG accuracy improvement reported ([ailog study](https://app.ailog.fr/en/blog/news/reranking-cross-encoders-study)); BEIR NDCG@10: BM25 41.7 → bi-encoder 51.0 → **+cross-encoder ~60** ([localaimaster](https://localaimaster.com/blog/reranking-cross-encoders-guide)).
- **Options:** Cohere Rerank (hosted, strong multilingual quality, ~32K context on v4 per ailog) vs **BGE-reranker-v2-m3** (open weights, multilingual, self-host via `sentence-transformers`/TEI) vs Jina Reranker v2 ([bigdataboutique](https://bigdataboutique.com/blog/rag-reranking-improving-retrieval-quality-with-cross-encoders)). Start **Cohere Rerank** for zero-ops; swap to self-hosted BGE if data-residency/cost dictates.

### 4.4 Embeddings

- Current: OpenAI `text-embedding-3-*` (hosted, simple, 3072-dim max) — keep as the pragmatic baseline.
- **bge-m3** (BAAI): dense + sparse + multi-vector (ColBERT) from one model, 1024-dim, strong multilingual — pairs naturally with hybrid search; **snowflake-arctic-embed2** (335M "large" variant was the top MTEB retrieval model under 500M params pre-2.0 — [innovativeais](https://innovativeais.com/blog/best-embedding-models-for-rag-in-2026)) is efficient and multilingual.
- Recommendation: keep OpenAI embeddings at first (zero migration), **re-embed with bge-m3 if multilingual (DACH etc.) or lexical-recall needs grow**; store a `model_name`/dimension column per chunk so you can re-embed incrementally.

### 4.5 Chunking

- Technical manuals (this domain): **structural/recursive chunking by headings/sections** with 10–20% overlap is the robust default ([Firecrawl guide](https://www.firecrawl.dev/blog/best-chunking-strategies-rag)); "recursive semantic" chunking papers support hierarchy-aware splitting ([ICNLSP 2025](https://aclanthology.org/2025.icnlsp-1.15.pdf)).
- Advanced: **late chunking** (embed long context, then slice; Jina) improves chunk-level recall for long documents ([arXiv:2409.04701](https://arxiv.org/abs/2409.04701)) — evaluate later.
- Store chunk metadata (doc id, section path, page) — needed for source citations in support answers and for RAGAS context-precision evals.

---

## 5. Orchestration & Real-time

Sources: [SSE beats WS for 80% of AI UIs](https://dev.to/raxxostudios/server-sent-events-beat-websockets-for-80-of-my-ai-streaming-uis-5-patterns-49ac), [sse-starlette](https://github.com/sysid/sse-starlette), [FastAPI ARQ vs BackgroundTasks](https://davidmuraya.com/blog/fastapi-background-tasks-arq-vs-built-in/), [task-queue decision guide](https://dev.to/datanestdigital/background-jobs-in-python-celery-vs-rq-vs-dramatiq-vs-arq-2026-decision-guide-37m6), [Redis background tasks](https://oneuptime.com/blog/post/2026-03-31-redis-fastapi-background-tasks/view), [End-to-end SSE through Nginx](https://dev.to/martin_palopoli/how-i-implemented-end-to-end-sse-streaming-from-llm-to-browser-through-nginx-4bjo).

### 5.1 FastAPI stays

FastAPI is the right async web layer: native `async def` endpooints, `StreamingResponse`/`EventSourceResponse`, WebSocket support, and first-class agent-SDK integration (all three candidate frameworks are async).

### 5.2 SSE vs WebSockets

| Dimension | SSE (EventSourceResponse) | WebSockets |
|---|---|---|
| Direction | Server→client only (control via separate REST calls) | Full-duplex |
| Reconnect | Built-in auto-reconnect (~3s), resume via `Last-Event-ID` | Manual |
| Proxy/Nginx | Simple; buffering must be disabled (`X-Accel-Buffering: no`) | Needs upgrade headers, more proxy config |
| Fit | **Token deltas, tool events, sources, done/error** — the 80% case | Steering/cancel mid-generation, collaboration |
| Libraries | `sse-starlette` (or raw `StreamingResponse`) | `fastapi.websocket` |

Recommendation: **SSE as the primary channel** (one HTTP/2 connection, `EventSource` auto-reconnect, typed event names like `token`, `tool`, `sources`, `done`, `error`), control via REST (`POST /chat/{id}/cancel`). Add WebSockets later **only** if mid-stream steering becomes a product requirement — with the OpenAI Agents SDK's `run()` async generator, bridging to either is trivial. Note the Nginx buffering gotcha in the SSE article above.

### 5.3 Task queue

| Option | Async-native | Durable/retry | Ops | Fit here |
|---|---|---|---|---|
| FastAPI `BackgroundTasks` | ✓ | ✗ (in-process, lost on restart) | zero | Trivial fire-and-forget (logging) |
| **ARQ + Redis** | **✓ (asyncio-first)** | ✓ retries, cron-like jobs | Redis already in `pyproject` | **✔ Ingestion pipeline, re-embedding, nightly eval runs** |
| Celery | ✗ (async bolted on; worker async is awkward) | ✓ rich | heavier | Only if legacy ecosystem forces it |
| Plain asyncio tasks | ✓ | ✗ | zero | Long-lived in-process runs (active agent turns) |

Recommendation: **ARQ + Redis** for durable background work (document ingestion → chunking → embedding → upsert; offline RAGAS eval runs), plain asyncio for the live agent-turn execution path (long-running but supervised by the event loop and cancelable via task handles). The project already depends on `redis>=6.0` — no new infra.

---

## 6. Web Search APIs comparison

Sources: [Stork.ai 2026 comparison](https://www.stork.ai/blog/best-web-search-apis-for-ai-applications-2026), [Rhumb: Exa vs Tavily vs Serper vs Brave](https://rhumb.dev/blog/exa-vs-tavily-vs-serper-vs-brave-search), [MCP directory search comparison](https://mcpdirectory.app/blog/best-web-search-mcp-2026), [DDG rate-limit issues](https://github.com/LearningCircuit/local-deep-research/issues/18), [Tavily pricing](https://www.tavily.com/pricing) + [credits docs](https://docs.tavily.com/documentation/api-credits), [api-pick guide](https://www.apipick.com/blog/best-web-search-apis-for-ai-agents-2026).

| API | Pricing / free tier | Strengths | Weaknesses |
|---|---|---|---|
| **Tavily** (recommended) | 1,000 free credits/mo; pay-as-you-go after | **Purpose-built for agents/RAG**: clean LLM-ready snippets + content, optional `include_answer`, extraction/crawl endpoints; LangChain/Agent-SDK integrations; MCP server | Per-query cost at deep search depth |
| **Exa** | Free tier; usage-based | Semantic/neural search over its own index; research/discovery ("find similar pages"); fast/auto/deep modes; bundled content | Weaker at pure keyword/SERP queries |
| **Brave Search API** | Free tier removed Feb 2026 (small signup credit only) | Independent index, low latency, no query logging (privacy) | No free tier anymore; less "agent-ready" formatting |
| **Serper** | Cheap per query (100 free-ish trial) | Raw Google SERP JSON — cheapest at volume, Google features | You do the cleaning; not agent-native |
| **Perplexity Sonar** | Usage-based | Finished, cited LLM answers — best for "answer with citations" product surface | Slower + pricier per call than retrieval APIs |
| **DuckDuckGo (current)** | Free (unofficial lib) | Free, zero key | **Unofficial/rate-limited**, no SLA, blocks in CI ([rate-limit issue](https://github.com/LearningCircuit/local-deep-research/issues/18)) |

Recommendation: **Tavily primary** (agent-native results + 1k free credits/mo), **DuckDuckGo as no-cost fallback** only, **Exa** if a deep-research mode ships, **Perplexity Sonar** if "synthesized cited answer" becomes a product feature rather than a tool result. All except Serper have MCP servers — the OpenAI Agents SDK consumes MCP directly, so a `tavily-mcp` server is an integration option ([mcpdirectory](https://mcpdirectory.app/blog/best-web-search-mcp-2026)).

---

## 7. Observability & Eval

Sources: [Langfuse × OpenAI Agents SDK](https://langfuse.com/integrations/frameworks/openai-agents), [Langfuse OTEL ingest](https://langfuse.com/integrations/native/opentelemetry), [Langfuse RAGAS cookbook](https://langfuse.com/guides/cookbook/evaluation_of_rag_with_ragas), [Langfuse agent-eval engineering post](https://langfuse.com/resources/engineering/ai-agent-evaluation), [mlflow observability roundup](https://mlflow.org/top-5-agent-observability-tools), [OTEL GenAI conventions](https://dev.to/vola-trebla/opentelemetry-just-standardized-llm-tracing-heres-what-it-actually-looks-like-in-code-2e5f), [RAGAS metrics reference](https://eval-hub.github.io/adapters/ragas/metrics/), [RAGAS tracing docs](https://docs.ragas.io/en/stable/howtos/customizations/metrics/tracing/).

### 7.1 Tracing — Langfuse

- **Langfuse**: open-source, self-hostable, ClickHouse-backed ([mlflow](https://mlflow.org/top-5-agent-observability-tools)); nested trace trees capture the full multi-agent run (handoffs, tool calls, retrieval, embeddings, per-span latency/tokens/cost); LLM-as-judge evals at observation level; **native instrumentation for the OpenAI Agents SDK** ([langfuse integration](https://langfuse.com/integrations/frameworks/openai-agents)) and a LiteLLM integration (already in the current stack); RAGAS integration for scored traces ([cookbook](https://langfuse.com/guides/cookbook/evaluation_of_rag_with_ragas)).
- **OpenTelemetry GenAI semantic conventions**: the emerging cross-vendor standard for LLM span attributes; Langfuse can ingest OTEL traces, so instrumenting with OTEL keeps you portable ([dev.to explainer](https://dev.to/vola-trebla/opentelemetry-just-standardized-llm-tracing-heres-what-it-actually-looks-like-in-code-2e5f)). Add OTEL for infra-level traces (HTTP, DB) if/when the deployment grows; don't build a parallel tracing system.

### 7.2 Evaluation — RAGAS

- **RAGAS** metrics for the RAG layer: **faithfulness** (claims supported by retrieved context), **answer relevancy**, **context precision**, **context recall** (five canonical metrics per [EvalHub](https://eval-hub.github.io/adapters/ragas/metrics/)). Run offline against a curated support-QA golden set (they already have `tests/simulate_conversations.py` to build from).
- **Agent-level evals**: trajectory/tool-call correctness and task completion via Langfuse LLM-as-judge on the root span ([langfuse engineering](https://langfuse.com/resources/engineering/ai-agent-evaluation)); also gate the SQL agent with a read-only Postgres role and query validation in CI (already partially in `tests/test_sql_agent.py`).

---

## 8. Recommended Target Stack

### 8.1 Concrete stack

| Layer | Choice | Replaces |
|---|---|---|
| Web/API | **FastAPI** + `uvicorn` (async), **sse-starlette** for SSE | planned FastAPI (confirmed) |
| Agent framework | **openai-agents-python** (supervisor `support_agent` + handoffs to rag/sql/web sub-agents; guardrails; sessions; built-in tracing) | google-adk |
| LLM routing | **LiteLLM** (keep — proxy/multi-provider routing, Langfuse integration) or plain OpenAI SDK | LiteLLM (keep) |
| DB layer | **SQLAlchemy 2.0 async** (`create_async_engine` + `async_sessionmaker`) + **asyncpg**; **Alembic** migrations | sync `create_engine` + psycopg2 in async tools (bug) |
| Vector search | **Postgres + pgvector ≥0.9** (HNSW), **hybrid FTS+vector with RRF** in SQL; optional pgvectorscale/ParadeDB `pg_search` | pgvector (upgrade + hybrid) |
| Reranker | **Cohere Rerank** (start) → self-hosted **BGE-reranker-v2-m3** (optional) | none (new) |
| Embeddings | **OpenAI text-embedding-3-large** (start) → **bge-m3** if multilingual (dense+sparse) | OpenAI (keep, with migration path) |
| Chunking | Structural/section-based with overlap; metadata + doc/section path | `pypdf` dump (rework ingest) |
| Web search | **Tavily** primary (agent-native), DuckDuckGo fallback, Exa optional | duckduckgo-search |
| Task queue | **ARQ + Redis** (ingestion, evals); plain asyncio for live turns | redis (already dep) |
| Observability | **Langfuse** (self-host or cloud) + **RAGAS** offline evals; OTEL GenAI optional later | ad-hoc logging |
| Real-time | SSE for tokens/tool events/sources; REST for control; WS later if steering needed | CLI loop in `main.py` |
| Dev harness | **pi** stays as the developer agent; optional `pi --mode rpc` internal ops/research agent | — |

### 8.2 Migration path (from current code)

1. **DB layer first (bug fix):** the RAG and SQL agents currently call **sync** `sqlalchemy.create_engine(...)` inside async tool functions — this blocks the event loop ([rag_agent.py](src/agent_team/rag_agent.py), [sql_agent.py](src/agent_team/sql_agent.py)). Replace with one module-level `create_async_engine` (asyncpg driver), `async_sessionmaker`, and async tools; move schema to Alembic. `pyproject.toml` already lists both `asyncpg` and `psycopg2-binary` — drop psycopg2 (or keep only as sync fallback).
2. **Agent framework swap:** port `init_support_agent` / `init_rag_agent` / `init_sql_agent` / `init_web_agent` from ADK `Agent(...)` + `FunctionTool(transfer_to_agent)` to OpenAI Agents SDK `Agent` + `handoff()`; keep the three tools as async Python functions. Enable built-in tracing and wire Langfuse. Resolve the README-vs-code inconsistency (README claims OpenAI Agents SDK).
3. **Retrieval upgrade:** rework `config/sql/retrieval.sql` — add `tsvector` generated column + GIN index, HNSW vector index, RRF-fused hybrid query; add reranker call (Cohere) between retrieval and prompt; add chunk metadata columns. Re-run `config/ingest.py` with structural chunking + overlap.
4. **API layer:** move the `while True` CLI loop in `main.py` out; expose `POST /chat` (async agent run) + `GET /chat/{id}/stream` (SSE) + `POST /chat/{id}/cancel`; persist sessions (support-agent `Session` class or own Postgres table keyed by user/ticket); move ingestion behind ARQ jobs.
5. **Web search:** swap duckduckgo for Tavily inside `web_agent.py` (keep DDG fallback behind a flag); add env-config (`TAVILY_API_KEY`).
6. **Observability:** `pip install langfuse`; instrument with the OpenAI-Agents callback handler; create a golden set from `tests/simulate_conversations.py` and run RAGAS (faithfulness, context precision/recall) in CI and as an ARQ nightly job.
7. **pi:** no production changes required; optionally add a `pi --mode rpc` subprocess for an internal research/ops agent, and use pi extensions (`pi.registerTool`) only for dev-side automation.

### 8.3 Known gaps / risks in the recommendation

- OpenAI Agents SDK's handoff/tracing is optimized for OpenAI-model function calling; if you must use Anthropic/Gemini as the primary model, re-evaluate (LangGraph or ADK may fit better) ([breakingcube](https://tech.breakingcube.com/2026/03/15/google-adk-vs-openai-agents-sdk-comparison/)).
- pi SDK APIs (v0.8x) change across releases; anything built on it should pin versions ([aibuilderclub](https://www.aibuilderclub.com/blog/pi-agent-extensions-guide)).
- Reranker + hybrid retrieval add per-query latency (~100–300 ms) — cache retrieval for repeated queries.
- Vector-DB benchmark figures vary by workload; the latency numbers in §4.1 are indicative, not a guarantee for this corpus ([digitalapplied](https://www.digitalapplied.com/blog/vector-databases-for-ai-agents-pinecone-qdrant-2026)).
