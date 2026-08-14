/**
 * kb_search — knowledge-base retrieval tool (the rag specialist's one tool,
 * also exposed directly to the supervisor).
 *
 * Calls the retrieval contract (src/retrieval/index.ts). The local copy serves
 * the mock KB by default (RETRIEVAL_MODE=mock); with RETRIEVAL_MODE=real it
 * loads the real hybrid search (pgvector + FTS + RRF + optional rerank).
 */

import { defineTool } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { searchHybrid } from "../retrieval/index.js";
import type { HybridResult } from "../retrieval/index.js";
import { TOOL_NAMES } from "../config/limits.js";
import type { RetrievalStrategy } from "../retrieval/strategy.js";

function formatResults(results: HybridResult[]): string {
  if (results.length === 0) {
    return "No matching knowledge-base entries found.";
  }
  const lines = results.map((r, i) => {
    const src = r.source;
    const where = src.type === "kb"
      ? [src.docName, src.sectionPath].filter(Boolean).join(" > ")
      : (src.title ?? "ticket");
    const url = src.url ? ` (${src.url})` : "";
    return `[${i + 1}] ${where} — score ${r.score.toFixed(2)}${url}\n    ${r.text}`;
  });
  return lines.join("\n\n");
}

export const kbSearchTool = defineTool({
  name: TOOL_NAMES.kbSearch,
  label: "Knowledge Base Search",
  description:
    "Search the product manuals / knowledge base for technical how-to and troubleshooting " +
    "information (e.g. Wi-Fi reset on an LG TV). Hybrid search: vector similarity + full-text, " +
    "optionally reranked when a reranker key is configured. Returns matching chunks with " +
    "sources (manual name, section, URL).",
  parameters: Type.Object({
    query: Type.String({ description: "The search query, e.g. 'lg tv wifi reset'" }),
    topK: Type.Optional(Type.Number({ description: "Max results to return (1-10, default 5)" })),
  }),
  execute: async (_toolCallId, params, signal) => executeKbSearch(params, signal),
});

/**
 * Build a kb_search tool bound to a specific retrieval strategy (Phase 5c).
 * Used by the specialist sub-agents so the UI's retrieval config reaches the
 * retrieval layer without any shared mutable state.
 */
export function buildKbTool(strategy: RetrievalStrategy): typeof kbSearchTool {
  return defineTool({
    name: TOOL_NAMES.kbSearch,
    label: "Knowledge Base Search",
    description: kbSearchTool.description,
    parameters: kbSearchTool.parameters as never,
    execute: async (_toolCallId, params, signal) =>
      executeKbSearch(params, signal, strategy),
  });
}

async function executeKbSearch(
  params: { query: string; topK?: number },
  signal?: AbortSignal,
  strategy?: RetrievalStrategy,
) {
  signal?.throwIfAborted();
  const topK = Math.min(Math.max(params.topK ?? strategy?.topK ?? 5, 1), 10);
  const { results, queryTimeMs, relaxed } = await searchHybrid({
    query: params.query,
    topK,
    sourceTypes: ["kb"],
    strategy,
  });
  signal?.throwIfAborted();

  const sources = results.map((r) => ({
    type: r.source.type,
    title: r.source.docName ?? r.source.title ?? null,
    sectionPath: r.source.sectionPath ?? null,
    url: r.source.url ?? null,
    score: r.score,
  }));

  return {
    content: [{ type: "text" as const, text: formatResults(results) }],
    details: {
      tool: TOOL_NAMES.kbSearch,
      query: params.query,
      count: results.length,
      queryTimeMs,
      sources,
      ...(relaxed ? { relaxed: true } : {}),
      mode: strategy?.mode ?? "hybrid",
    },
  };
}
