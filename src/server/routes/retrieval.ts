/**
 * src/server/routes/retrieval.ts — per-strategy retrieval comparison (Phase 5h).
 *
 *   POST /api/retrieval/search   { query, strategy }   → one strategy's top results
 *   POST /api/retrieval/compare  { query, modes? }     → every mode side by side
 *
 * Runs searchHybrid directly (not through the agent) so the UI can show "what
 * each strategy would return" for a query — the eval comparison, live.
 */
import type { FastifyPluginAsync } from "fastify";
import { searchHybrid } from "../../retrieval/index.js";
import { RETRIEVAL_MODES, type RetrievalMode } from "../../retrieval/strategy.js";
import { getPool } from "../../db/pool.js";

interface SearchBody {
  query?: string;
  strategy?: Record<string, unknown>;
  topK?: number;
}

interface CompareBody {
  query?: string;
  /** Modes to compare (default: all except the LLM-costly ones unless asked). */
  modes?: RetrievalMode[];
  topK?: number;
}

function cleanQuery(q: string | undefined): string {
  return (q ?? "").trim().slice(0, 500);
}

export const retrievalRoutes: FastifyPluginAsync = async (app) => {
  app.post<{ Body: SearchBody }>("/api/retrieval/search", async (request, reply) => {
    const query = cleanQuery(request.body?.query);
    if (!query) {
      return reply.code(400).send({ error: "invalid_body", message: "query is required" });
    }
    const { results, relaxed, queryTimeMs, strategy } = await searchHybrid({
      query,
      topK: Math.min(10, Math.max(1, request.body?.topK ?? 5)),
      strategy: request.body?.strategy as never,
    });
    return {
      mode: strategy?.mode,
      relaxed: relaxed ?? false,
      queryTimeMs,
      results: results.slice(0, 10).map((r) => ({
        type: r.source.type,
        title: r.source.docName ?? r.source.title ?? null,
        sectionPath: r.source.sectionPath ?? null,
        page: r.source.page ?? null,
        score: Number(r.score.toFixed(4)),
        text: r.text.slice(0, 400),
      })),
    };
  });

  app.post<{ Body: CompareBody }>("/api/retrieval/compare", async (request, reply) => {
    const query = cleanQuery(request.body?.query);
    if (!query) {
      return reply.code(400).send({ error: "invalid_body", message: "query is required" });
    }
    const requested = request.body?.modes;
    const modes = (requested && requested.length > 0 ? requested : RETRIEVAL_MODES).slice(0, 6);
    const topK = Math.min(5, Math.max(1, request.body?.topK ?? 3));

    const rows = [];
    for (const mode of modes) {
      try {
        const { results, relaxed, queryTimeMs } = await searchHybrid({
          query,
          topK,
          sourceTypes: ["kb"],
          strategy: { mode },
        });
        rows.push({
          mode,
          relaxed: relaxed ?? false,
          queryTimeMs,
          top: results.slice(0, topK).map((r) => ({
            docName: r.source.docName ?? null,
            sectionPath: r.source.sectionPath ?? null,
            score: Number(r.score.toFixed(4)),
          })),
        });
      } catch (err) {
        rows.push({ mode, error: (err as Error).message, relaxed: false, queryTimeMs: 0, top: [] });
      }
    }
    return { query, topK, modes: rows };
  });

  // Manual count for the manuals table (also used by the retrieval panel header)
  app.get("/api/retrieval/corpus", async () => {
    const pool = getPool();
    const docs = await pool.query("SELECT count(*)::int AS manuals FROM documents");
    const chunks = await pool.query("SELECT count(*)::int AS chunks FROM document_chunks");
    return { manuals: docs.rows[0]?.manuals ?? 0, chunks: chunks.rows[0]?.chunks ?? 0 };
  });
};
