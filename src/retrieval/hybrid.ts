/**
 * src/retrieval/hybrid.ts — strategy-aware hybrid retrieval (Phase 5c).
 *
 * Sources:
 *   "kb"  — document_chunks: Postgres FTS (GIN tsvector) + pgvector cosine (HNSW),
 *           fused with Reciprocal Rank Fusion (RRF: score = Σ 1/(k + rank)).
 *   "sql" — tickets: FTS over search_tsv (always keyword + relax; modes apply
 *           to the KB source).
 *
 * Configurable via RetrievalStrategy (see strategy.ts):
 *   mode hybrid | vector | keyword | hyde | hyde-hybrid
 *   rrfK / topK / relax / multiQuery / queryExpansion / rerank
 *
 * Parameterized SQL ONLY (v1 lesson #2). Every value is bound via $1..$n.
 */
import { getPool } from "../db/pool.js";
import { embedTexts } from "./embed.js";
import { tsQueryVariants, relaxedSearch } from "./relax.js";
import { expandQuery } from "./expand.js";
import { rerank } from "./rerank.js";
import { normalizeStrategy, type RetrievalStrategy } from "./strategy.js";
import { generateHypothesis, generateQueryVariants } from "../runtime/generate.js";

export interface HybridSource {
  type: "kb" | "sql";
  title?: string;
  docName?: string;
  sectionPath?: string;
  page?: number;
  url?: string | null;
  row?: Record<string, unknown>;
}

export interface HybridResult {
  text: string;
  source: HybridSource;
  score: number;
}

export interface HybridSearchOptions {
  query: string;
  topK?: number; // default 5
  sourceTypes?: ("kb" | "sql")[]; // default ["kb", "sql"]
  /** Allowlisted filters: { docName?: string; section?: string } */
  filter?: Record<string, unknown>;
  /** Retrieval strategy knobs (UI/API-supplied; defaults reproduce old behavior). */
  strategy?: Partial<RetrievalStrategy>;
}

export interface HybridSearchResponse {
  results: HybridResult[];
  queryTimeMs: number;
  /** True when FTS needed relaxation (an AND term was dropped to find results). */
  relaxed?: boolean;
  /** The strategy actually used (after normalization). */
  strategy?: RetrievalStrategy;
}

const DEFAULT_RRF_K = 60;
const FTS_LIMIT = 50;

/** Stable identity for fusing/deduping results. */
function resultKey(r: HybridResult): string {
  if (r.source.type === "sql") {
    return `sql:${String((r.source.row as { ticket_id?: unknown } | undefined)?.ticket_id ?? "")}`;
  }
  return `kb:${r.source.docName ?? ""}:${r.text.slice(0, 120)}`;
}

/** RRF fuse multiple ordered candidate groups (dedup by key). */
function fuseByRrf(groups: HybridResult[][], k: number, perGroupTop: number): HybridResult[] {
  const scores = new Map<string, { score: number; result: HybridResult }>();
  for (const group of groups) {
    for (const [rank, result] of group.slice(0, perGroupTop).entries()) {
      const key = resultKey(result);
      const entry = scores.get(key) ?? { score: 0, result };
      entry.score += 1 / (k + rank + 1);
      scores.set(key, entry);
    }
  }
  return [...scores.values()].sort((a, b) => b.score - a.score).map((e) => ({ ...e.result, score: e.score }));
}

// ---------------------------------------------------------------------------
// kb — FTS + vector (mode-aware) fused by RRF
// ---------------------------------------------------------------------------

interface KbCoreParams {
  query: string; // the actual text passed to FTS (possibly expanded)
  embeddingText: string; // text embedded for the vector side (query or HYDE)
  topK: number;
  filter: Record<string, unknown>;
  strategy: RetrievalStrategy;
}

/** One KB search execution (vector / fts / both per mode), no multiQuery. */
async function kbSearchCore(p: KbCoreParams): Promise<{ rows: HybridResult[]; relaxed: boolean }> {
  const pool = getPool();
  const embedding = (await embedTexts([p.embeddingText]))[0]!;

  const clauses: string[] = [];
  const filterParams: unknown[] = [];
  if (typeof p.filter.docName === "string" && p.filter.docName) {
    filterParams.push(p.filter.docName);
    clauses.push(`d.doc_name = $${filterParams.length + 2}`);
  }
  if (typeof p.filter.section === "string" && p.filter.section) {
    filterParams.push(p.filter.section);
    clauses.push(`c.section = $${filterParams.length + 2}`);
  }
  const filterSql = clauses.length > 0 ? `WHERE ${clauses.join(" AND ")}` : "";
  const filterJoin = clauses.length > 0 ? ` AND (${clauses.join(" AND ")})` : "";

  const fts = p.strategy.mode === "vector" || p.strategy.mode === "hyde" ? null : true;
  const vec = p.strategy.mode === "keyword" ? null : true;
  const useHypothesisVec = p.strategy.mode === "hyde" || p.strategy.mode === "hyde-hybrid";
  const fromClause = fts && vec ? "fts FULL OUTER JOIN vec USING (id)" : fts ? "fts" : "vec";
  const scoreExpr = fts && vec
    ? `(COALESCE(1.0 / (${p.strategy.rrfK} + fts.rank), 0.0) + COALESCE(1.0 / (${p.strategy.rrfK} + vec.rank), 0.0))::float8`
    : `COALESCE(1.0 / (${p.strategy.rrfK} + ${fts ? "fts" : "vec"}.rank), 0.0)::float8`;

  const sql = `
    WITH fts AS (
      SELECT c.chunk_id AS id,
             row_number() OVER (
               ORDER BY ts_rank_cd(to_tsvector('english', c.chunk_text),
                                   websearch_to_tsquery('english', $1)) DESC
             ) AS rank
      FROM document_chunks c
      JOIN documents d ON d.doc_id = c.doc_id
      WHERE to_tsvector('english', c.chunk_text) @@ websearch_to_tsquery('english', $1)
      ${filterJoin}
      LIMIT ${FTS_LIMIT}
    ),
    vec AS (
      SELECT c.chunk_id AS id,
             row_number() OVER (ORDER BY c.embedding <=> $2::vector) AS rank
      FROM document_chunks c
      JOIN documents d ON d.doc_id = c.doc_id
      WHERE c.embedding IS NOT NULL
      ${filterJoin}
      LIMIT ${FTS_LIMIT}
    ),
    rrf AS (
      SELECT id,
             ${scoreExpr} AS score
      FROM ${fromClause}
    )
    SELECT c.chunk_id, c.chunk_text, d.doc_name, c.section, c.heading_path, c.page_start, rrf.score
    FROM rrf
    JOIN document_chunks c ON c.chunk_id = rrf.id
    JOIN documents d      ON d.doc_id = c.doc_id
    ${filterSql}
    ORDER BY rrf.score DESC
    LIMIT $3`;

  // Relaxation applies to the FTS side; vector-only modes run the strict query once.
  const runVariant = async (variant: string) => {
    const params = [variant, `[${embedding.join(",")}]`, p.topK, ...filterParams];
    const { rows } = await pool.query(sql, params);
    return rows;
  };

  let rows: Record<string, unknown>[];
  let relaxed = false;
  if (fts) {
    const searchQuery = p.strategy.queryExpansion ? expandQuery(p.query) : p.query;
    const variants = p.strategy.relax ? tsQueryVariants(searchQuery) : [searchQuery];
    const r = await relaxedSearch(variants, runVariant);
    rows = r.rows;
    relaxed = r.relaxed;
  } else {
    rows = await runVariant(p.query);
  }

  return {
    rows: rows.map((r) => ({
      text: r.chunk_text as string,
      source: {
        type: "kb" as const,
        docName: r.doc_name as string,
        sectionPath: (r.heading_path as string) ?? (r.section as string | null) ?? undefined,
        page: (r.page_start as number | null) ?? undefined,
        url: null,
      },
      score: r.score as number,
    })),
    relaxed,
  };
}

async function searchKb(
  query: string,
  topK: number,
  filter: Record<string, unknown>,
  strategy: RetrievalStrategy,
): Promise<{ rows: HybridResult[]; relaxed: boolean }> {
  // HYDE: embed a hypothetical answer instead of the raw query.
  let embeddingText = query;
  if (strategy.mode === "hyde" || strategy.mode === "hyde-hybrid") {
    embeddingText = await generateHypothesis(query);
  }

  const base = { query, embeddingText, topK, filter, strategy };

  if (!strategy.multiQuery) {
    return kbSearchCore(base);
  }

  // multiQuery: retrieve per paraphrase, then RRF-fuse the groups.
  const variants = [query, ...(await generateQueryVariants(query, strategy.numVariants))];
  const groups: HybridResult[][] = [];
  let relaxed = false;
  for (const v of variants) {
    const g = await kbSearchCore({ ...base, query: v });
    groups.push(g.rows);
    if (g.relaxed) relaxed = true;
  }
  return { rows: fuseByRrf(groups, strategy.rrfK, topK * 2), relaxed };
}

// ---------------------------------------------------------------------------
// sql — tickets FTS (search_tsv), relax + rrfK aware
// ---------------------------------------------------------------------------

async function searchSql(
  query: string,
  topK: number,
  strategy: RetrievalStrategy,
): Promise<{ rows: HybridResult[]; relaxed: boolean }> {
  const pool = getPool();
  const searchQuery = strategy.queryExpansion ? expandQuery(query) : query;
  const sql = `
    WITH fts AS (
      SELECT ticket_id, search_tsv AS tsv
      FROM tickets
      WHERE search_tsv @@ websearch_to_tsquery('english', $1)
    ),
    ranked AS (
      SELECT ticket_id, row_number() OVER (ORDER BY ts_rank_cd(tsv, websearch_to_tsquery('english', $1)) DESC) AS rank
      FROM fts
    )
    SELECT t.ticket_id, t.ticket_subject, t.complaint_narrative, t.product_purchased,
           t.ticket_type, t.ticket_priority, t.ticket_channel, t.status,
           t.customer_name, t.customer_email, t.date_of_purchase, t.source,
           t.source_ticket_id,
           (1.0 / (${strategy.rrfK} + ranked.rank))::float8 AS score
    FROM ranked
    JOIN tickets t USING (ticket_id)
    ORDER BY ranked.rank
    LIMIT $2`;

  const runVariant = async (variant: string) => {
    const { rows } = await pool.query(sql, [variant, topK]);
    return rows;
  };
  const variants = strategy.relax ? tsQueryVariants(searchQuery) : [searchQuery];
  const { rows, relaxed } = await relaxedSearch(variants, runVariant);

  return {
    rows: rows.map((r) => ({
      text: (r.complaint_narrative as string) ?? (r.ticket_subject as string) ?? "",
      source: {
        type: "sql" as const,
        title: `ticket #${r.ticket_id as number} (${r.product_purchased as string})`,
        row: {
          ticket_id: r.ticket_id,
          source: r.source,
          source_ticket_id: r.source_ticket_id,
          product_purchased: r.product_purchased,
          ticket_type: r.ticket_type,
          ticket_priority: r.ticket_priority,
          ticket_channel: r.ticket_channel,
          ticket_status: r.status,
          ticket_subject: r.ticket_subject,
          date_of_purchase: r.date_of_purchase,
        },
        url: null,
      },
      score: r.score as number,
    })),
    relaxed,
  };
}

// ---------------------------------------------------------------------------
// searchHybrid — orchestration
// ---------------------------------------------------------------------------

export async function searchHybrid(opts: HybridSearchOptions): Promise<HybridSearchResponse> {
  const started = Date.now();
  const query = opts.query.trim();
  const strategy = normalizeStrategy(opts.strategy);
  const topK = Math.max(1, strategy.topK);
  const sourceTypes = opts.sourceTypes ?? ["kb", "sql"];
  const filter = opts.filter ?? {};

  const hasLexemes = /[a-zA-Z0-9]/.test(query);
  const perSource = sourceTypes.length > 1 ? Math.max(1, Math.ceil(topK / 2)) : topK;
  const results: HybridResult[] = [];
  let relaxed = false;

  if (sourceTypes.includes("kb") && hasLexemes) {
    const kb = await searchKb(query, perSource, filter, strategy);
    results.push(...kb.rows);
    if (kb.relaxed) relaxed = true;
  }
  if (sourceTypes.includes("sql") && hasLexemes) {
    const sql = await searchSql(query, perSource, strategy);
    results.push(...sql.rows);
    if (sql.relaxed) relaxed = true;
  }

  // Optional cross-encoder rerank of the final candidates (key-gated).
  let finalResults = results;
  if (strategy.rerank && results.length > 1) {
    const items = results.map((r, i) => ({ text: r.text.slice(0, 1000), id: String(i) }));
    const ranked = await rerank(query, items, results.length);
    const byId = new Map(items.map((it, i) => [it.id, results[i]]));
    finalResults = ranked.map((it) => byId.get(it.id)!).filter(Boolean);
  }

  return {
    results: finalResults,
    queryTimeMs: Date.now() - started,
    relaxed,
    strategy,
  };
}
