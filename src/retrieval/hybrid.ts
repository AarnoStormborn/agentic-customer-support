/**
 * src/retrieval/hybrid.ts — hybrid retrieval per contract + design
 * backend-agent-retrieval §4.1 / data-management §5 (H1).
 *
 * Two source types (contract HybridResult.source.type):
 *   "kb"  — document_chunks: Postgres FTS (GIN tsvector) + pgvector cosine (HNSW),
 *           fused with Reciprocal Rank Fusion RRF: score = 1/(k + rank), k = 60.
 *   "sql" — tickets: FTS over narrative+subject+product, RRF-style score from rank.
 *
 * Parameterized SQL ONLY (v1 lesson #2: the old f-string `LIMIT {top_k}` injection
 * bug). Every value is bound via $1..$n; the query text is static.
 *
 * Merge semantics: when both sources are enabled, each contributes
 * ceil(topK/2) results, kb first (see CONTRACT-NOTES.md).
 */
import { getPool } from "../db/pool.js";
import { embedTexts } from "./embed.js";
import { tsQueryVariants, relaxedSearch } from "./relax.js";

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
}

export interface HybridSearchResponse {
  results: HybridResult[];
  queryTimeMs: number;
  /** True when FTS needed relaxation (an AND term was dropped to find results). */
  relaxed?: boolean;
}

const RRF_K = 60;
const FTS_LIMIT = 50; // candidates per retriever before fusion

export async function searchHybrid(opts: HybridSearchOptions): Promise<HybridSearchResponse> {
  const started = Date.now();
  const query = opts.query.trim();
  const topK = Math.max(1, opts.topK ?? 5);
  const sourceTypes = opts.sourceTypes ?? ["kb", "sql"];
  const filter = opts.filter ?? {};

  const hasLexemes = /[a-zA-Z0-9]/.test(query);
  const perSource = sourceTypes.length > 1 ? Math.max(1, Math.ceil(topK / 2)) : topK;
  const results: HybridResult[] = [];
  let relaxed = false;

  if (sourceTypes.includes("kb") && hasLexemes) {
    const kb = await searchKb(query, perSource, filter);
    results.push(...kb.rows);
    if (kb.relaxed) relaxed = true;
  }
  if (sourceTypes.includes("sql") && hasLexemes) {
    const sql = await searchSql(query, perSource);
    results.push(...sql.rows);
    if (sql.relaxed) relaxed = true;
  }

  return { results, queryTimeMs: Date.now() - started, relaxed };
}

// ---------------------------------------------------------------------------
// kb — FTS + vector fused by RRF in one query (design §4.1 SQL)
// ---------------------------------------------------------------------------

async function searchKb(
  query: string,
  topK: number,
  filter: Record<string, unknown>,
): Promise<{ rows: HybridResult[]; relaxed: boolean }> {
  const pool = getPool();
  const embedding = (await embedTexts([query]))[0]!;

  // Allowlisted, statically-typed filter → safe to interpolate into SQL text.
  const clauses: string[] = [];
  const filterParams: unknown[] = [];
  if (typeof filter.docName === "string" && filter.docName) {
    filterParams.push(filter.docName);
    clauses.push(`d.doc_name = $${filterParams.length + 3}`);
  }
  if (typeof filter.section === "string" && filter.section) {
    filterParams.push(filter.section);
    clauses.push(`c.section = $${filterParams.length + 3}`);
  }
  const filterSql = clauses.length > 0 ? `WHERE ${clauses.join(" AND ")}` : "";
  const filterJoin = clauses.length > 0 ? ` AND (${clauses.join(" AND ")})` : "";

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
             (COALESCE(1.0 / (${RRF_K} + fts.rank), 0.0) +
              COALESCE(1.0 / (${RRF_K} + vec.rank), 0.0))::float8 AS score
      FROM fts FULL OUTER JOIN vec USING (id)
    )
    SELECT c.chunk_id, c.chunk_text, d.doc_name, c.section, c.heading_path, c.page_start, rrf.score
    FROM rrf
    JOIN document_chunks c ON c.chunk_id = rrf.id
    JOIN documents d      ON d.doc_id = c.doc_id
    ${filterSql}
    ORDER BY rrf.score DESC
    LIMIT $3`;

  // FTS relaxation: websearch_to_tsquery ANDs terms — retry with progressively
  // fewer terms when the strict query matches nothing (vector side is unaffected).
  const runVariant = async (variant: string) => {
    const params = [variant, `[${embedding.join(",")}]`, topK, ...filterParams];
    const { rows } = await pool.query(sql, params);
    return rows;
  };
  const { rows, relaxed } = await relaxedSearch(tsQueryVariants(query), runVariant);

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

// ---------------------------------------------------------------------------
// sql — tickets FTS (narrative + subject + product), RRF-style score
// ---------------------------------------------------------------------------

async function searchSql(query: string, topK: number): Promise<{ rows: HybridResult[]; relaxed: boolean }> {
  const pool = getPool();
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
           (1.0 / (${RRF_K} + ranked.rank))::float8 AS score
    FROM ranked
    JOIN tickets t USING (ticket_id)
    ORDER BY ranked.rank
    LIMIT $2`;

  const runVariant = async (variant: string) => {
    const { rows } = await pool.query(sql, [variant, topK]);
    return rows;
  };
  const { rows, relaxed } = await relaxedSearch(tsQueryVariants(query), runVariant);

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
