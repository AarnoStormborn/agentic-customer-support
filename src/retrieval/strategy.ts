/**
 * src/retrieval/strategy.ts — configurable retrieval strategy (Phase 5c).
 *
 * The UI sends a strategy with every chat message; it flows through the API
 * into the kb_search tool and searchHybrid. Every knob is optional — defaults
 * reproduce the current hybrid behavior exactly.
 */
export type RetrievalMode = "hybrid" | "vector" | "keyword" | "hyde" | "hyde-hybrid";

export interface RetrievalStrategy {
  /** Which retrieval pipeline to run (applies to the KB source; tickets stay FTS+relax). */
  mode: RetrievalMode;
  /** Number of results per source (1-10). */
  topK: number;
  /** RRF fusion constant: higher = more rank-position dominance (10-120). */
  rrfK: number;
  /** FTS query relaxation — auto-drop unmatched terms (Phase 5b.8). */
  relax: boolean;
  /** Generate N paraphrased queries, retrieve each, fuse (LLM cost + latency). */
  multiQuery: boolean;
  /** How many paraphrases when multiQuery is on. */
  numVariants: number;
  /** Rule-based synonym expansion (no LLM). */
  queryExpansion: boolean;
  /** Cross-encoder rerank of candidates (requires COHERE_API_KEY; else skipped). */
  rerank: boolean;
}

export const DEFAULT_STRATEGY: RetrievalStrategy = {
  mode: "hybrid",
  topK: 5,
  rrfK: 60,
  relax: true,
  multiQuery: false,
  numVariants: 3,
  queryExpansion: false,
  rerank: false,
};

export const RETRIEVAL_MODES: RetrievalMode[] = ["hybrid", "vector", "keyword", "hyde", "hyde-hybrid"];

function clamp(n: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, n));
}

/** Merge a partial (API/UI-supplied) strategy over defaults, validating ranges. */
export function normalizeStrategy(input?: Partial<RetrievalStrategy> | null): RetrievalStrategy {
  const s: RetrievalStrategy = { ...DEFAULT_STRATEGY };
  if (!input) return s;
  if (typeof input.mode === "string" && (RETRIEVAL_MODES as string[]).includes(input.mode)) {
    s.mode = input.mode as RetrievalMode;
  }
  if (typeof input.topK === "number") s.topK = clamp(Math.round(input.topK), 1, 10);
  if (typeof input.rrfK === "number") s.rrfK = clamp(Math.round(input.rrfK), 10, 120);
  if (typeof input.relax === "boolean") s.relax = input.relax;
  if (typeof input.multiQuery === "boolean") s.multiQuery = input.multiQuery;
  if (typeof input.numVariants === "number") s.numVariants = clamp(Math.round(input.numVariants), 2, 5);
  if (typeof input.queryExpansion === "boolean") s.queryExpansion = input.queryExpansion;
  if (typeof input.rerank === "boolean") s.rerank = input.rerank;
  return s;
}
