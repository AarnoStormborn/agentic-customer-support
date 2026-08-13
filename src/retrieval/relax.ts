/**
 * src/retrieval/relax.ts — query relaxation for FTS (Phase 5b.8).
 *
 * `websearch_to_tsquery` ANDs all terms: ONE unmatched term zeroes the result
 * set (the eval's "television" killed a refund query — tickets say "LG Smart
 * TV"). Relaxation progressively drops the least-specific trailing terms and
 * retries until results come back.
 *
 * Variant order: strictest first (all terms) → drop last → … → single term.
 * Quoted phrases ("exact phrase") are preserved as single units.
 */

/** Tokenize a websearch query, keeping quoted phrases as single units. */
export function queryTerms(query: string): string[] {
  const terms: string[] = [];
  const re = /"([^"]+)"|(\S+)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(query))) {
    terms.push(m[0]); // preserve quotes for phrases
  }
  return terms;
}

/**
 * Ordered tsquery input variants for a query: all terms, then progressively
 * dropping trailing terms. Never empty (last variant = single term).
 */
export function tsQueryVariants(query: string): string[] {
  const terms = queryTerms(query.trim());
  if (terms.length === 0) return [];
  const variants: string[] = [];
  for (let n = terms.length; n >= 1; n--) {
    variants.push(terms.slice(0, n).join(" "));
  }
  return variants;
}

export interface RelaxedResult<T> {
  rows: T[];
  /** The variant that produced rows (first non-empty in strictest-first order). */
  variant: string;
  /** Index into variants — 0 = no relaxation needed. */
  attempts: number;
  relaxed: boolean;
}

/**
 * Run `queryFn(variant)` per variant (strictest first); return the first
 * non-empty result set. `queryFn` must return [] when a variant matches nothing.
 */
export async function relaxedSearch<T>(
  variants: string[],
  queryFn: (variant: string) => Promise<T[]>,
): Promise<RelaxedResult<T>> {
  for (let i = 0; i < variants.length; i++) {
    const rows = await queryFn(variants[i]!);
    if (rows.length > 0) {
      return { rows, variant: variants[i]!, attempts: i + 1, relaxed: i > 0 };
    }
  }
  return { rows: [], variant: variants[variants.length - 1] ?? "", attempts: variants.length, relaxed: false };
}
