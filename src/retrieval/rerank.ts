/**
 * src/retrieval/rerank.ts — cross-encoder reranking (Phase 5c).
 *
 * Key-gated: only active when COHERE_API_KEY is set (else a no-op passthrough,
 * so the strategy flag never breaks an unconfigured deployment).
 * Uses Cohere's /v2/rerank REST API directly (no SDK dep).
 */
export interface Rerankable {
  text: string;
  /** Opaque id the caller re-attaches to the result. */
  id: string;
}

export function rerankEnabled(): boolean {
  return Boolean(process.env.COHERE_API_KEY);
}

/**
 * Rerank candidates by relevance to the query (descending). Returns the same
 * items reordered. Passthrough (original order) when no key is configured.
 */
export async function rerank(
  query: string,
  items: Rerankable[],
  topN = items.length,
): Promise<Rerankable[]> {
  const apiKey = process.env.COHERE_API_KEY;
  if (!apiKey || items.length < 2) return items;

  try {
    const res = await fetch("https://api.cohere.com/v2/rerank", {
      method: "POST",
      headers: {
        authorization: `Bearer ${apiKey}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: process.env.RERANK_MODEL ?? "rerank-v3.5",
        query,
        documents: items.map((i) => i.text),
        top_n: Math.min(topN, items.length),
        return_documents: false,
      }),
      signal: AbortSignal.timeout(10_000),
    });
    if (!res.ok) {
      console.warn(`[rerank] HTTP ${res.status} — using original order`);
      return items;
    }
    const data = (await res.json()) as { results?: { index: number; relevance_score: number }[] };
    const order = data.results ?? [];
    return order
      .map((r) => items[r.index])
      .filter((x): x is Rerankable => Boolean(x));
  } catch (err) {
    console.warn("[rerank] failed — using original order:", (err as Error).message);
    return items;
  }
}
