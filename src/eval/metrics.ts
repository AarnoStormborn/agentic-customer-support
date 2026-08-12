/**
 * src/eval/metrics.ts — retrieval metrics (deterministic, no LLM needed).
 *
 * recall@k   — fraction of expected sources found in the top-k results
 * precision@k — fraction of top-k results that are expected sources
 * MRR        — mean reciprocal rank of the first expected source
 * hitRate    — did ANY expected source appear in top-k
 */
export function recallAtK(expected: Set<string>, retrieved: string[], k: number): number {
  const top = retrieved.slice(0, k);
  if (expected.size === 0) return top.length === 0 ? 1 : 0;
  const hits = top.filter((id) => expected.has(id)).length;
  return hits / expected.size;
}

export function precisionAtK(expected: Set<string>, retrieved: string[], k: number): number {
  const top = retrieved.slice(0, k);
  if (top.length === 0) return 0;
  const hits = top.filter((id) => expected.has(id)).length;
  return hits / top.length;
}

export function reciprocalRank(expected: Set<string>, retrieved: string[]): number {
  for (let i = 0; i < retrieved.length; i++) {
    const id = retrieved[i];
    if (id !== undefined && expected.has(id)) return 1 / (i + 1);
  }
  return 0;
}

export function hitRate(expected: Set<string>, retrieved: string[]): boolean {
  return retrieved.some((id) => expected.has(id));
}

export interface EvalScore {
  recallAtK: number;
  precisionAtK: number;
  mrr: number;
  hit: boolean;
}

export function scoreQuery(expected: Set<string>, retrieved: string[], k: number): EvalScore {
  return {
    recallAtK: recallAtK(expected, retrieved, k),
    precisionAtK: precisionAtK(expected, retrieved, k),
    mrr: reciprocalRank(expected, retrieved),
    hit: hitRate(expected, retrieved),
  };
}

export function average(scores: EvalScore[], key: keyof EvalScore): number {
  if (scores.length === 0) return 0;
  return scores.reduce((sum, s) => sum + Number(s[key]), 0) / scores.length;
}
