/**
 * src/retrieval/embed.ts — OpenAI embeddings with a deterministic fallback.
 *
 * PRIMARY: OpenAI `text-embedding-3-small` (EMBEDDING_MODEL), batched 100/request,
 * retried with exponential backoff on 429/5xx/network errors (v1 lesson #4: no
 * swallowed errors — a failed batch fails the ingest job).
 *
 * FALLBACK: if OPENAI_API_KEY is unset we emit deterministic feature-hash vectors
 * (see CONTRACT-NOTES.md). The fallback is a hashing-trick embedding: word +
 * bigram tokens are hashed into a fixed-dim vector with random signs, then L2
 * normalized. It is stable across runs/machines (pure-JS FNV-1a), so ingest stays
 * idempotent and cosine similarity roughly reflects token overlap — enough for the
 * pipeline + query CLI to run end-to-end without an API key.
 */
import OpenAI from "openai";
import "dotenv/config";

const BATCH_SIZE = 100;
const MAX_RETRIES = 5;
const RETRY_BASE_MS = 1000;

export const EMBEDDING_MODEL: string = process.env.EMBEDDING_MODEL ?? "text-embedding-3-small";

/** Vector dimension must match `vector(n)` in schema.sql. */
export function embeddingDim(): number {
  const m = EMBEDDING_MODEL.toLowerCase();
  if (m.includes("large") || m.includes("3-large") || m.includes("002") && m.includes("ada")) return 3072;
  return 1536; // text-embedding-3-small / text-embedding-ada-002
}

export function embeddingsEnabled(): boolean {
  return Boolean(process.env.OPENAI_API_KEY);
}

let client: OpenAI | null = null;
function getClient(): OpenAI | null {
  if (!process.env.OPENAI_API_KEY) return null;
  client ??= new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  return client;
}

/** Embed a list of texts. Empty list → []. Result order matches input order. */
export async function embedTexts(texts: string[]): Promise<number[][]> {
  const dim = embeddingDim();
  const c = getClient();
  if (!c) return texts.map((t) => hashEmbedding(t, dim));

  const out: number[][] = [];
  for (let i = 0; i < texts.length; i += BATCH_SIZE) {
    const batch = texts.slice(i, i + BATCH_SIZE);
    out.push(...(await embedBatchWithRetry(c, batch, i)));
  }
  return out;
}

async function embedBatchWithRetry(c: OpenAI, batch: string[], offset: number): Promise<number[][]> {
  let lastErr: unknown = null;
  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      const res = await c.embeddings.create({ model: EMBEDDING_MODEL, input: batch });
      // OpenAI returns results in arbitrary order → sort by index to match input.
      const byIndex = new Map(res.data.map((d) => [d.index, d.embedding]));
      return batch.map((_, i) => byIndex.get(i) ?? []);
    } catch (err) {
      lastErr = err;
      const status = (err as { status?: number }).status;
      const retryable = status === 429 || (status !== undefined && status >= 500) || status === undefined;
      if (!retryable) throw err;
      if (attempt < MAX_RETRIES) {
        const delay = RETRY_BASE_MS * 2 ** (attempt - 1);
        console.warn(`[embed] batch ${offset}-${offset + batch.length} failed (${status ?? "network"}); retry ${attempt}/${MAX_RETRIES - 1} in ${delay}ms`);
        await new Promise((r) => setTimeout(r, delay));
      }
    }
  }
  throw new Error(`embedding batch ${offset}-${offset + batch.length} failed after ${MAX_RETRIES} attempts: ${String(lastErr)}`);
}

// ---------------------------------------------------------------------------
// Deterministic fallback — feature hashing (a.k.a. hashing trick)
// ---------------------------------------------------------------------------

/** FNV-1a 32-bit — pure JS, deterministic across machines/runs. */
function fnv1a(str: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** Tokenize: lowercase alnum words + word bigrams + char trigrams of the whole string. */
function tokens(text: string): string[] {
  const words = (text.toLowerCase().match(/[a-z0-9']+/g) ?? []).filter((w) => w.length > 1);
  const out = [...words];
  for (let i = 0; i + 1 < words.length; i++) out.push(`${words[i]} ${words[i + 1]}`);
  const norm = text.toLowerCase().replace(/[^a-z0-9]/g, "");
  for (let i = 0; i + 3 <= norm.length; i++) out.push(`#${norm.slice(i, i + 3)}`);
  return out.length > 0 ? out : ["<empty>"];
}

/** Deterministic unit vector in `dim` dimensions. */
export function hashEmbedding(text: string, dim = 1536): number[] {
  const vec = new Float64Array(dim);
  for (const t of tokens(text)) {
    const h1 = fnv1a(t);
    const h2 = fnv1a(`${t}\u0001`);
    const idx = h1 % dim;
    vec[idx] = (vec[idx] ?? 0) + (h2 % 2 === 0 ? 1 : -1);
  }
  // L2 normalize (v1 used normalized vectors; cosine rank == inner-product rank)
  let norm = 0;
  for (let i = 0; i < dim; i++) norm += (vec[i] ?? 0) * (vec[i] ?? 0);
  norm = Math.sqrt(norm) || 1;
  const out = new Array<number>(dim);
  for (let i = 0; i < dim; i++) out[i] = (vec[i] ?? 0) / norm;
  return out;
}
