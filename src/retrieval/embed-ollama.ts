/**
 * src/retrieval/embed-ollama.ts — offline embeddings via a local Ollama server.
 *
 * Truly local: model runs on this machine, no API key, no network egress.
 * Ollama's /api/embed accepts a batch of inputs and returns dense vectors.
 */
const OLLAMA_URL = process.env.OLLAMA_URL ?? "http://localhost:11434";
export const OLLAMA_MODEL = process.env.OLLAMA_MODEL ?? "nomic-embed-text";
export const OLLAMA_DIM = 768; // nomic-embed-text

export function ollamaConfigured(): boolean {
  return process.env.EMBEDDING_BACKEND === "ollama" || process.env.OLLAMA_URL !== undefined;
}

/** Embed a batch of texts with the local model. Throws on any failure. */
export async function embedViaOllama(texts: string[], model = OLLAMA_MODEL): Promise<number[][]> {
  const res = await fetch(`${OLLAMA_URL}/api/embed`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model, input: texts }),
    signal: AbortSignal.timeout(60_000),
  });
  if (!res.ok) {
    throw new Error(`ollama /api/embed HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
  }
  const data = (await res.json()) as { embeddings?: number[][] };
  if (!Array.isArray(data.embeddings) || data.embeddings.length !== texts.length) {
    throw new Error(`ollama returned ${data.embeddings?.length ?? 0} embeddings for ${texts.length} inputs`);
  }
  return data.embeddings;
}
