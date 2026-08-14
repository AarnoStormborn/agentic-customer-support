/**
 * src/runtime/generate.ts — LLM text generation for retrieval techniques.
 *
 * HYDE (Hypothetical Document Embeddings): generate a hypothetical passage that
 * would answer the query, embed IT, and use that for vector search — the
 * hypothesis often sits closer to the relevant chunks than the short query.
 *
 * multiQuery: generate N paraphrases of the query, retrieve for each, fuse.
 *
 * Uses a dedicated no-tools session with a cheap model (the SDK import stays in
 * src/runtime per the isolation rule; retrieval/ imports these helpers).
 */
import {
  createAgentSession,
  DefaultResourceLoader,
  getAgentDir,
  ModelRuntime,
  SessionManager,
  SettingsManager,
} from "@earendil-works/pi-coding-agent";
import { resolveSpecialistModel } from "./model.js";

const HYDE_PROMPT = (query: string) =>
  `Write a short, factual passage (2-4 sentences) that would answer this customer-support question. ` +
  `It should read like a knowledge-base/manual excerpt, using product terms. Do NOT answer the user ` +
  `directly or add commentary — output ONLY the passage.\n\nQUESTION: ${query}`;

const VARIANT_PROMPT = (query: string, n: number) =>
  `Rewrite this customer-support query as ${n} DIFFERENT search queries. Keep them concise; ` +
  `vary the wording, synonyms, and specificity (one broad, one narrow, one different angle). ` +
  `Output ONLY a JSON array of strings, e.g. ["q1","q2","q3"].\n\nQUERY: ${query}`;

let modelRuntimePromise: Promise<ModelRuntime> | null = null;
async function runtime(): Promise<ModelRuntime> {
  if (!modelRuntimePromise) {
    modelRuntimePromise = ModelRuntime.create().catch((err) => {
      modelRuntimePromise = null;
      throw err;
    });
  }
  return modelRuntimePromise;
}

async function oneShot(prompt: string): Promise<string> {
  const mr = await runtime();
  const available = await mr.getAvailable();
  const model = available.find((m) => m.provider === "anthropic" && m.id.includes("haiku")) ?? available[0];
  if (!model) throw new Error("no model available for generation");

  const loader = new DefaultResourceLoader({
    cwd: process.cwd(),
    agentDir: getAgentDir(),
    systemPromptOverride: () => "You are a precise text-generation helper. Output only what is requested.",
  });
  await loader.reload();

  const { session } = await createAgentSession({
    cwd: process.cwd(),
    agentDir: getAgentDir(),
    model,
    thinkingLevel: "off",
    modelRuntime: mr,
    resourceLoader: loader,
    noTools: "all",
    sessionManager: SessionManager.inMemory(),
    settingsManager: SettingsManager.inMemory({ compaction: { enabled: false } }),
  });
  try {
    let out = "";
    session.subscribe((event) => {
      const e = event as { type?: string; assistantMessageEvent?: { type?: string; delta?: string } };
      if (e.type === "message_update" && e.assistantMessageEvent?.type === "text_delta") {
        out += e.assistantMessageEvent.delta ?? "";
      }
    });
    await session.prompt(prompt);
    return out.trim();
  } finally {
    session.dispose();
  }
}

/** HYDE: hypothetical passage that would answer the query. */
export async function generateHypothesis(query: string): Promise<string> {
  return oneShot(HYDE_PROMPT(query));
}

/** multiQuery: N paraphrase variants of the query (JSON array, tolerant parse). */
export async function generateQueryVariants(query: string, n: number): Promise<string[]> {
  const raw = await oneShot(VARIANT_PROMPT(query, n));
  const start = raw.indexOf("[");
  const end = raw.lastIndexOf("]");
  if (start < 0 || end <= start) {
    // Fallback: split on lines
    const lines = raw.split("\n").map((l) => l.replace(/^[-*"\d.\s]+/, "").trim()).filter(Boolean);
    return lines.slice(0, n);
  }
  try {
    const parsed = JSON.parse(raw.slice(start, end + 1)) as unknown;
    if (Array.isArray(parsed)) return parsed.filter((x): x is string => typeof x === "string").slice(0, n);
  } catch {
    // fall through
  }
  return [query];
}
