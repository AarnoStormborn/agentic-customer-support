/**
 * src/eval/judge.ts — LLM judge for answer faithfulness (Phase 5b.9).
 *
 * The retrieval eval (run.ts) scores whether the RIGHT SOURCES were retrieved.
 * This judge scores whether the AGENT'S ANSWER is faithful to those sources:
 * every claim should trace back to retrieved content — no fabrication, no
 * contradiction, no unsupported citations.
 *
 * Verdict JSON shape (judge model is instructed to output exactly this):
 *   { "faithfulness": 1-5, "rationale": "…", "verdict": "pass" | "fail" }
 */
import { ModelRuntime } from "@earendil-works/pi-coding-agent";

export interface FaithfulnessVerdict {
  faithfulness: number; // 1-5
  rationale: string;
  verdict: "pass" | "fail";
  /** Raw judge output (for debugging). */
  raw: string;
}

export const FAITHFULNESS_THRESHOLD = 4; // ≥4 → pass

export function buildJudgePrompt(question: string, answer: string, sources: unknown[]): string {
  return `You are an evaluation judge. Score whether an AI support answer is FAITHFUL to its cited sources.

RULES:
- 5 = every claim in the answer is directly supported by the sources; nothing invented.
- 4 = mostly supported; minor wording, no substantive fabrication.
- 3 = some claims supported, some unsupported (hallucinated or contradicted).
- 2 = mostly unsupported or contradicts the sources.
- 1 = fabricated / ignores the sources entirely.
- An answer that honestly says "the sources don't cover this" is fine if it says so.
- A citation that the answer never actually used counts against faithfulness.

Respond with STRICT JSON only (no markdown, no commentary):
{"faithfulness": <1-5>, "rationale": "<1-2 sentences>", "verdict": "pass"|"fail"}
(verdict = "pass" when faithfulness >= ${FAITHFULNESS_THRESHOLD})

QUESTION:
${question}

ANSWER:
${answer}

SOURCES:
${JSON.stringify(sources, null, 1).slice(0, 12000)}`;
}

/** Parse the judge model's JSON output (tolerates code fences / trailing text). */
export function parseVerdict(raw: string): Omit<FaithfulnessVerdict, "raw"> {
  const cleaned = raw
    .replace(/```json\s*/gi, "")
    .replace(/```/g, "")
    .trim();
  const start = cleaned.indexOf("{");
  const end = cleaned.lastIndexOf("}");
  if (start < 0 || end <= start) {
    throw new Error(`judge output is not JSON: ${raw.slice(0, 120)}`);
  }
  const parsed = JSON.parse(cleaned.slice(start, end + 1)) as {
    faithfulness?: unknown;
    rationale?: unknown;
    verdict?: unknown;
  };
  const faithfulness = Number(parsed.faithfulness);
  const verdict = parsed.verdict === "pass" ? ("pass" as const) : ("fail" as const);
  return {
    faithfulness: Number.isFinite(faithfulness) ? Math.min(5, Math.max(1, faithfulness)) : 1,
    rationale: typeof parsed.rationale === "string" ? parsed.rationale : "(no rationale)",
    verdict,
  };
}

let modelRuntimePromise: Promise<ModelRuntime> | null = null;

async function judgeModelRuntime(): Promise<ModelRuntime> {
  if (!modelRuntimePromise) {
    modelRuntimePromise = ModelRuntime.create().catch((err) => {
      modelRuntimePromise = null; // allow retry on next call
      throw err;
    });
  }
  return modelRuntimePromise;
}

/** Run the judge on one (question, answer, sources) triple. */
export async function judgeAnswer(
  question: string,
  answer: string,
  sources: unknown[],
): Promise<FaithfulnessVerdict> {
  const runtime = await judgeModelRuntime();
  const available = await runtime.getAvailable();
  // Cheap judge: prefer an Anthropic haiku-class model, else first available.
  const judge =
    available.find((m) => m.provider === "anthropic" && m.id.includes("haiku")) ??
    available.find((m) => m.provider === "openai") ??
    available[0];
  if (!judge) throw new Error("no model available for judge");

  const { createAgentSession, DefaultResourceLoader, getAgentDir, SessionManager, SettingsManager } =
    await import("@earendil-works/pi-coding-agent");

  const loader = new DefaultResourceLoader({
    cwd: process.cwd(),
    agentDir: getAgentDir(),
    systemPromptOverride: () =>
      "You are a strict, concise evaluation judge. Output exactly the JSON schema requested.",
  });
  await loader.reload();

  const { session } = await createAgentSession({
    cwd: process.cwd(),
    agentDir: getAgentDir(),
    model: judge,
    thinkingLevel: "off",
    modelRuntime: runtime,
    resourceLoader: loader,
    noTools: "all", // the judge only reads input and replies
    sessionManager: SessionManager.inMemory(),
    settingsManager: SettingsManager.inMemory({ compaction: { enabled: false } }),
  });

  try {
    let raw = "";
    session.subscribe((event) => {
      const e = event as { type?: string; assistantMessageEvent?: { type?: string; delta?: string } };
      if (e.type === "message_update" && e.assistantMessageEvent?.type === "text_delta") {
        raw += e.assistantMessageEvent.delta ?? "";
      }
    });
    await session.prompt(buildJudgePrompt(question, answer, sources));
    return { ...parseVerdict(raw), raw: raw.slice(0, 1000) };
  } finally {
    session.dispose();
  }
}
