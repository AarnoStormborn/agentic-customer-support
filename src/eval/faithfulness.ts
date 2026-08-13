/**
 * src/eval/faithfulness.ts — answer-level faithfulness eval (npm run eval:answer).
 *
 * For each golden case: run a REAL agent turn (createSupportRuntime → prompt),
 * capture the final answer + sources, then have the LLM judge score faithfulness
 * (1-5) + verdict. Writes reports/faithfulness-report.json.
 *
 * Slower + uses model tokens (agent turns + judge calls) — deliberately separate
 * from `npm run eval` (fast, deterministic retrieval metrics).
 */
import "dotenv/config";
import { writeFile, mkdir } from "node:fs/promises";
import { GOLDEN_SET } from "./golden.js";
import { judgeAnswer, type FaithfulnessVerdict } from "./judge.js";
import { createSupportRuntime } from "../runtime/index.js";

interface AnswerCase {
  id: string;
  query: string;
  source: "kb" | "sql" | "web";
  answer: string;
  sources: unknown[];
  verdict: FaithfulnessVerdict;
}

/** Run one agent turn and capture the final answer text + sources. */
async function runAgentTurn(query: string): Promise<{ answer: string; sources: unknown[] }> {
  const rt = await createSupportRuntime({});
  try {
    let answer = "";
    let sources: unknown[] = [];
    rt.subscribe((event) => {
      const e = event as {
        type?: string;
        assistantMessageEvent?: { type?: string; delta?: string };
        sources?: unknown[];
      };
      if (e.type === "message_update" && e.assistantMessageEvent?.type === "text_delta") {
        answer += e.assistantMessageEvent.delta ?? "";
      }
      if (e.type === "agent_settled") {
        // SupportRuntimeImpl.subscribe enriches agent_settled with collected sources.
        sources = e.sources ?? [];
      }
    });
    await rt.prompt(query);
    return { answer: answer.trim(), sources };
  } finally {
    rt.dispose();
  }
}

export async function runFaithfulnessEval(): Promise<AnswerCase[]> {
  const cases: AnswerCase[] = [];
  for (const c of GOLDEN_SET) {
    console.log(`[faithfulness] ${c.id}: running agent turn…`);
    const { answer, sources } = await runAgentTurn(c.query);
    console.log(`[faithfulness] ${c.id}: judging (answer ${answer.length} chars, ${sources.length} sources)…`);
    const verdict = await judgeAnswer(c.query, answer, sources);
    cases.push({
      id: c.id,
      query: c.query,
      source: c.source,
      answer,
      sources,
      verdict,
    });
  }
  return cases;
}

function printReport(cases: AnswerCase[]): void {
  console.log(`\n${"case".padEnd(22)} ${"src".padEnd(4)} ${"faith".padEnd(6)} verdict  answer`);
  console.log("-".repeat(90));
  for (const c of cases) {
    console.log(
      `${c.id.padEnd(22)} ${c.source.padEnd(4)} ${c.verdict.faithfulness.toFixed(0).padEnd(6)} ` +
        `${c.verdict.verdict.padEnd(7)}  ${c.answer.slice(0, 60).replace(/\n/g, " ")}`,
    );
  }
  const avg = cases.reduce((s, c) => s + c.verdict.faithfulness, 0) / (cases.length || 1);
  const passRate = cases.filter((c) => c.verdict.verdict === "pass").length / (cases.length || 1);
  console.log("-".repeat(90));
  console.log(`avg faithfulness: ${avg.toFixed(2)}/5 · pass rate: ${(passRate * 100).toFixed(0)}%\n`);
  console.log("Caveats:");
  console.log("  - faithfulness ≠ correctness: an honest \"not found\" passes even when retrieval failed");
  console.log("    (the Sony case: retrieval finds the Xperia manual, the answer said it's not covered).");
  console.log("  - clarifying-question answers pass (no fabrication) but aren't answers — router tuning.");
  console.log("  - sources sent to the judge are the agent_settled sources (sub-agent flow).\n");
  console.log("report: reports/faithfulness-report.json\n");
}

const isMain = process.argv[1] && import.meta.url.endsWith(process.argv[1].split("/").pop()!);
if (isMain) {
  runFaithfulnessEval()
    .then(async (cases) => {
      printReport(cases);
      await mkdir("reports", { recursive: true });
      await writeFile(
        "reports/faithfulness-report.json",
        JSON.stringify({ generatedAt: new Date().toISOString(), cases }, null, 2),
      );
    })
    .catch((err) => {
      console.error("[faithfulness] failed:", err.message);
      process.exit(1);
    });
}
