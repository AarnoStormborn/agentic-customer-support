/**
 * src/eval/run.ts — retrieval evaluation runner (npm run eval).
 *
 * For each golden case: run searchHybrid (kb or sql), map results to expected
 * id space, score recall@k / precision@k / MRR / hit, then print a table and
 * write eval-report.json.
 */
import "dotenv/config";
import { writeFile, mkdir } from "node:fs/promises";
import { searchHybrid, type HybridResult } from "../retrieval/index.js";
import { scoreQuery, average, type EvalScore } from "./metrics.js";
import { GOLDEN_SET, matchesRow } from "./golden.js";

interface CaseReport {
  id: string;
  query: string;
  source: "kb" | "sql";
  note: string;
  topK: number;
  retrieved: string[];
  score: EvalScore;
}

function resultId(result: HybridResult): string {
  if (result.source.type === "sql") {
    const row = result.source.row as { ticket_id?: unknown } | undefined;
    return `sql:${String(row?.ticket_id ?? "")}`;
  }
  return `kb:${result.source.docName ?? result.source.title ?? ""}`;
}

/** Does ONE result satisfy one expected item? (kb: docName contains; sql: row predicate) */
function matchesExpected(source: "kb" | "sql", expectedItem: string, result: HybridResult): boolean {
  if (source === "kb") {
    return (result.source.docName ?? "").includes(expectedItem);
  }
  return matchesRow(result, expectedItem);
}

/** Does ANY result satisfy an expected item? */
function foundInResults(source: "kb" | "sql", expectedItem: string, results: HybridResult[]): boolean {
  return results.some((r) => matchesExpected(source, expectedItem, r));
}

export interface EvalFilter {
  sloppy?: boolean; // undefined = all cases
  paraphrase?: boolean; // only (or none-of) paraphrase-heavy cases
}

export async function runEval(
  strategyInput?: Parameters<typeof searchHybrid>[0]["strategy"],
  filter: EvalFilter = {},
): Promise<CaseReport[]> {
  const reports: CaseReport[] = [];

  const cases = GOLDEN_SET.filter((c) => {
    if (filter.sloppy !== undefined && Boolean(c.sloppy) !== filter.sloppy) return false;
    if (filter.paraphrase === true && !c.paraphrase) return false;
    return true;
  });
  for (const c of cases) {
    const { results } = await searchHybrid({
      query: c.query,
      topK: c.topK,
      sourceTypes: [c.source],
      strategy: strategyInput,
    });
    const retrieved = results.map(resultId);

    // recall@k: fraction of expected items found anywhere in the top-k results
    const found = c.expected.map((exp) => foundInResults(c.source, exp, results));
    const recallAtK = found.filter(Boolean).length / c.expected.length;
    // precision@k: fraction of top-k results that match at least one expected item
    const matched = results.filter((r) => c.expected.some((exp) => matchesExpected(c.source, exp, r))).length;
    const precisionAtK = results.length > 0 ? matched / results.length : 0;
    // MRR: first rank at which any expected item appears
    const firstHit = results.findIndex((r) => c.expected.some((exp) => matchesExpected(c.source, exp, r)));
    const mrr = firstHit >= 0 ? 1 / (firstHit + 1) : 0;

    const score: EvalScore = {
      recallAtK,
      precisionAtK,
      mrr,
      hit: recallAtK > 0,
    };

    reports.push({
      id: c.id,
      query: c.query,
      source: c.source,
      note: c.note,
      topK: c.topK,
      retrieved,
      score,
    });
  }

  return reports;
}

function printReport(reports: CaseReport[]): void {
  console.log(`\n${"case".padEnd(22)} ${"src".padEnd(4)} ${"r@k".padEnd(6)} ${"p@k".padEnd(6)} ${"mrr".padEnd(6)} ${"hit".padEnd(5)} note`);
  console.log("-".repeat(90));
  for (const r of reports) {
    console.log(
      `${r.id.padEnd(22)} ${r.source.padEnd(4)} ${r.score.recallAtK.toFixed(2).padEnd(6)} ` +
        `${r.score.precisionAtK.toFixed(2).padEnd(6)} ${r.score.mrr.toFixed(2).padEnd(6)} ` +
        `${r.score.hit ? "✓" : "✗".padEnd(4)} ${r.note}`,
    );
  }
  const avg = {
    recall: average(reports.map((r) => r.score), "recallAtK"),
    precision: average(reports.map((r) => r.score), "precisionAtK"),
    mrr: average(reports.map((r) => r.score), "mrr"),
    hitRate: average(reports.map((r) => r.score), "hit"),
  };
  console.log("-".repeat(90));
  console.log(
    `avg                         ${avg.recall.toFixed(2).padEnd(6)} ${avg.precision.toFixed(2).padEnd(6)} ${avg.mrr.toFixed(2).padEnd(6)} ${(avg.hitRate * 100).toFixed(0)}%`,
  );
  console.log(`\nreport: eval-report.json\n`);
}

async function main(): Promise<void> {
  const flag = (name: string): string | undefined => {
    const i = process.argv.indexOf(name);
    return i >= 0 ? process.argv[i + 1] : undefined;
  };
  const strategyInput = flag("--strategy") ? { mode: flag("--strategy") } : undefined;
  const sloppyFlag = flag("--set");
  const filter =
    sloppyFlag === "sloppy" ? { sloppy: true }
    : sloppyFlag === "clean" ? { sloppy: false }
    : sloppyFlag === "paraphrase" ? { paraphrase: true }
    : {};
  const reports = await runEval(strategyInput as never, filter);
  printReport(reports);
  await mkdir("reports", { recursive: true });
  await writeFile(
    "reports/eval-report.json",
    JSON.stringify(
      {
        generatedAt: new Date().toISOString(),
        config: { embeddingMode: process.env.OPENAI_API_KEY ? "openai" : "hash" },
        cases: reports,
      },
      null,
      2,
    ),
  );
}

if (process.argv[1] && import.meta.url.endsWith(process.argv[1].split("/").pop()!)) {
  main().catch((err) => {
    console.error("[eval] failed:", err.message);
    process.exit(1);
  });
}
