/**
 * src/retrieval/query.ts — `npm run query -- "lg tv wifi reset"`
 * Runs searchHybrid against the live DB and prints results with sources.
 * This is the end-to-end verification CLI for the retrieval track.
 */
import "dotenv/config";
import { searchHybrid } from "./hybrid.js";

const isMain = process.argv[1] && import.meta.url === new URL(`file://${process.argv[1]}`).href;

if (isMain) {
  const query = process.argv.slice(2).join(" ") || "lg tv wifi reset";
  const topK = Number(process.env.QUERY_TOP_K ?? 5) || 5;

  (async () => {
    try {
      const { results, queryTimeMs } = await searchHybrid({ query, topK });
      console.log(`query: "${query}"  (topK=${topK}, ${queryTimeMs}ms, ${results.length} results)`);
      console.log("-".repeat(72));
      results.forEach((r, i) => {
        const src = r.source;
        const where =
          src.type === "kb"
            ? `${src.docName ?? ""}${src.sectionPath ? ` › ${src.sectionPath}` : ""}${src.page ? ` (p.${src.page})` : ""}`
            : `${src.title ?? ""}${src.row ? ` | ${String((src.row as { ticket_status?: string }).ticket_status ?? "")}` : ""}`;
        console.log(`${String(i + 1).padStart(2)}. [${src.type}] score=${r.score.toFixed(5)} ${where}`);
        console.log(`   ${r.text.slice(0, 220).replace(/\n/g, " ")}${r.text.length > 220 ? "…" : ""}`);
      });
      if (results.length === 0) console.log("no results — have you run `npm run ingest`?");
    } catch (err) {
      console.error("[query] fatal:", (err as Error).message);
      process.exitCode = 1;
    } finally {
      const { closePool } = await import("../db/pool.js");
      await closePool();
    }
  })();
}
