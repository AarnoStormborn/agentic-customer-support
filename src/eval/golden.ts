/**
 * src/eval/golden.ts — golden question set for retrieval evaluation.
 *
 * Each case pins a query to expected sources, hand-curated against the actual
 * provisioned data (suraj520 tickets + 3 manuals + CFPB complaints).
 *
 * - kb cases: expected = document names that MUST appear in the top-k chunk results
 * - sql cases: expected = predicates a returned row must satisfy
 */
import type { HybridResult } from "../retrieval/index.js";

export interface GoldenCase {
  id: string;
  query: string;
  source: "kb" | "sql";
  /** kb: expected docNames; sql: row predicates (checked via matchesRow) */
  expected: string[];
  topK: number;
  note: string;
}

export const GOLDEN_SET: GoldenCase[] = [
  // --- knowledge base (manuals) ---
  {
    id: "kb-lg-wifi",
    query: "how do i reset the wifi on my lg tv",
    source: "kb",
    expected: ["lg_oled_55b9pla.pdf"],
    topK: 5,
    note: "LG OLED manual — network/connection section",
  },
  {
    id: "kb-sony-sim",
    query: "insert sim card sony xperia",
    source: "kb",
    expected: ["sony_xperia_1v_manual.pdf"],
    topK: 5,
    note: "Sony Xperia 1 V manual — SIM card section",
  },
  {
    id: "kb-kenmore-ice",
    query: "kenmore refrigerator ice maker not working",
    source: "kb",
    expected: ["kenmore_fridge_25331115308.pdf"],
    topK: 5,
    note: "Kenmore fridge use & care guide",
  },
  // --- tickets (SQL) ---
  {
    id: "sql-lg-refund",
    query: "refund request lg oled",
    source: "sql",
    expected: ["product_purchased ILIKE '%lg%'", "ticket_type ILIKE '%refund%'"],
    topK: 5,
    note: "suraj520: LG product + refund request type (FTS ANDs terms — unmatched words return 0; keep golden queries realistic)",
  },
  {
    id: "sql-credit-card",
    query: "credit card billing dispute complaint",
    source: "sql",
    expected: ["product_purchased ILIKE '%credit%'"],
    topK: 5,
    note: "CFPB: credit card complaints dominate",
  },
  {
    id: "sql-tv-technical",
    query: "smart tv technical issue",
    source: "sql",
    expected: ["ticket_type ILIKE '%technical%'", "product_purchased ILIKE '%tv%'"],
    topK: 5,
    note: "suraj520 technical issues on TVs",
  },
];

/** Does a hybrid result row satisfy an expected predicate? */
export function matchesRow(result: HybridResult, predicate: string): boolean {
  const row = result.source.row as Record<string, unknown> | undefined;
  if (!row) return false;
  const [col, value] = splitPredicate(predicate);
  if (!col) return false;
  const cell = row[col];
  if (typeof cell !== "string") return false;
  const pattern = value.replace(/%/g, "");
  return cell.toLowerCase().includes(pattern.toLowerCase());
}

function splitPredicate(predicate: string): [string, string] {
  const m = predicate.match(/^(\w+)\s+ILIKE\s+'%([^']*)%'$/i);
  return m ? [m[1]!, m[2]!] : ["", ""];
}
