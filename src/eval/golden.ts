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
  /** Sloppy (real-user-style) query — typos/filler/grammar. multiQuery/expansion
   *  are expected to help most on these; the comparison is run separately. */
  sloppy?: boolean;
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
    expected: ["Kenmore", "Refrigerator"],
    topK: 5,
    note: "Kenmore fridge use & care guide",
  },
  // --- tickets (SQL) ---
  {
    id: "sql-lg-refund",
    query: "refund request lg oled television",
    source: "sql",
    expected: ["product_purchased ILIKE '%lg%'", "ticket_type ILIKE '%refund%'"],
    topK: 5,
    note: "suraj520 LG refunds — 'television' is unmatched (tickets say 'LG Smart TV'); FTS relaxation drops it and still finds results",
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
    id: "kb-kenmore-washer",
    query: "kenmore washer spin cycle not draining",
    source: "kb",
    expected: ["Kenmore", "Washer"],
    topK: 5,
    note: "Kenmore washer manual — drain/spin",
  },
  {
    id: "kb-dishwasher",
    query: "kenmore dishwasher not cleaning dishes properly",
    source: "kb",
    expected: ["Dishwasher"],
    topK: 5,
    note: "Kenmore dishwasher manual",
  },
  {
    id: "kb-clothes-dryer",
    query: "clothes dryer not heating up",
    source: "kb",
    expected: ["Dryer"],
    topK: 5,
    note: "Kenmore clothes dryer manual",
  },
  {
    id: "kb-range",
    query: "electric range oven temperature calibration",
    source: "kb",
    expected: ["Range"],
    topK: 5,
    note: "Kenmore range manual",
  },
  {
    id: "kb-laptop",
    query: "hp pavilion battery replacement",
    source: "kb",
    expected: ["hp_pavilion"],
    topK: 5,
    note: "HP Pavilion notebook guide",
  },
  {
    id: "kb-refrigerator-water",
    query: "refrigerator water dispenser not working",
    source: "kb",
    expected: ["Refrigerator"],
    topK: 5,
    note: "Kenmore/LG fridge manual",
  },
  {
    id: "sql-tv-technical",
    query: "smart tv technical issue",
    source: "sql",
    expected: ["ticket_type ILIKE '%technical%'", "product_purchased ILIKE '%tv%'"],
    topK: 5,
    note: "suraj520 technical issues on TVs",
  },
  // --- sloppy (real-user-style) queries — multiQuery/expansion should shine ---
  {
    id: "kb-lg-wifi-sloppy",
    query: "my lg tv keeps dropin the wifi connection plz help reset",
    source: "kb",
    expected: ["lg_oled_55b9pla.pdf"],
    topK: 5,
    note: "typos (dropin) + filler (plz) — paired with kb-lg-wifi",
    sloppy: true,
  },
  {
    id: "kb-kenmore-ice-sloppy",
    query: "kenmore frig not makin ice cubes any more",
    source: "kb",
    expected: ["Kenmore", "Refrigerator"],
    topK: 5,
    note: "colloquial + misspellings — paired with kb-kenmore-ice",
    sloppy: true,
  },
  {
    id: "kb-dryer-sloppy",
    query: "cloth dryer dont heat up no more",
    source: "kb",
    expected: ["Dryer"],
    topK: 5,
    note: "double-negative colloquial — paired with kb-clothes-dryer",
    sloppy: true,
  },
  {
    id: "kb-dishwasher-sloppy",
    query: "dishwasher leaving food on plates after cycle wat do",
    source: "kb",
    expected: ["Dishwasher"],
    topK: 5,
    note: "informal + shortened — paired with kb-dishwasher",
    sloppy: true,
  },
  {
    id: "kb-washer-sloppy",
    query: "washing machine spin cycle stuck not draining water",
    source: "kb",
    expected: ["Kenmore", "Washer"],
    topK: 5,
    note: "multi-word noun + state — paired with kb-kenmore-washer",
    sloppy: true,
  },
  {
    id: "kb-laptop-sloppy",
    query: "hp pavilion batterie aint holding charge",
    source: "kb",
    expected: ["hp_pavilion"],
    topK: 5,
    note: "misspelled battery — paired with kb-laptop",
    sloppy: true,
  },
  {
    id: "kb-fridge-water-sloppy",
    query: "fridge water dispenser brok help",
    source: "kb",
    expected: ["Refrigerator"],
    topK: 5,
    note: "laconic + slang — paired with kb-refrigerator-water",
    sloppy: true,
  },
  {
    id: "kb-sony-sim-sloppy",
    query: "how put sim in my sony fone",
    source: "kb",
    expected: ["sony_xperia_1v_manual.pdf"],
    topK: 5,
    note: "phonetic spelling (fone) — paired with kb-sony-sim",
    sloppy: true,
  },
  {
    id: "kb-range-sloppy",
    query: "oven temp off need calibrate my range",
    source: "kb",
    expected: ["Range"],
    topK: 5,
    note: "fragment + missing words — paired with kb-range",
    sloppy: true,
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
