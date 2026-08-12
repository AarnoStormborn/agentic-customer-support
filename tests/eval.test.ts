/**
 * Eval harness tests — deterministic metric math + golden-set wiring.
 * The live retrieval run is covered by `npm run eval` (DB required).
 */
import { describe, it, expect } from "vitest";
import {
  recallAtK,
  precisionAtK,
  reciprocalRank,
  hitRate,
  scoreQuery,
  average,
} from "../src/eval/metrics.js";
import { GOLDEN_SET, matchesRow } from "../src/eval/golden.js";

describe("metrics", () => {
  const expected = new Set(["a", "c"]);

  it("recall@k counts expected sources in the top-k", () => {
    expect(recallAtK(expected, ["a", "b", "c", "d"], 5)).toBe(1);
    expect(recallAtK(expected, ["a", "b", "d"], 5)).toBe(0.5);
    expect(recallAtK(expected, ["x", "y"], 5)).toBe(0);
  });

  it("precision@k measures top-k purity", () => {
    expect(precisionAtK(expected, ["a", "b"], 2)).toBe(0.5);
    expect(precisionAtK(expected, ["a", "c"], 2)).toBe(1);
    expect(precisionAtK(expected, [], 5)).toBe(0);
  });

  it("MRR rewards the first expected source's rank", () => {
    expect(reciprocalRank(expected, ["x", "a", "c"])).toBeCloseTo(1 / 2);
    expect(reciprocalRank(expected, ["c"])).toBe(1);
    expect(reciprocalRank(expected, ["z"])).toBe(0);
  });

  it("hitRate + scoreQuery compose correctly", () => {
    expect(hitRate(expected, ["a"])).toBe(true);
    const s = scoreQuery(expected, ["x", "a"], 5);
    expect(s).toMatchObject({ recallAtK: 0.5, mrr: 0.5, hit: true });
  });

  it("average works over scores", () => {
    const scores = [
      { recallAtK: 1, precisionAtK: 0.5, mrr: 1, hit: true },
      { recallAtK: 0, precisionAtK: 0, mrr: 0, hit: false },
    ];
    expect(average(scores, "recallAtK")).toBe(0.5);
    expect(average(scores, "hit")).toBe(0.5);
  });
});

describe("golden set", () => {
  it("has valid cases (kb expects doc names, sql expects predicates)", () => {
    for (const c of GOLDEN_SET) {
      expect(c.query.length).toBeGreaterThan(3);
      expect(c.topK).toBeGreaterThan(0);
      expect(c.expected.length).toBeGreaterThan(0);
      if (c.source === "sql") {
        expect(c.expected.every((p) => /^\w+\s+ILIKE\s+'%[^']*%'$/i.test(p))).toBe(true);
      }
    }
  });

  it("matchesRow evaluates sql predicates", () => {
    const result = {
      text: "",
      score: 1,
      source: {
        type: "sql" as const,
        row: { ticket_id: 5, product_purchased: "LG OLED TV 65C4", ticket_type: "Refund request" },
      },
    };
    expect(matchesRow(result, "product_purchased ILIKE '%lg%'")).toBe(true);
    expect(matchesRow(result, "ticket_type ILIKE '%refund%'")).toBe(true);
    expect(matchesRow(result, "product_purchased ILIKE '%sony%'")).toBe(false);
  });
});
