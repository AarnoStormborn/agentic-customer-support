/**
 * Query relaxation tests (Phase 5b.8) — term dropping logic (unit) +
 * live FTS behavior (DB-guarded, ACS_TEST_DB=1).
 */
import { describe, it, expect } from "vitest";
import { queryTerms, tsQueryVariants, relaxedSearch } from "../src/retrieval/relax.js";

describe("queryTerms", () => {
  it("splits on whitespace and keeps quoted phrases intact", () => {
    expect(queryTerms('lg tv wifi reset')).toEqual(["lg", "tv", "wifi", "reset"]);
    expect(queryTerms('"credit card" complaint refund')).toEqual(['"credit card"', "complaint", "refund"]);
  });

  it("handles empty input", () => {
    expect(queryTerms("   ")).toEqual([]);
  });
});

describe("tsQueryVariants", () => {
  it("produces strictest-first variants down to a single term", () => {
    expect(tsQueryVariants("lg tv wifi")).toEqual(["lg tv wifi", "lg tv", "lg"]);
  });

  it("keeps quoted phrases together in every variant", () => {
    expect(tsQueryVariants('"lg oled" refund')).toEqual(['"lg oled" refund', '"lg oled"']);
  });

  it("returns [] for empty queries", () => {
    expect(tsQueryVariants("   ")).toEqual([]);
  });
});

describe("relaxedSearch", () => {
  it("returns the first non-empty variant and flags relaxation", async () => {
    const calls: string[] = [];
    const result = await relaxedSearch(["a b c", "a b", "a"], async (variant) => {
      calls.push(variant);
      return variant === "a b" ? [{ id: 1 }] : [];
    });
    expect(calls).toEqual(["a b c", "a b"]); // strict variant tried first, then relaxed
    expect(result).toMatchObject({ relaxed: true, attempts: 2, variant: "a b" });
    expect(result.rows).toEqual([{ id: 1 }]);
  });

  it("does not flag relaxation when the strictest variant matches", async () => {
    const result = await relaxedSearch(["a b", "a"], async () => [{ id: 1 }]);
    expect(result).toMatchObject({ relaxed: false, attempts: 1 });
  });

  it("returns empty rows when nothing matches", async () => {
    const result = await relaxedSearch(["zzz"], async () => []);
    expect(result.rows).toEqual([]);
  });
});

describe("live FTS relaxation (DB-guarded)", () => {
  const run = process.env.ACS_TEST_DB === "1" ? it : it.skip;

  run("an unmatched term no longer zeroes results — relaxation drops it", async () => {
    const { searchHybrid } = await import("../src/retrieval/index.js");
    // 'television' matches nothing in the tickets corpus ('LG Smart TV'), but the
    // other terms do. Strict websearch_to_tsquery would return 0 rows.
    const { results, relaxed } = await searchHybrid({
      query: "refund request lg oled television",
      topK: 5,
      sourceTypes: ["sql"],
    });
    expect(results.length).toBeGreaterThan(0);
    expect(relaxed).toBe(true);
  });

  run("the strict query still works when all terms match", async () => {
    const { searchHybrid } = await import("../src/retrieval/index.js");
    const { results, relaxed } = await searchHybrid({
      query: "refund request lg oled",
      topK: 5,
      sourceTypes: ["sql"],
    });
    expect(results.length).toBeGreaterThan(0);
    expect(relaxed).toBe(false);
  });
});
