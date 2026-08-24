/**
 * Sloppy-query eval + model-resolution regression tests (Phase 5f).
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { GOLDEN_SET } from "../src/eval/golden.js";

describe("golden set sloppy grouping", () => {
  it("has both clean and sloppy cases, and clean cases outnumber sloppy", () => {
    const sloppy = GOLDEN_SET.filter((c) => c.sloppy);
    const clean = GOLDEN_SET.filter((c) => !c.sloppy);
    expect(sloppy.length).toBeGreaterThan(0);
    expect(clean.length).toBeGreaterThan(sloppy.length);
    // every sloppy case has a clean counterpart style (not required, but sanity)
    for (const c of sloppy) expect(c.note).toContain("paired");
  });
});
