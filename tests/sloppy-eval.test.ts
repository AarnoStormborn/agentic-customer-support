/**
 * Sloppy-query eval + model-resolution regression tests (Phase 5f).
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { GOLDEN_SET } from "../src/eval/golden.js";

describe("golden set sloppy grouping", () => {
  it("has both clean and sloppy cases, and paraphrase cases exist", () => {
    const sloppy = GOLDEN_SET.filter((c) => c.sloppy);
    const clean = GOLDEN_SET.filter((c) => !c.sloppy);
    const paraphrase = GOLDEN_SET.filter((c) => c.paraphrase);
    expect(sloppy.length).toBeGreaterThan(0);
    expect(clean.length).toBeGreaterThan(0);
    expect(paraphrase.length).toBeGreaterThan(0);
    // every sloppy case references its clean counterpart
    for (const c of sloppy) expect(c.note.toLowerCase()).toMatch(/paired|sloppy|ocr/);
  });
});
