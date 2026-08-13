/**
 * Judge unit tests — pure prompt/parse logic (no model calls).
 */
import { describe, it, expect } from "vitest";
import { buildJudgePrompt, parseVerdict, FAITHFULNESS_THRESHOLD } from "../src/eval/judge.js";

describe("parseVerdict", () => {
  it("parses clean JSON output", () => {
    const v = parseVerdict('{"faithfulness": 4, "rationale": "supported", "verdict": "pass"}');
    expect(v).toEqual({ faithfulness: 4, rationale: "supported", verdict: "pass" });
  });

  it("tolerates markdown code fences and trailing text", () => {
    const v = parseVerdict('```json\n{"faithfulness": 2, "rationale": "hallucinated", "verdict": "fail"}\n```');
    expect(v).toMatchObject({ faithfulness: 2, verdict: "fail" });
  });

  it("clamps out-of-range scores and normalizes unknown verdicts to fail", () => {
    expect(parseVerdict('{"faithfulness": 9, "verdict": "pass"}').faithfulness).toBe(5);
    expect(parseVerdict('{"faithfulness": 0, "verdict": "pass"}').faithfulness).toBe(1);
    expect(parseVerdict('{"faithfulness": 3, "verdict": "maybe"}').verdict).toBe("fail");
  });

  it("throws on non-JSON output", () => {
    expect(() => parseVerdict("I think it was fine.")).toThrow();
  });
});

describe("buildJudgePrompt", () => {
  it("includes the question, answer, and sources", () => {
    const p = buildJudgePrompt("q", "a", [{ type: "kb", title: "lg.pdf" }]);
    expect(p).toContain("q");
    expect(p).toContain("a");
    expect(p).toContain("lg.pdf");
    expect(p).toContain(String(FAITHFULNESS_THRESHOLD));
  });
});
