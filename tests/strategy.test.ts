/**
 * Retrieval strategy tests (Phase 5c) — normalization, expansion, rerank gating,
 * strategy-bound kb tool.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { normalizeStrategy, DEFAULT_STRATEGY, RETRIEVAL_MODES } from "../src/retrieval/strategy.js";
import { expandQuery } from "../src/retrieval/expand.js";
import { rerank, rerankEnabled } from "../src/retrieval/rerank.js";

describe("normalizeStrategy", () => {
  it("defaults reproduce the current hybrid behavior", () => {
    expect(normalizeStrategy(undefined)).toEqual(DEFAULT_STRATEGY);
    expect(normalizeStrategy(null)).toEqual(DEFAULT_STRATEGY);
    expect(normalizeStrategy({})).toEqual(DEFAULT_STRATEGY);
  });

  it("merges provided knobs and clamps out-of-range values", () => {
    const s = normalizeStrategy({ mode: "hyde", topK: 99, rrfK: 5, multiQuery: true });
    expect(s.mode).toBe("hyde");
    expect(s.topK).toBe(10);
    expect(s.rrfK).toBe(10);
    expect(s.multiQuery).toBe(true);
    expect(s.relax).toBe(true); // untouched default
  });

  it("ignores invalid modes and non-boolean flags", () => {
    const s = normalizeStrategy({ mode: "bogus", relax: "yes" as never });
    expect(s.mode).toBe(DEFAULT_STRATEGY.mode);
    expect(s.relax).toBe(true);
  });

  it("exposes all supported modes", () => {
    expect(RETRIEVAL_MODES).toContain("hybrid");
    expect(RETRIEVAL_MODES).toContain("hyde");
  });
});

describe("expandQuery", () => {
  it("appends synonyms for known terms", () => {
    const out = expandQuery("lg tv wifi reset");
    expect(out).toContain("television"); // tv → television
    expect(out).toContain("wi-fi"); // wifi → wi-fi
    expect(out).toContain("restart"); // reset → restart
    expect(out.startsWith("lg tv wifi reset")).toBe(true); // original preserved
  });

  it("caps extra terms and leaves unknown words alone", () => {
    const out = expandQuery("foo bar baz qux");
    expect(out).toBe("foo bar baz qux");
  });
});

describe("rerank", () => {
  beforeEach(() => {
    delete process.env.COHERE_API_KEY;
    vi.unstubAllGlobals();
  });
  afterEach(() => vi.unstubAllGlobals());

  it("is a passthrough without a Cohere key", async () => {
    expect(rerankEnabled()).toBe(false);
    const items = [{ id: "a", text: "x" }, { id: "b", text: "y" }];
    await expect(rerank("q", items)).resolves.toEqual(items);
  });

  it("reorders by relevance score when a key is configured", async () => {
    process.env.COHERE_API_KEY = "test-key";
    const fetchMock = vi.fn(async () =>
      new Response(
        JSON.stringify({ results: [{ index: 1, relevance_score: 0.9 }, { index: 0, relevance_score: 0.4 }] }),
        { status: 200, headers: { "content-type": "application/json" } },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const items = [{ id: "a", text: "first" }, { id: "b", text: "second" }];
    const out = await rerank("q", items);
    expect(out.map((i) => i.id)).toEqual(["b", "a"]);

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain("cohere.com");
    expect((init.headers as Record<string, string>).authorization).toContain("test-key");
  });

  it("falls back to original order on API errors", async () => {
    process.env.COHERE_API_KEY = "test-key";
    vi.stubGlobal("fetch", vi.fn(async () => new Response("boom", { status: 500 })));
    const items = [{ id: "a", text: "x" }, { id: "b", text: "y" }];
    const out = await rerank("q", items);
    expect(out.map((i) => i.id)).toEqual(["a", "b"]);
  });
});
