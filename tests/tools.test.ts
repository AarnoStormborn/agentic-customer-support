/**
 * Tool tests — rag (kb_search) + web (web_search) with mocked dependencies.
 * The SQL tool's allowlist is covered separately in sql-tool.test.ts.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// Mock the retrieval module BEFORE importing rag-tool so kb_search hits the mock.
vi.mock("../src/retrieval/index.js", () => ({
  searchHybrid: vi.fn(),
  embedTexts: vi.fn(),
}));

import { searchHybrid } from "../src/retrieval/index.js";
import { kbSearchTool } from "../src/tools/rag-tool.js";
import { webSearchTool } from "../src/tools/web-tool.js";

const mockSearch = vi.mocked(searchHybrid);

beforeEach(() => {
  mockSearch.mockReset();
  process.env.WEB_SEARCH_ENGINE = "duckduckgo";
  delete process.env.TAVILY_API_KEY;
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("kb_search tool", () => {
  it("calls searchHybrid with sourceTypes ['kb'] and returns sources in details", async () => {
    mockSearch.mockResolvedValue({
      results: [
        {
          text: "Press Settings > Network > Wi-Fi",
          source: { type: "kb", docName: "lg-oled.pdf", sectionPath: "4.2", page: 17, url: null },
          score: 0.9,
        },
      ],
      queryTimeMs: 12,
    });

    const out = await kbSearchTool.execute("call_1", { query: "lg tv wifi reset" }, undefined as any);
    expect(mockSearch).toHaveBeenCalledWith({
      query: "lg tv wifi reset",
      topK: 5,
      sourceTypes: ["kb"],
    });
    expect(out.details.sources[0]!.title).toBe("lg-oled.pdf");
    expect(out.details.sources[0]!.score).toBe(0.9);
    expect(out.content[0]!.text).toContain("Press Settings");
  });

  it("clamps topK to 1..10", async () => {
    mockSearch.mockResolvedValue({ results: [], queryTimeMs: 1 });
    await kbSearchTool.execute("call_1", { query: "x", topK: 99 }, undefined as any);
    expect(mockSearch).toHaveBeenCalledWith(expect.objectContaining({ topK: 10 }));
  });
});

describe("web_search tool", () => {
  it("scrapes DuckDuckGo Lite via fetch when no Tavily key", async () => {
    const html = `<div><a class="result-link" href="https://example.com/wifi">Wi-Fi reset guide</a></div>`;
    vi.stubGlobal("fetch", vi.fn(async () => new Response(html, { status: 200 })));

    const out = await webSearchTool.execute("call_2", { query: "lg tv wifi reset" }, undefined as any);
    expect(out.details.sources[0]).toMatchObject({ title: "Wi-Fi reset guide", url: "https://example.com/wifi" });
  });

  it("uses Tavily when a key is configured", async () => {
    delete process.env.WEB_SEARCH_ENGINE;
    process.env.TAVILY_API_KEY = "tvly-test";
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify({ results: [{ title: "T", url: "https://t.example", content: "c" }] }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const out = await webSearchTool.execute("call_3", { query: "lg tv wifi" }, undefined as any);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain("tavily.com");
    expect(init.body).toContain("tvly-test");
    expect(out.details.sources[0]!.url).toBe("https://t.example");
  });
});
