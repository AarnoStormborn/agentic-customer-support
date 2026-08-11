/**
 * web_search — live web search tool (the web specialist's one tool).
 *
 * Primary: Tavily Search API (TAVILY_API_KEY). Fallback behind a flag:
 * WEB_SEARCH_ENGINE=duckduckgo (or automatically when TAVILY_API_KEY is unset)
 * uses DuckDuckGo Lite HTML scraping via plain fetch — no extra dependency.
 */

import { defineTool } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { MAX_WEB_QUERY_LEN, TOOL_NAMES, WEB_SEARCH_TIMEOUT_MS } from "../config/limits.js";

export interface WebResult {
  title: string;
  url: string;
  snippet: string;
}

function combineSignals(parent?: AbortSignal, ms = WEB_SEARCH_TIMEOUT_MS): AbortSignal {
  if (!parent) return AbortSignal.timeout(ms);
  const ctrl = new AbortController();
  const onAbort = () => ctrl.abort(parent.reason);
  parent.addEventListener("abort", onAbort, { once: true });
  const timer = setTimeout(() => ctrl.abort(new Error("web search timeout")), ms);
  const originalAbort = ctrl.abort.bind(ctrl);
  ctrl.abort = (reason) => {
    clearTimeout(timer);
    originalAbort(reason);
  };
  void onAbort;
  return ctrl.signal;
}

async function tavilySearch(query: string, maxResults: number, signal?: AbortSignal): Promise<WebResult[]> {
  const apiKey = process.env.TAVILY_API_KEY;
  if (!apiKey) throw new Error("TAVILY_API_KEY is not set");
  const res = await fetch("https://api.tavily.com/search", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      api_key: apiKey,
      query,
      search_depth: "basic",
      max_results: maxResults,
      include_answer: false,
    }),
    signal,
  });
  if (!res.ok) throw new Error(`Tavily API error: HTTP ${res.status}`);
  const data = (await res.json()) as {
    results?: { title?: string; url?: string; content?: string }[];
  };
  return (data.results ?? []).map((r) => ({
    title: r.title ?? "",
    url: r.url ?? "",
    snippet: r.content ?? "",
  }));
}

/** DuckDuckGo Lite HTML scraping (no API key needed). */
export async function duckduckgoSearch(query: string, maxResults: number, signal?: AbortSignal): Promise<WebResult[]> {
  const url = new URL("https://lite.duckduckgo.com/lite/");
  url.searchParams.set("q", query);
  const res = await fetch(url.toString(), {
    headers: { "user-agent": "agentic-customer-support/2.0" },
    signal,
  });
  if (!res.ok) throw new Error(`DuckDuckGo error: HTTP ${res.status}`);
  const html = await res.text();

  // lite.duckduckgo.com renders result rows as <a class="result-link" href="...">title</a>
  // followed by <td class="result-snippet">text</td>. Quote style varies (' vs "), so
  // match both. Parse with regex (learner-grade, fine for fallback).
  const results: WebResult[] = [];
  const tagRe = /<a[^>]*class=["']result-link["'][^>]*>([\s\S]*?)<\/a>/gi;
  const hrefRe = /href="([^"]+)"/;
  const snippetRe = /<td class=["']result-snippet["']>([\s\S]*?)<\/td>/gi;
  const links: { url: string; title: string }[] = [];
  let m: RegExpExecArray | null;
  while ((m = tagRe.exec(html)) && links.length < maxResults) {
    const rawHref = m[0].match(hrefRe)?.[1];
    if (!rawHref) continue;
    // DDG redirect URLs wrap the real URL in ?uddg=<encoded>
    let url = rawHref;
    const uddg = rawHref.match(/[?&]uddg=([^&]+)/);
    if (uddg?.[1]) {
      try {
        url = decodeURIComponent(uddg[1]);
      } catch {
        url = uddg[1];
      }
    }
    const title = (m[1] ?? "").replace(/<[^>]+>/g, "").trim();
    // skip DDG sponsored/redirect helpers (y.js ad links, help pages)
    if (/duckduckgo\.com\/(y\.js|l\/)|duckduckgo-help-pages/i.test(url)) continue;
    if (url && title && url.startsWith("http")) links.push({ url, title });
  }
  const snippets: string[] = [];
  while ((m = snippetRe.exec(html)) && snippets.length < links.length) {
    snippets.push((m[1] ?? "").replace(/<[^>]+>/g, "").trim());
  }
  links.forEach((l, i) => {
    results.push({ title: l.title, url: l.url, snippet: snippets[i] ?? "" });
  });
  return results;
}

export const webSearchTool = defineTool({
  name: TOOL_NAMES.webSearch,
  label: "Web Search",
  description:
    "Search the live web (Tavily; DuckDuckGo fallback) for current information: pricing, promotions, " +
    "service outages, news, third-party details. Returns top results with title, URL and snippet.",
  parameters: Type.Object({
    query: Type.String({ description: "The search query, e.g. 'LG OLED TV 65C4 price 2026'" }),
    maxResults: Type.Optional(Type.Number({ description: "Max results (1-10, default 5)" })),
  }),
  execute: async (_toolCallId, params, signal) => {
    if (params.query.length > MAX_WEB_QUERY_LEN) {
      throw new Error(`web_search blocked: query too long (${params.query.length} chars)`);
    }
    const max = Math.min(Math.max(params.maxResults ?? 5, 1), 10);
    const engine = process.env.WEB_SEARCH_ENGINE ?? (process.env.TAVILY_API_KEY ? "tavily" : "duckduckgo");
    const combined = combineSignals(signal);

    let results: WebResult[];
    let used: string;
    try {
      if (engine === "tavily") {
        results = await tavilySearch(params.query, max, combined);
        used = "tavily";
      } else {
        results = await duckduckgoSearch(params.query, max, combined);
        used = "duckduckgo";
      }
    } finally {
      // AbortSignal.timeout already clears itself; nothing to clean for combined.
      void combined;
    }

    const text = results.length === 0
      ? "Web search returned no results."
      : results.map((r, i) => `[${i + 1}] ${r.title}\n    ${r.url}\n    ${r.snippet}`).join("\n\n");

    return {
      content: [{ type: "text", text }],
      details: {
        tool: TOOL_NAMES.webSearch,
        query: params.query,
        count: results.length,
        engine: used,
        sources: results.map((r) => ({ title: r.title, url: r.url })),
      },
    };
  },
});
