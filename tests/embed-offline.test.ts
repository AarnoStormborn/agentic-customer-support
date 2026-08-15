/**
 * Offline embedding tests (Phase 5d) — backend selection + Ollama HTTP client.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { embeddingBackend, embeddingDim } from "../src/retrieval/embed.js";
import { embedViaOllama } from "../src/retrieval/embed-ollama.js";

beforeEach(() => {
  delete process.env.OPENAI_API_KEY;
  delete process.env.EMBEDDING_BACKEND;
  delete process.env.EMBEDDING_DIM;
  vi.unstubAllGlobals();
});

afterEach(() => vi.unstubAllGlobals());

describe("embeddingBackend selection", () => {
  it("prefers openai when a key is set", () => {
    process.env.OPENAI_API_KEY = "sk-test";
    expect(embeddingBackend()).toBe("openai");
  });

  it("defaults to local ollama without a key", () => {
    expect(embeddingBackend()).toBe("ollama");
  });

  it("honors an explicit override", () => {
    process.env.EMBEDDING_BACKEND = "hash";
    expect(embeddingBackend()).toBe("hash");
    process.env.EMBEDDING_BACKEND = "openai";
    expect(embeddingBackend()).toBe("openai");
  });
});

describe("embeddingDim", () => {
  it("returns the ollama dim (768) for the local backend", () => {
    expect(embeddingDim()).toBe(768);
  });

  it("returns 1536 for hash", () => {
    process.env.EMBEDDING_BACKEND = "hash";
    expect(embeddingDim()).toBe(1536);
  });

  it("EMBEDDING_DIM overrides everything", () => {
    process.env.EMBEDDING_DIM = "512";
    expect(embeddingDim()).toBe(512);
  });
});

describe("embedViaOllama (HTTP client)", () => {
  it("posts a batch and returns embeddings in order", async () => {
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify({ embeddings: [[0.1, 0.2], [0.3, 0.4]] }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const out = await embedViaOllama(["a", "b"], "nomic-embed-text");
    expect(out).toEqual([[0.1, 0.2], [0.3, 0.4]]);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain("11434");
    expect(String(init.body)).toContain("nomic-embed-text");
  });

  it("throws on non-200 responses", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("boom", { status: 500 })));
    await expect(embedViaOllama(["a"])).rejects.toThrow(/HTTP 500/);
  });

  it("throws when the embedding count mismatches the input", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(JSON.stringify({ embeddings: [[0.1]] }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      ),
    );
    await expect(embedViaOllama(["a", "b"])).rejects.toThrow(/embeddings for 2 inputs/);
  });
});
