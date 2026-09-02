/**
 * Retrieval compare endpoint tests (Phase 5h) — mocked searchHybrid, no DB.
 */
import { describe, it, expect, vi, beforeAll, afterAll } from "vitest";
import type { FastifyInstance } from "fastify";
import { buildApp } from "../src/server/app.js";
import type { SupportRuntime } from "../src/runtime/index.js";

const fakeQueue = { add: async () => ({ id: "job-1" }) } as never;
const fakeRuntime: SupportRuntime = {
  async prompt() {},
  async steer() {},
  async abort() {},
  subscribe() {
    return () => {};
  },
  getLastMessages: () => [],
  dispose() {},
};

vi.mock("../src/retrieval/index.js", async (importOriginal) => {
  const mod = await importOriginal<typeof import("../src/retrieval/index.js")>();
  return {
    ...mod,
    searchHybrid: vi.fn(async (opts: { query: string; strategy?: { mode?: string } }) => ({
      results:
        opts.strategy?.mode === "keyword"
          ? []
          : [
              {
                text: "Settings > Network > Wi-Fi",
                source: { type: "kb", docName: "lg_oled_55b9pla.pdf", sectionPath: "4.2" },
                score: 0.9,
              },
            ],
      relaxed: opts.query.includes("television"),
      queryTimeMs: 12,
      strategy: { mode: opts.strategy?.mode ?? "hybrid" },
    })),
  };
});

import { searchHybrid } from "../src/retrieval/index.js";

describe("POST /api/retrieval/compare", () => {
  let app: FastifyInstance;

  beforeAll(async () => {
    app = await buildApp({
      createRuntime: async () => fakeRuntime,
      taskQueue: fakeQueue,
    });
    await app.ready();
  });

  afterAll(async () => {
    await app.close();
  });

  it("returns per-mode results with scores", async () => {
    const res = await app.inject({
      method: "POST",
      url: "/api/retrieval/compare",
      payload: { query: "reset lg tv wifi", modes: ["hybrid", "vector", "keyword"], topK: 2 },
    });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.modes).toHaveLength(3);
    expect(body.modes[0]).toMatchObject({ mode: "hybrid", relaxed: false, queryTimeMs: 12 });
    expect(body.modes[0].top[0]).toMatchObject({ docName: "lg_oled_55b9pla.pdf", score: 0.9 });
    // keyword returns empty in the mock
    expect(body.modes[2].top).toEqual([]);
  });

  it("validates query is required", async () => {
    const res = await app.inject({
      method: "POST",
      url: "/api/retrieval/compare",
      payload: { modes: ["hybrid"] },
    });
    expect(res.statusCode).toBe(400);
  });

  it("defaults to all modes when none requested", async () => {
    const res = await app.inject({
      method: "POST",
      url: "/api/retrieval/compare",
      payload: { query: "x", topK: 1 },
    });
    expect(res.statusCode).toBe(200);
    const { modes } = res.json();
    expect(modes.map((m: { mode: string }) => m.mode).sort()).toEqual(
      ["hybrid", "hyde", "hyde-hybrid", "keyword", "vector"].sort(),
    );
    expect(searchHybrid).toHaveBeenCalled();
  });

  it("POST /api/retrieval/search returns one strategy's results + relaxed flag", async () => {
    const res = await app.inject({
      method: "POST",
      url: "/api/retrieval/search",
      payload: { query: "my lg television wifi is broken", strategy: { mode: "hybrid" } },
    });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.relaxed).toBe(true);
    expect(body.results[0]).toMatchObject({ type: "kb", title: "lg_oled_55b9pla.pdf" });
  });
});
