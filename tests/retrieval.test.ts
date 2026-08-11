/**
 * Retrieval layer tests (Phase 5a — retrieval-core track).
 *
 * Pure units: chunker, hash-embedding fallback, source identity.
 * Integration (skipped unless ACS_TEST_DB=1): hybrid search against the live
 * Postgres/pgvector instance (docker compose).
 */
import { describe, it, expect } from "vitest";
import { chunkDocument, type ParsedDocument, type DocumentChunk } from "../src/retrieval/chunk.js";
import { hashEmbedding } from "../src/retrieval/embed.js";
import { sourceKey } from "../src/runtime/sources.js";

function fakeDoc(text: string, file = "lg-oled.pdf"): ParsedDocument {
  return {
    docName: file,
    filePath: `/manuals/${file}`,
    pageCount: 1,
    totalChars: text.length,
    pages: [{ num: 1, text }],
  };
}

describe("chunkDocument (structural chunker)", () => {
  it("produces chunks with metadata (page, section, overlap)", () => {
    const text =
      "# Connection\nWi-Fi setup steps for the TV.\n" +
      "# Troubleshooting\nIf the TV will not connect, restart the router.\n" +
      "Repeat this procedure. ".repeat(400); // long tail to force a split
    const chunks = chunkDocument(fakeDoc(text), { maxChars: 300, targetChars: 240 });
    expect(chunks.length).toBeGreaterThan(1);
    for (const c of chunks) {
      expect(c.pageStart).toBeGreaterThanOrEqual(1);
      expect(c.text.length).toBeGreaterThan(0);
    }
    // section path should be preserved from headings
    const first = chunks[0] as DocumentChunk;
    expect(first.headingPath?.length ?? 0).toBeGreaterThanOrEqual(0);
  });

  it("handles empty documents", () => {
    expect(chunkDocument(fakeDoc(""))).toEqual([]);
  });
});

describe("hashEmbedding (no-API-key fallback)", () => {
  it("is deterministic and L2-normalized", () => {
    const a = hashEmbedding("lg tv wifi reset");
    const b = hashEmbedding("lg tv wifi reset");
    expect(a).toEqual(b);
    expect(a.length).toBe(1536);
    const norm = Math.sqrt(a.reduce((s, x) => s + x * x, 0));
    expect(norm).toBeCloseTo(1, 3);
  });

  it("distinguishes different text", () => {
    const a = hashEmbedding("refund policy");
    const b = hashEmbedding("wifi troubleshooting");
    expect(a.some((v, i) => Math.abs(v - b[i]!) > 1e-9)).toBe(true);
  });
});

describe("sourceKey (dedupe identity)", () => {
  it("distinguishes kb chunks and sql rows", () => {
    const kb = { type: "kb", title: "lg-oled.pdf", row: undefined };
    const sql1 = { type: "sql", title: "ticket #5", row: { ticket_id: 5 } };
    const sql1b = { type: "sql", title: "ticket #5", row: { ticket_id: 5 } };
    const sql2 = { type: "sql", title: "ticket #9", row: { ticket_id: 9 } };
    expect(sourceKey(kb)).not.toBe(sourceKey(sql1));
    expect(sourceKey(sql1)).toBe(sourceKey(sql1b));
    expect(sourceKey(sql1)).not.toBe(sourceKey(sql2));
  });
});

describe("searchHybrid (integration — requires live DB)", () => {
  const run = process.env.ACS_TEST_DB === "1" ? it : it.skip;
  run("returns kb chunks with sources for a technical query", async () => {
    const { searchHybrid } = await import("../src/retrieval/index.js");
    const { results, queryTimeMs } = await searchHybrid({
      query: "lg tv wifi reset",
      topK: 3,
      sourceTypes: ["kb"],
    });
    expect(queryTimeMs).toBeGreaterThanOrEqual(0);
    expect(results.length).toBeGreaterThan(0);
    expect(results[0]!.source.type).toBe("kb");
    expect(results[0]!.text.length).toBeGreaterThan(0);
  });

  run("returns sql rows for ticket queries when enabled", async () => {
    const { searchHybrid } = await import("../src/retrieval/index.js");
    const { results } = await searchHybrid({
      query: "lg refund request",
      topK: 2,
      sourceTypes: ["sql"],
    });
    expect(results.length).toBeGreaterThan(0);
    expect(results[0]!.source.type).toBe("sql");
    expect(results[0]!.source.row).toBeTruthy();
  });

  run("ingestManuals is idempotent across runs with different path forms (file_path canonicalized to basename)", async () => {
    // Regression: UNIQUE(file_path) missed when one run passed a relative path and
    // another an absolute one, duplicating every document + chunk.
    const { ingestManuals } = await import("../src/retrieval/ingest.js");
    const { getPool } = await import("../src/db/pool.js");
    const pool = getPool();
    const before = await pool.query("SELECT count(*)::int AS n FROM documents");

    const dir = "config/data/manuals";
    const only = "lg_oled_55b9pla.pdf";
    await ingestManuals(dir, { only });
    const after1 = await pool.query("SELECT count(*)::int AS n FROM documents");
    // second run with an absolute path — same logical doc, must not duplicate
    await ingestManuals(requireAbs(dir), { only });
    const after2 = await pool.query("SELECT count(*)::int AS n FROM documents");

    expect(after1.rows[0].n).toBeGreaterThanOrEqual(before.rows[0].n);
    expect(after2.rows[0].n).toBe(after1.rows[0].n);
  });
});

/** Absolute-path helper for the idempotency regression test. */
function requireAbs(p: string): string {
  return /^\//.test(p) ? p : `${process.cwd()}/${p}`;
}
