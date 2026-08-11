/**
 * Queue tests (Phase 5b.2) — job payload validation + real handler adapters.
 * Retrieval/DB modules are mocked; the handlers' payload→call wiring is what's
 * under test (the ingest pipelines themselves are covered in tests/retrieval).
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const ingest = vi.hoisted(() => ({
  ingestTickets: vi.fn(),
  ingestManuals: vi.fn(),
}));
const pool = vi.hoisted(() => ({
  getPool: vi.fn(),
}));
const embed = vi.hoisted(() => ({
  embedTexts: vi.fn(),
  EMBEDDING_MODEL: "text-embedding-3-small",
}));

vi.mock("../src/retrieval/ingest.js", () => ingest);
vi.mock("../src/db/pool.js", () => pool);
vi.mock("../src/retrieval/embed.js", () => embed);

import { parseTaskType } from "../src/queue/jobs.js";
import {
  handleIngestDocument,
  handleIngestTickets,
  handleReembed,
} from "../src/queue/handlers.js";

describe("parseTaskType", () => {
  it("accepts the three known types and rejects everything else", () => {
    expect(parseTaskType("ingest.document")).toBe("ingest.document");
    expect(parseTaskType("ingest.tickets")).toBe("ingest.tickets");
    expect(parseTaskType("reembed")).toBe("reembed");
    expect(parseTaskType("drop.database")).toBeNull();
    expect(parseTaskType(42)).toBeNull();
  });
});

describe("handleIngestTickets", () => {
  beforeEach(() => ingest.ingestTickets.mockReset());

  it("delegates to ingestTickets with the source and returns a summary", async () => {
    ingest.ingestTickets.mockResolvedValue({
      source: "suraj520",
      rowsRead: 8469,
      rowsInserted: 8469,
      rowsUpdated: 0,
      failures: [],
      embeddingMode: "hash",
      dryRun: false,
    });
    const out = await handleIngestTickets({ source: "suraj520" });
    expect(ingest.ingestTickets).toHaveBeenCalledWith("suraj520", expect.any(Object));
    expect(out).toMatchObject({ handled: true, source: "suraj520", rowsRead: 8469 });
  });
});

describe("handleIngestDocument", () => {
  beforeEach(() => ingest.ingestManuals.mockReset());

  it("ingests a directory of PDFs as-is", async () => {
    const dir = await mkdtemp(join(tmpdir(), "acs-test-manuals-"));
    await writeFile(join(dir, "a.pdf"), "fake pdf bytes");
    ingest.ingestManuals.mockResolvedValue({
      source: "manuals",
      docs: 1,
      chunks: 5,
      failures: [],
      embeddingMode: "hash",
      dryRun: false,
      rowsRead: 0,
      rowsInserted: 0,
      rowsUpdated: 0,
    });

    const out = await handleIngestDocument({ path: dir });
    expect(ingest.ingestManuals).toHaveBeenCalledWith(dir);
    expect(out).toMatchObject({ handled: true, docs: 1, chunks: 5 });
  });

  it("ingests a single PDF via a temp dir symlink", async () => {
    const dir = await mkdtemp(join(tmpdir(), "acs-test-single-"));
    const pdf = join(dir, "manual.pdf");
    await writeFile(pdf, "fake pdf bytes");
    ingest.ingestManuals.mockResolvedValue({
      source: "manuals",
      docs: 1,
      chunks: 3,
      failures: [],
      embeddingMode: "hash",
      dryRun: false,
      rowsRead: 0,
      rowsInserted: 0,
      rowsUpdated: 0,
    });

    const out = await handleIngestDocument({ path: pdf, docName: "lg-oled.pdf" });
    // Temp dir should have contained the symlinked PDF
    const calledDir = ingest.ingestManuals.mock.calls[0]![0] as string;
    expect(out.handled).toBe(true);
    expect(ingest.ingestManuals).toHaveBeenCalledTimes(1);
    // cleanup removed the temp dir
    await expect(import("node:fs/promises").then((f) => f.access(calledDir))).rejects.toThrow();
  });
});

describe("handleReembed", () => {
  beforeEach(() => {
    pool.getPool.mockReset();
    embed.embedTexts.mockReset();
  });

  it("re-embeds a slice of chunks in batches", async () => {
    const query = vi.fn(async () => ({
      rows: [
        { chunk_id: 1, chunk_text: "a" },
        { chunk_id: 2, chunk_text: "b" },
      ],
    }));
    pool.getPool.mockReturnValue({ query });
    embed.embedTexts.mockResolvedValue([
      Array(1536).fill(0.1),
      Array(1536).fill(0.2),
    ]);

    const out = await handleReembed({ limit: 2 });
    expect(embed.embedTexts).toHaveBeenCalledWith(["a", "b"]);
    expect(query).toHaveBeenCalledWith(expect.stringContaining("UPDATE document_chunks"), [
      expect.any(Array),
      1,
    ]);
    expect(out).toMatchObject({ handled: true, reembedded: 2 });
  });
});
