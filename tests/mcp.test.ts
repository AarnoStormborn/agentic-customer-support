/**
 * MCP handler tests (Phase 5b.4) — real handlers with mocked retrieval/DB.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

const retrieval = vi.hoisted(() => ({ searchHybrid: vi.fn() }));
const db = vi.hoisted(() => ({ getPool: vi.fn() }));
const tools = vi.hoisted(() => ({ validateSelectQuery: vi.fn() }));

vi.mock("../src/retrieval/index.js", () => retrieval);
vi.mock("../src/db/pool.js", () => db);
vi.mock("../src/tools/sql-tool.js", () => tools);

import { kbSearchHandler, ticketsQueryHandler } from "../src/mcp/handlers.js";

describe("kbSearchHandler", () => {
  beforeEach(() => retrieval.searchHybrid.mockReset());

  it("calls searchHybrid for kb sources and formats results with sources", async () => {
    retrieval.searchHybrid.mockResolvedValue({
      results: [
        {
          text: "Press Settings > Network > Wi-Fi",
          source: { type: "kb", docName: "lg-oled.pdf", sectionPath: "4.2", page: 17, url: null },
          score: 0.91,
        },
      ],
      queryTimeMs: 12,
    });

    const out = await kbSearchHandler("lg tv wifi reset", 3);
    expect(retrieval.searchHybrid).toHaveBeenCalledWith({
      query: "lg tv wifi reset",
      topK: 3,
      sourceTypes: ["kb"],
    });
    expect(out.content[0]!.text).toContain("lg-oled.pdf");
    expect(out.content[0]!.text).toContain("Press Settings > Network > Wi-Fi");
    expect(out.content[0]!.text).toContain("score=0.9100");
  });

  it("defaults topK to 5 and handles empty results", async () => {
    retrieval.searchHybrid.mockResolvedValue({ results: [], queryTimeMs: 1 });
    const out = await kbSearchHandler("nothing matches");
    expect(retrieval.searchHybrid).toHaveBeenCalledWith(expect.objectContaining({ topK: 5 }));
    expect(out.content[0]!.text).toBe("No knowledge base results.");
  });
});

describe("ticketsQueryHandler", () => {
  beforeEach(() => {
    db.getPool.mockReset();
    tools.validateSelectQuery.mockReset();
  });

  it("builds, validates, and executes a SELECT in a read-only transaction", async () => {
    const client = {
      query: vi.fn(async () => ({ rows: [{ ticket_id: 42, product_purchased: "LG OLED", ticket_type: "Technical issue", complaint_narrative: "wifi keeps dropping" }] })),
      release: vi.fn(),
    };
    db.getPool.mockReturnValue({ connect: async () => client });
    tools.validateSelectQuery.mockImplementation((sql: string) => ({
      ok: true,
      sql: sql.replace(/;$/, ""),
    }));

    const out = await ticketsQueryHandler("product_purchased ILIKE '%lg%'", 10);

    // read-only txn + timeout were issued before the query
    const calls = client.query.mock.calls.map((c) => c[0]);
    expect(calls[1]).toBe("BEGIN TRANSACTION READ ONLY");
    expect(calls[0]).toContain("statement_timeout");
    // the actual query is the validated one
    expect(calls[2]).toContain("FROM tickets WHERE product_purchased ILIKE '%lg%'");
    expect(calls[2]).toContain("LIMIT 10");
    expect(client.release).toHaveBeenCalled();
    expect(out.content[0]!.text).toContain("ticket #42");
    expect(out.content[0]!.text).toContain("wifi keeps dropping");
  });

  it("rejects queries the allowlist blocks", async () => {
    tools.validateSelectQuery.mockReturnValue({ ok: false, reason: "multiple statements are not allowed" });
    await expect(ticketsQueryHandler("1=1; DROP TABLE tickets", 5)).rejects.toThrow("blocked");
  });

  it("clamps limit to 200 and rejects empty where", async () => {
    const client = { query: vi.fn(async () => ({ rows: [] })), release: vi.fn() };
    db.getPool.mockReturnValue({ connect: async () => client });
    tools.validateSelectQuery.mockImplementation((sql: string) => ({ ok: true, sql }));

    await ticketsQueryHandler("status = 'open'", 99999);
    const sql = client.query.mock.calls[2]![0] as string;
    expect(sql).toContain("LIMIT 200");

    await expect(ticketsQueryHandler("   ", 5)).rejects.toThrow("empty WHERE");
  });
});
