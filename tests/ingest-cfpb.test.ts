/**
 * CFPB ingest tests (Phase 5b.3) — row mapping unit tests + a small
 * DB-guarded end-to-end CSV ingest (ACS_TEST_DB=1).
 */
import { describe, it, expect } from "vitest";
import { writeFile, mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { mapCfpbRecord } from "../src/retrieval/ingest-cfpb.js";

describe("mapCfpbRecord", () => {
  it("maps CFPB headers to tickets fields", () => {
    const row = mapCfpbRecord({
      "Complaint ID": "1234567",
      "Date received": "01/15/2024",
      Product: "Credit card or prepaid card",
      "Sub-product": "General-purpose credit card",
      Issue: "Billing disputes",
      Company: "Acme Bank",
      State: "CA",
      "ZIP code": "90210",
      "Submitted via": "Web",
      "Consumer complaint narrative": "I was charged twice.",
      "Company response to consumer": "Closed with monetary relief",
    });
    expect(row).toMatchObject({
      source_ticket_id: "1234567",
      product_purchased: "Credit card or prepaid card",
      ticket_type: "Billing disputes",
      company: "Acme Bank",
      state: "CA",
      zip_code: "90210",
      ticket_channel: "Web",
      complaint_narrative: "I was charged twice.",
      company_response: "Closed with monetary relief",
    });
  });

  it("tolerates missing narrative / optional fields (NULL)", () => {
    const row = mapCfpbRecord({ "Complaint ID": "1", Product: "Debt collection", Issue: "x" });
    expect(row.complaint_narrative).toBeNull();
    expect(row.state).toBeNull();
    expect(row.ticket_channel).toBeNull();
    expect(row.product_purchased).toBe("Debt collection");
  });

  it("defaults missing product to Unknown", () => {
    const row = mapCfpbRecord({ "Complaint ID": "2", Issue: "y" });
    expect(row.product_purchased).toBe("Unknown");
  });
});

describe("ingestCfpbCsv (live DB)", () => {
  const run = process.env.ACS_TEST_DB === "1" ? it : it.skip;

  run("streams a small CSV into tickets with source='cfpb'", async () => {
    const dir = await mkdtemp(join(tmpdir(), "acs-cfpb-test-"));
    const csv = join(dir, "sample.csv");
    await writeFile(
      csv,
      [
        "Complaint ID,Date received,Product,Sub-product,Issue,Company,State,\"ZIP code\",\"Submitted via\",\"Consumer complaint narrative\"",
        "9001,01/01/2025,Bank account or service,Checking account,Deposits and withdrawals,Bank X,NY,10001,Web,\"Narrative with, a comma\"",
        '9002,01/02/2025,Debt collection,,Attempts to collect debt not owed,Collector Y,,,Phone,',
      ].join("\n"),
    );

    const { ingestCfpbCsv } = await import("../src/retrieval/ingest-cfpb.js");
    const { getPool } = await import("../src/db/pool.js");
    const pool = getPool();

    // clean slate for the sample ids
    await pool.query("DELETE FROM tickets WHERE source = 'cfpb' AND source_ticket_id IN ('9001','9002')");

    const summary = await ingestCfpbCsv(csv);
    expect(summary.rowsRead).toBe(2);
    expect(summary.rowsInserted).toBe(2);

    const res = await pool.query(
      "SELECT * FROM tickets WHERE source = 'cfpb' AND source_ticket_id = '9001'",
    );
    expect(res.rows[0]).toMatchObject({
      product_purchased: "Bank account or service",
      ticket_channel: "Web",
      state: "NY",
    });
    expect(res.rows[0].complaint_narrative).toContain("a comma");

    // idempotent on re-run
    const second = await ingestCfpbCsv(csv);
    expect(second.rowsUpdated).toBe(2);

    await pool.query("DELETE FROM tickets WHERE source = 'cfpb' AND source_ticket_id IN ('9001','9002')");
  });
});
