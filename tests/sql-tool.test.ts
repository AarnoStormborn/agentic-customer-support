/**
 * SQL tool tests — the SELECT-only allowlist is the security boundary for the
 * LLM-generated SQL (v1 lesson: unrestricted SQL + PII = disaster).
 */
import { describe, it, expect } from "vitest";
import { validateSelectQuery } from "../src/tools/sql-tool.js";
import { executeMockQuery, parseSelect } from "../src/tools/sql-mock.js";

describe("validateSelectQuery (allowlist)", () => {
  it("accepts plain SELECTs", () => {
    expect(validateSelectQuery("SELECT ticket_id FROM tickets LIMIT 5").ok).toBe(true);
    expect(validateSelectQuery("explain select * from tickets").ok).toBe(true);
  });

  it("rejects mutations", () => {
    for (const q of [
      "DELETE FROM tickets",
      "UPDATE tickets SET status='x'",
      "DROP TABLE tickets",
      "INSERT INTO tickets VALUES (1)",
      "TRUNCATE tickets",
      "ALTER TABLE tickets ADD COLUMN x int",
      "CREATE TABLE evil(x int)",
    ]) {
      const v = validateSelectQuery(q);
      expect(v.ok).toBe(false);
      expect(v.ok || v.reason).toContain("only SELECT");
    }
  });

  it("rejects multiple statements and comment-only input", () => {
    expect(validateSelectQuery("SELECT 1; DROP TABLE tickets").ok).toBe(false);
    expect(validateSelectQuery("-- just a comment").ok).toBe(false);
  });

  it("rejects forbidden keywords embedded in SELECTs", () => {
    expect(validateSelectQuery("SELECT 1 FROM pg_sleep(10)").ok).toBe(false);
    expect(validateSelectQuery("SELECT * FROM tickets WHERE 1=1 UNION SELECT password FROM users").ok).toBe(false);
  });

  it("strips trailing semicolons and allows param placeholders", () => {
    const v = validateSelectQuery("SELECT * FROM tickets WHERE product_purchased ILIKE $1;;");
    expect(v.ok).toBe(true);
    if (v.ok) expect(v.sql).toContain("$1");
  });
});

describe("executeMockQuery (dev-mode in-memory tickets)", () => {
  it("parses and executes simple SELECTs", () => {
    const res = executeMockQuery("SELECT ticket_id, product_purchased FROM tickets LIMIT 3");
    expect(res.rows.length).toBeGreaterThan(0);
    expect(res.columns).toContain("ticket_id");
  });

  it("supports ILIKE filtering", () => {
    const res = executeMockQuery("SELECT * FROM tickets WHERE product ILIKE '%lg%' LIMIT 20");
    expect(res.rows.length).toBeGreaterThan(0);
    expect(parseSelect("SELECT * FROM tickets WHERE product ILIKE '%lg%'")).not.toHaveProperty("error");
  });
});
