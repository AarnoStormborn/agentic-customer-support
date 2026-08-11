/**
 * tickets_query — the SQL specialist's one tool (also exposed to the supervisor).
 *
 * SECURITY (defense in depth, architecture rule 3):
 *   1. SELECT-only allowlist validation lives HERE (never trust the model).
 *   2. The guardrails extension re-validates the same input at the tool_call
 *      interception layer.
 *   3. Real-mode execution sets statement_timeout=1000ms + READ ONLY transaction
 *      and assumes a Postgres role with SELECT-only grants (read-only role note).
 *
 * Modes:
 *   SQL_MODE=mock (default) — in-memory tickets table (src/tools/sql-mock.ts).
 *   SQL_MODE=real — loads the module at SQL_IMPL (must export getPool(): Pool)
 *                   and runs the query against Postgres.
 */

import { defineTool } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { executeMockQuery } from "./sql-mock.js";
import { MAX_SQL_QUERY_LEN, MAX_SQL_RESULT_ROWS, TOOL_NAMES } from "../config/limits.js";

export type QueryVerdict = { ok: true; sql: string } | { ok: false; reason: string };

const FORBIDDEN = /\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|merge|call|do|vacuum|analyze|set|reset|show|with\s+.*\b(delete|update|insert))\b/i;

/** Strip SQL comments (-- line and /* *​/ block) so they can't smuggle statements. */
export function stripSqlComments(sql: string): string {
  return sql
    .replace(/\/\*[\s\S]*?\*\//g, " ")
    .split("\n")
    .map((line) => line.replace(/--.*$/, " "))
    .join("\n");
}

/**
 * SELECT-only allowlist check.
 * - Accepts SELECT... and EXPLAIN SELECT...
 * - Rejects anything else, multiple statements (stray ";" after the terminator),
 *   writes, and oversized queries.
 */
export function validateSelectQuery(raw: string): QueryVerdict {
  if (typeof raw !== "string" || raw.trim().length === 0) {
    return { ok: false, reason: "empty query" };
  }
  const cleaned = stripSqlComments(raw).trim().replace(/;+$/, "");
  if (cleaned.length === 0) return { ok: false, reason: "query is only comments" };
  if (cleaned.length > MAX_SQL_QUERY_LEN) {
    return { ok: false, reason: `query too long (${cleaned.length} > ${MAX_SQL_QUERY_LEN} chars)` };
  }
  if (/;/.test(cleaned)) {
    return { ok: false, reason: "multiple statements are not allowed (single SELECT only)" };
  }
  if (!/^select\b/i.test(cleaned) && !/^explain\s+select\b/i.test(cleaned)) {
    return { ok: false, reason: "only SELECT queries are allowed (read-only database)" };
  }
  const forbidden = cleaned.match(FORBIDDEN);
  if (forbidden) {
    return { ok: false, reason: `forbidden keyword in query: '${forbidden[0].trim()}'` };
  }
  return { ok: true, sql: cleaned };
}

async function executeReal(query: string, params: unknown[], signal?: AbortSignal): Promise<{ rows: Record<string, unknown>[]; mode: string }> {
  const implPath = process.env.SQL_IMPL;
  if (!implPath) throw new Error("SQL_MODE=real requires SQL_IMPL (module exporting getPool())");
  const mod: unknown = await import(implPath);
  const pool = (mod as { getPool?: () => { connect(): Promise<unknown> } }).getPool?.();
  if (!pool) throw new Error(`SQL_IMPL module '${implPath}' does not export getPool()`);
  const client = await pool.connect() as {
    query(sql: string, p?: unknown[]): Promise<{ rows: Record<string, unknown>[] }>;
    release(): void;
  };
  try {
    // max 1s statement timeout + read-only transaction (belt & braces on top of the allowlist)
    await client.query("SET statement_timeout = 1000");
    await client.query("BEGIN TRANSACTION READ ONLY");
    const result = await client.query(query, params);
    await client.query("COMMIT");
    return { rows: result.rows, mode: "real" };
  } catch (err) {
    await client.query("ROLLBACK").catch(() => {});
    throw err;
  } finally {
    client.release();
  }
}

export const ticketsQueryTool = defineTool({
  name: TOOL_NAMES.ticketsQuery,
  label: "Tickets Query",
  description:
    "Query the support-tickets database (SELECT-only, read-only). Returns matching ticket rows as JSON. " +
    "Columns include id, customer_name, product, issue, status, priority, created_at. " +
    "Example: SELECT id, product, status FROM tickets WHERE product ILIKE '%lg tv%' ORDER BY id DESC LIMIT 5. " +
    "Write the query yourself; the tool validates it (only SELECT / EXPLAIN SELECT, single statement).",
  parameters: Type.Object({
    query: Type.String({ description: "A single SELECT statement (optionally with $1-style params)" }),
    params: Type.Optional(Type.Array(Type.Unknown(), { description: "Values for $1, $2… placeholders" })),
  }),
  execute: async (_toolCallId, params, signal) => {
    signal?.throwIfAborted();
    const verdict = validateSelectQuery(params.query);
    if (!verdict.ok) {
      throw new Error(`tickets_query blocked: ${verdict.reason}`);
    }

    const mode = process.env.SQL_MODE ?? "mock";
    let rows: Record<string, unknown>[];
    let columns: string[] = [];
    let truncated = false;

    if (mode === "real") {
      const result = await executeReal(verdict.sql, params.params ?? [], signal);
      rows = result.rows.slice(0, MAX_SQL_RESULT_ROWS);
      truncated = result.rows.length > MAX_SQL_RESULT_ROWS;
      columns = rows.length > 0 ? Object.keys(rows[0] ?? {}) : [];
    } else {
      const result = executeMockQuery(verdict.sql);
      rows = result.rows;
      columns = result.columns;
      truncated = result.truncated;
    }

    const text = rows.length === 0
      ? "No tickets match the query."
      : `Returned ${rows.length} row${rows.length === 1 ? "" : "s"} (columns: ${columns.join(", ")}):\n${JSON.stringify(rows, null, 2)}`;

    return {
      content: [{ type: "text", text }],
      details: {
        tool: TOOL_NAMES.ticketsQuery,
        query: verdict.sql,
        rowCount: rows.length,
        columns,
        truncated,
        mode,
        readOnly: true,
      },
    };
  },
});
