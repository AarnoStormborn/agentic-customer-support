/**
 * src/mcp/handlers.ts — real MCP tool handlers (Phase 5b.4).
 *
 * Exported separately from the McpServer wiring so they are unit-testable
 * without an MCP transport.
 */
import { searchHybrid } from "../retrieval/index.js";
import { getPool } from "../db/pool.js";
import { validateSelectQuery } from "../tools/sql-tool.js";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";

const MAX_TICKET_ROWS = 200;
const MCP_SQL_TIMEOUT_MS = 1500;

/** kb_search — hybrid retrieval over manuals/knowledge base. */
export async function kbSearchHandler(query: string, topK?: number): Promise<CallToolResult> {
  const { results, queryTimeMs } = await searchHybrid({
    query,
    topK: topK ?? 5,
    sourceTypes: ["kb"],
  });

  const text =
    results.length === 0
      ? "No knowledge base results."
      : results
          .map((r, i) => {
            const where = [
              r.source.docName ?? r.source.title,
              r.source.sectionPath ? `› ${r.source.sectionPath}` : "",
              r.source.page ? `(p.${r.source.page})` : "",
              r.source.url ?? "",
            ]
              .filter(Boolean)
              .join(" ");
            return `[${i + 1}] score=${r.score.toFixed(4)} ${where}\n${r.text}`;
          })
          .join("\n\n") +
        `\n\n(${results.length} results in ${queryTimeMs}ms)`;

  return { content: [{ type: "text", text }] };
}

/**
 * tickets_query — read-only WHERE-clause search over tickets.
 *
 * The full SQL is built here and run through the same SELECT-only allowlist as
 * the agent tool (blocks semicolons, DML/DDL, UNION, pg_sleep, etc.), then
 * executed in a read-only transaction with a statement timeout.
 */
export async function ticketsQueryHandler(where: string, limit?: number): Promise<CallToolResult> {
  const cleaned = where.trim().replace(/;+$/, "");
  if (!cleaned) throw new Error("tickets_query: empty WHERE clause");
  const capped = Math.min(MAX_TICKET_ROWS, Math.max(1, Math.floor(limit ?? 50)));

  const sql = `SELECT ticket_id, source, customer_name, product_purchased, date_of_purchase,
                      ticket_type, ticket_priority, ticket_channel, ticket_subject,
                      complaint_narrative, company, status
               FROM tickets WHERE ${cleaned} ORDER BY ticket_id DESC LIMIT ${capped}`;

  const verdict = validateSelectQuery(sql);
  if (!verdict.ok) throw new Error(`tickets_query blocked: ${verdict.reason}`);

  const pool = getPool();
  const client = await pool.connect();
  try {
    await client.query(`SET statement_timeout = ${MCP_SQL_TIMEOUT_MS}`);
    await client.query("BEGIN TRANSACTION READ ONLY");
    const res = await client.query(verdict.sql);
    await client.query("COMMIT");

    const text =
      res.rows.length === 0
        ? "No tickets match the WHERE clause."
        : res.rows
            .map((r) =>
              [
                `ticket #${r.ticket_id}`,
                r.product_purchased ? `product=${r.product_purchased}` : "",
                r.ticket_type ? `type=${r.ticket_type}` : "",
                r.ticket_priority ? `priority=${r.ticket_priority}` : "",
                r.status ? `status=${r.status}` : "",
                r.complaint_narrative ? `narrative=${String(r.complaint_narrative).slice(0, 200)}` : "",
              ]
                .filter(Boolean)
                .join(" | "),
            )
            .join("\n") + `\n\n(${res.rows.length} rows)`;

    return { content: [{ type: "text", text }] };
  } catch (err) {
    await client.query("ROLLBACK").catch(() => {});
    throw err;
  } finally {
    client.release();
  }
}
