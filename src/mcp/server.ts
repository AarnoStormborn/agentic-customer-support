/**
 * src/mcp/server.ts — MCP server exposing the retrieval tools.
 *
 *   kb_search       hybrid (vector + FTS + RRF) search over the knowledge base
 *   tickets_query   read-only, allowlisted search over the tickets table
 *
 * Real implementations live in src/mcp/handlers.ts (wired here; unit-tested
 * separately in tests/mcp.test.ts).
 */
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { kbSearchHandler, ticketsQueryHandler } from "./handlers.js";

export const MCP_SERVER_NAME = "acs-retrieval";
export const MCP_SERVER_VERSION = "0.2.0";

export function buildMcpServer(): McpServer {
  const server = new McpServer({
    name: MCP_SERVER_NAME,
    version: MCP_SERVER_VERSION,
  });

  server.registerTool(
    "kb_search",
    {
      title: "Knowledge Base Search",
      description:
        "Hybrid search (vector similarity + full-text + RRF fusion) over the support " +
        "knowledge base and product manuals. Returns ranked chunks with manual, section " +
        "and page references.",
      inputSchema: {
        query: z.string().min(1).max(500).describe("The user's question or search terms"),
        topK: z
          .number()
          .int()
          .min(1)
          .max(20)
          .optional()
          .describe("Number of results (default 5)"),
      },
    },
    async ({ query, topK }) => kbSearchHandler(query, topK),
  );

  server.registerTool(
    "tickets_query",
    {
      title: "Tickets Database Query",
      description:
        "Read-only search over support tickets. Provide a SQL WHERE clause; the query is " +
        "validated (SELECT-only allowlist: no DML/DDL, no UNION, no stacked statements) and " +
        "run in a read-only transaction with a short timeout. Example where: " +
        "ticket_type ILIKE '%refund%' AND product_purchased ILIKE '%lg%'",
      inputSchema: {
        where: z
          .string()
          .min(1)
          .max(500)
          .describe("SQL WHERE clause on the tickets table"),
        limit: z
          .number()
          .int()
          .min(1)
          .max(200)
          .optional()
          .describe("Max rows (clamped to 200, default 50)"),
      },
    },
    async ({ where, limit }) => ticketsQueryHandler(where, limit),
  );

  return server;
}
