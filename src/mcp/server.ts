/**
 * src/mcp/server.ts — MCP server exporting retrieval tools (scaffold).
 *
 * Exposes the two retrieval tools as an MCP server:
 *   kb_search       hybrid (vector + FTS) search over the knowledge base
 *   tickets_query   read-only search over the tickets table
 *
 * Real implementations land after integration with retrieval-core
 * (`searchHybrid` from src/retrieval/index.ts). For now the handlers return a
 * placeholder so the tool schemas + registration are provably correct.
 */
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

export const MCP_SERVER_NAME = "acs-retrieval";
export const MCP_SERVER_VERSION = "0.1.0";

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
        "Hybrid search (pgvector + Postgres full-text + RRF) over the support knowledge base " +
        "and manuals. Returns ranked chunks with section paths and pages.",
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
    async ({ query, topK }) => {
      // TODO(integration): call searchHybrid({ query, topK }) from src/retrieval.
      return {
        content: [
          {
            type: "text" as const,
            text: `[mock] kb_search(query=${JSON.stringify(query)}, topK=${topK ?? 5}) — real retrieval lands at integration`,
          },
        ],
      };
    },
  );

  server.registerTool(
    "tickets_query",
    {
      title: "Tickets Database Query",
      description:
        "Read-only search over support tickets (SELECT-only, parameterized, LIMIT-clamped). " +
        "Use for ticket history, product issues, and customer context.",
      inputSchema: {
        where: z
          .string()
          .min(1)
          .max(500)
          .describe("SQL WHERE clause on tickets (e.g. ticket_subject ILIKE '%wifi%')"),
        limit: z
          .number()
          .int()
          .min(1)
          .max(200)
          .optional()
          .describe("Max rows (clamped to 200)"),
      },
    },
    async ({ where, limit }) => {
      // TODO(integration): call the read-only SQL executor from src/tools/sql-tool.
      return {
        content: [
          {
            type: "text" as const,
            text: `[mock] tickets_query(where=${JSON.stringify(where)}, limit=${limit ?? 50}) — real SQL lands at integration`,
          },
        ],
      };
    },
  );

  return server;
}
