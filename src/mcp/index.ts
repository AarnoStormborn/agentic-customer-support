/**
 * src/mcp/index.ts — MCP server entry (npm run mcp).
 *
 * Runs the retrieval MCP server over stdio. Logs go to stderr so stdout stays clean
 * for the MCP protocol. Scaffold only — the tools return placeholders until
 * integration with retrieval-core.
 */
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { buildMcpServer, MCP_SERVER_NAME, MCP_SERVER_VERSION } from "./server.js";

async function main(): Promise<void> {
  const server = buildMcpServer();
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error(`${MCP_SERVER_NAME} v${MCP_SERVER_VERSION} listening on stdio`);
}

main().catch((err) => {
  console.error("mcp server failed:", err);
  process.exit(1);
});
