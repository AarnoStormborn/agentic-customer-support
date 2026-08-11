/**
 * Public exports of the agent-runtime track (integration contract:
 * docs/design/integration-contract.md).
 */

export { createSupportRuntime } from "./session.js";
export type { SupportRuntime, CreateSupportRuntimeOptions } from "./session.js";

export { guardrailsExtension, supportGuardrails } from "../guardrails/extension.js";

// Tools (exposed for tests / future registry)
export { kbSearchTool } from "../tools/rag-tool.js";
export { ticketsQueryTool, validateSelectQuery } from "../tools/sql-tool.js";
export { webSearchTool } from "../tools/web-tool.js";
export { routeToAgent } from "../agent/route-to-agent.js";
