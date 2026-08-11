/**
 * Shared limits & constants for the agent runtime.
 *
 * Kept in one place so the API/streaming track and the guardrails can agree on
 * budgets without importing each other's internals.
 */

/** Max chars accepted from the user per turn (input hook truncates beyond this). */
export const MAX_INPUT_CHARS = 4000;

/** Max chars of a tool result kept before an LLM call (context hook clamps). */
export const MAX_TOOL_RESULT_CHARS = 6000;

/** Per-turn budget for the whole supervisor run (CLI/API apply this). */
export const TURN_BUDGET_MS = 120_000;

/** Per-call budget for a route_to_agent child session. */
export const CHILD_TIMEOUT_MS = 60_000;

/** Max child specialist sessions running at the same time. */
export const MAX_CONCURRENT_CHILDREN = 3;

/** Max rows returned by the tickets query tool. */
export const MAX_SQL_RESULT_ROWS = 50;

/** Max source refs attached to the SSE `done` event (deduped). */
export const MAX_DONE_SOURCES = 25;

/** Max SQL text length accepted by the tickets query tool. */
export const MAX_SQL_QUERY_LEN = 2000;

/** Web search HTTP timeout. */
export const WEB_SEARCH_TIMEOUT_MS = 15_000;

/** Max web search query length accepted by the tool + guardrail. */
export const MAX_WEB_QUERY_LEN = 500;

/** Sub-agents route_to_agent may dispatch to. */
export const ALLOWED_AGENTS = new Set(["rag", "sql", "web"]);

/** The four tools the supervisor session exposes (contract: rag, sql, web, route_to_agent). */
export const TOOL_NAMES = {
  kbSearch: "kb_search",
  ticketsQuery: "tickets_query",
  webSearch: "web_search",
  routeToAgent: "route_to_agent",
} as const;
