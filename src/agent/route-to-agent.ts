/**
 * route_to_agent — the sub-agent dispatch tool (docs/design/backend-agent-retrieval.md §3.2).
 *
 * pi has no native handoffs, so routing is a custom tool that spawns a CHILD
 * AgentSession per call: specialist system prompt (systemPromptOverride), ONE
 * tool each, disposed after use. Children are in-memory and cheap.
 *
 * Safety: bounded concurrency (MAX_CONCURRENT_CHILDREN), per-child timeout
 * (CHILD_TIMEOUT_MS), parent AbortSignal propagation. The tool_call guardrail
 * hook additionally validates `agent ∈ {rag,sql,web}` before we spawn.
 */

import {
  createAgentSession,
  DefaultResourceLoader,
  defineTool,
  getAgentDir,
  ModelRuntime,
  SessionManager,
  SettingsManager,
} from "@earendil-works/pi-coding-agent";
import type { AgentSession, AgentSessionEvent } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { SPECIALIST_PROMPTS } from "./specialists.js";
import type { AgentKind } from "./specialists.js";
import { kbSearchTool } from "../tools/rag-tool.js";
import { ticketsQueryTool } from "../tools/sql-tool.js";
import { webSearchTool } from "../tools/web-tool.js";
import { supportGuardrails } from "../guardrails/extension.js";
import { resolveSpecialistModel } from "../runtime/model.js";
import type { ModelLike } from "../runtime/model.js";
import { ALLOWED_AGENTS, CHILD_TIMEOUT_MS, MAX_CONCURRENT_CHILDREN, TOOL_NAMES } from "../config/limits.js";

const SPECIALIST_TOOLS = {
  rag: kbSearchTool,
  sql: ticketsQueryTool,
  web: webSearchTool,
} as const;

// ---- shared ModelRuntime (created once per process; children reuse it) ----

interface SharedDeps {
  modelRuntime: ModelRuntime;
  supervisorModel: ModelLike;
}

let shared: SharedDeps | null = null;

/** Called by createSupportRuntime so child sessions reuse the parent's runtime. */
export function configureRouteToAgent(deps: SharedDeps): void {
  shared = deps;
}

async function getShared(): Promise<SharedDeps> {
  if (!shared) {
    const modelRuntime = await ModelRuntime.create();
    const available = await modelRuntime.getAvailable();
    if (available.length === 0) {
      throw new Error("No authenticated models found for specialist sessions.");
    }
    shared = { modelRuntime, supervisorModel: available[0]! };
  }
  return shared;
}

// ---- bounded concurrency (tiny semaphore) ----

let activeChildren = 0;
const waiters: (() => void)[] = [];

async function acquireSlot(signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) throw new Error("route_to_agent aborted by parent");
  if (activeChildren < MAX_CONCURRENT_CHILDREN) {
    activeChildren += 1;
    return;
  }
  await new Promise<void>((resolve) => {
    const onAbort = () => {
      const idx = waiters.indexOf(resolve);
      if (idx >= 0) waiters.splice(idx, 1);
      resolve();
    };
    signal?.addEventListener("abort", onAbort, { once: true });
    waiters.push(resolve);
  });
  if (signal?.aborted) throw new Error("route_to_agent aborted by parent");
}

function releaseSlot(): void {
  activeChildren -= 1;
  const next = waiters.shift();
  if (next) {
    activeChildren += 1;
    next();
  }
}

// ---- child session run ----

interface CollectedRun {
  finalText: string;
  sources: unknown[];
  turnCount: number;
  tokens: number;
  modelLabel: string;
  toolCalls: Array<{ tool: string; args: unknown }>;
}

function collectEvents(session: AgentSession): { tokens: number; turnCount: number; toolCalls: Array<{ tool: string; args: unknown }> } {
  let tokens = 0;
  let turnCount = 0;
  const toolCalls: Array<{ tool: string; args: unknown }> = [];
  session.subscribe((event) => {
    if (event.type === "message_update" && event.assistantMessageEvent.type === "text_delta") {
      tokens += event.assistantMessageEvent.delta.length;
    } else if (event.type === "turn_start") {
      turnCount += 1;
    } else if (event.type === "tool_execution_start") {
      toolCalls.push({ tool: event.toolName, args: event.args });
    }
  });
  return { tokens, turnCount, toolCalls };
}

/** Extract final assistant text + tool sources from the child's message history. */
function extractFromMessages(session: AgentSession): { finalText: string; sources: unknown[] } {
  const sources: unknown[] = [];
  let finalText = "";
  for (const m of session.messages) {
    const msg = m as {
      role: string;
      content?: unknown;
      details?: { sources?: unknown[] };
    };
    if (msg.role === "toolResult") {
      if (Array.isArray(msg.details?.sources)) sources.push(...msg.details!.sources!);
    } else if (msg.role === "assistant") {
      const parts = Array.isArray(msg.content) ? msg.content : [];
      for (const part of parts) {
        const p = part as { type?: string; text?: string };
        if (p.type === "text" && typeof p.text === "string") finalText += p.text;
      }
    }
  }
  return { finalText: finalText.trim(), sources };
}

async function runChild(kind: AgentKind, query: string, parentSignal?: AbortSignal): Promise<CollectedRun> {
  const { modelRuntime, supervisorModel } = await getShared();
  const model = await resolveSpecialistModel(modelRuntime, supervisorModel);
  const tool = SPECIALIST_TOOLS[kind];

  const settingsManager = SettingsManager.inMemory({
    compaction: { enabled: false },
    retry: { enabled: true, maxRetries: 1 },
  });
  const loader = new DefaultResourceLoader({
    cwd: process.cwd(),
    agentDir: getAgentDir(),
    settingsManager,
    systemPromptOverride: () => SPECIALIST_PROMPTS[kind],
    appendSystemPromptOverride: () => [],
    extensionFactories: [supportGuardrails],
  });
  await loader.reload();

  const { session } = await createAgentSession({
    cwd: process.cwd(),
    agentDir: getAgentDir(),
    model,
    thinkingLevel: "off", // specialists answer fast; no hidden reasoning
    modelRuntime,
    resourceLoader: loader,
    sessionManager: SessionManager.inMemory(),
    settingsManager,
    noTools: "builtin", // child has NO filesystem tools either
    customTools: [tool],
    tools: [tool.name], // allowlist: only the specialist tool is callable
  });

  const counters = collectEvents(session);

  const abortFromParent = () => {
    if (parentSignal?.aborted) void session.abort();
  };
  parentSignal?.addEventListener("abort", abortFromParent, { once: true });

  let timer: NodeJS.Timeout | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      void session.abort();
      reject(new Error(`specialist '${kind}' timed out after ${CHILD_TIMEOUT_MS}ms`));
    }, CHILD_TIMEOUT_MS);
  });

  try {
    await Promise.race([session.prompt(query), timeout]);
  } catch (err) {
    throw new Error(`specialist '${kind}' failed: ${(err as Error).message}`);
  } finally {
    clearTimeout(timer);
    parentSignal?.removeEventListener("abort", abortFromParent);
  }

  try {
    const { finalText, sources } = extractFromMessages(session);
    return {
      finalText,
      sources,
      turnCount: counters.turnCount,
      tokens: counters.tokens,
      modelLabel: `${model.provider}/${model.id}`,
      toolCalls: counters.toolCalls,
    };
  } finally {
    session.dispose(); // ← dispose after use (design §3.2)
  }
}

// ---- the tool ----

export const routeToAgent = defineTool({
  name: TOOL_NAMES.routeToAgent,
  label: "Route to Specialist Agent",
  description:
    "Route a query to a focused specialist sub-agent. agent must be one of: " +
    "'rag' (knowledge base / product manuals), 'sql' (tickets database), 'web' (live web search). " +
    "Call it with the user's exact question; it runs the specialist and returns its answer with sources. " +
    "Prefer it for complex or multi-step queries; use the direct tools for quick single lookups. " +
    "Never call it for small talk.",
  parameters: Type.Object({
    agent: Type.String({ description: "One of: rag | sql | web" }),
    query: Type.String({ description: "The query for the specialist (user's question, verbatim)" }),
  }),
  execute: async (_toolCallId, params, signal) => {
    const kind = params.agent as string;
    if (!ALLOWED_AGENTS.has(kind)) {
      throw new Error(`route_to_agent: unknown sub-agent '${kind}' (allowed: rag, sql, web)`);
    }

    await acquireSlot(signal);
    try {
      const run = await runChild(kind as AgentKind, params.query, signal);
      const text =
        run.finalText ||
        `Specialist '${kind}' returned no text. (Its tool may have found nothing; see sources.)`;
      return {
        content: [{ type: "text", text }],
        details: {
          tool: TOOL_NAMES.routeToAgent,
          agent: kind,
          query: params.query,
          sources: run.sources,
          turnCount: run.turnCount,
          tokens: run.tokens,
          model: run.modelLabel, // model used by the child
          childToolCalls: run.toolCalls,
        },
      };
    } finally {
      releaseSlot();
    }
  },
});
