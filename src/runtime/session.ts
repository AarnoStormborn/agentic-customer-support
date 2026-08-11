/**
 * Support runtime — one AgentSession per conversation turn, wired exactly as
 * docs/design/backend-agent-retrieval.md §3.1 + the integration contract:
 *
 * - noTools: "builtin"            → support sessions get NO filesystem tools (locked rule)
 * - customTools: [rag, sql, web, route_to_agent]
 * - systemPromptOverride          → supervisor prompt (src/agent/support-prompt.ts)
 * - SettingsManager.inMemory({ compaction: false })
 * - SessionManager.inMemory()     (optional sessionDir → JSONL persistence)
 * - ModelRuntime.create() + model selection (PI_MODEL env → first available fallback)
 * - thinkingLevel "off"
 *
 * All pi SDK imports stay inside src/runtime/ (AGENTS.md isolation rule).
 */

import {
  createAgentSession,
  DefaultResourceLoader,
  getAgentDir,
  ModelRuntime,
  SessionManager,
  SettingsManager,
} from "@earendil-works/pi-coding-agent";
import type { AgentSession, AgentSessionEvent } from "@earendil-works/pi-coding-agent";
import { createSourceEnricher, type EnrichedEvent } from "./sources.js";
import { SUPPORT_SYSTEM_PROMPT } from "../agent/support-prompt.js";
import { routeToAgent, configureRouteToAgent } from "../agent/route-to-agent.js";
import { kbSearchTool } from "../tools/rag-tool.js";
import { ticketsQueryTool } from "../tools/sql-tool.js";
import { webSearchTool } from "../tools/web-tool.js";
import { supportGuardrails } from "../guardrails/extension.js";
import { resolveSupervisorModel } from "./model.js";
import { TURN_BUDGET_MS } from "../config/limits.js";

export interface SupportRuntime {
  prompt(text: string, opts?: { images?: unknown[] }): Promise<void>;
  steer(text: string): Promise<void>;
  abort(): Promise<void>;
  subscribe(fn: (event: unknown) => void): () => void; // pi SDK AgentSessionEvent
  getLastMessages(): unknown[];
  dispose(): void;
}

export interface CreateSupportRuntimeOptions {
  model?: string; // "provider/model" — overrides PI_MODEL
  chatId?: string;
  sessionDir?: string; // undefined = in-memory
}

export class SupportRuntimeImpl implements SupportRuntime {
  constructor(
    private readonly session: AgentSession,
    private readonly modelRuntime: ModelRuntime,
    readonly modelLabel: string,
    readonly chatId: string,
  ) {}

  async prompt(text: string, opts?: { images?: unknown[] }): Promise<void> {
    await this.session.prompt(text, opts as Parameters<AgentSession["prompt"]>[1]);
  }

  /**
   * Prompt with a per-turn budget: races the run against TURN_BUDGET_MS and
   * aborts the session on timeout (design §3.5 "turn timeout" row).
   */
  async promptWithBudget(text: string, budgetMs: number = TURN_BUDGET_MS): Promise<void> {
    let timer: NodeJS.Timeout | undefined;
    const timeout = new Promise<never>((_, reject) => {
      timer = setTimeout(() => {
        void this.session.abort();
        reject(new Error(`turn timed out after ${budgetMs}ms`));
      }, budgetMs);
    });
    try {
      await Promise.race([this.session.prompt(text), timeout]);
    } finally {
      clearTimeout(timer);
    }
  }

  async steer(text: string): Promise<void> {
    await this.session.steer(text);
  }

  async abort(): Promise<void> {
    await this.session.abort();
  }

  subscribe(fn: (event: unknown) => void): () => void {
    // Wrap the subscriber to attach tool `details.sources` to the `agent_settled`
    // event (integration contract: api-streaming's bridge reads `event.sources`
    // on `agent_settled` to build the `done` payload's sources[]).
    const enricher = createSourceEnricher((sources) => ({
      type: "agent_settled",
      sources,
    }));
    const wrapped = (e: AgentSessionEvent) => {
      const out = enricher.handle(e as unknown as EnrichedEvent);
      if (out && out !== (e as unknown as EnrichedEvent)) {
        // agent_settled was replaced with the enriched copy
        return fn(out);
      }
      return fn(e);
    };
    return this.session.subscribe(wrapped);
  }

  getLastMessages(): unknown[] {
    return [...this.session.messages];
  }

  dispose(): void {
    this.session.dispose();
  }
}

export async function createSupportRuntime(
  opts: CreateSupportRuntimeOptions = {},
): Promise<SupportRuntimeImpl> {
  const chatId = opts.chatId ?? `chat-${Date.now()}`;

  const modelRuntime = await ModelRuntime.create();
  const model = await resolveSupervisorModel(modelRuntime, opts.model);
  const modelLabel = `${model.provider}/${model.id}`;

  const settingsManager = SettingsManager.inMemory({
    compaction: { enabled: false }, // compacted long conversations bounded; disable for now per contract
    retry: { enabled: true, maxRetries: 2 },
  });

  const sessionManager = opts.sessionDir
    ? SessionManager.create(opts.sessionDir)
    : SessionManager.inMemory();

  const loader = new DefaultResourceLoader({
    cwd: process.cwd(),
    agentDir: getAgentDir(),
    settingsManager,
    systemPromptOverride: () => SUPPORT_SYSTEM_PROMPT, // replaces pi's coding prompt entirely
    appendSystemPromptOverride: () => [], // no AGENTS.md / append leaks into product sessions
    extensionFactories: [supportGuardrails],
  });
  await loader.reload();

  const { session } = await createAgentSession({
    cwd: process.cwd(),
    agentDir: getAgentDir(),
    model,
    thinkingLevel: "off",
    modelRuntime,
    resourceLoader: loader,
    sessionManager,
    settingsManager,
    noTools: "builtin", // ← architecture rule 1: support sessions get NO filesystem tools
    customTools: [kbSearchTool, ticketsQueryTool, webSearchTool, routeToAgent],
  });

  // Share the (expensive) ModelRuntime + supervisor model with child sessions.
  configureRouteToAgent({ modelRuntime, supervisorModel: model });

  return new SupportRuntimeImpl(session, modelRuntime, modelLabel, chatId);
}
