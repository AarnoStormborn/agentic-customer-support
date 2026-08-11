/**
 * SDK spike — verify @earendil-works/pi-coding-agent works in-process:
 * 1. ModelRuntime resolves an available model
 * 2. createAgentSession with a custom tool (both paths: customTools option + extension factory)
 * 3. Verify tool exposure via session.agent.state.tools
 * 4. prompt() runs, events stream, tool gets called, result returned
 */
import "dotenv/config";
import {
  createAgentSession,
  defineTool,
  getAgentDir,
  ModelRuntime,
  SessionManager,
  SettingsManager,
  DefaultResourceLoader,
} from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

const modelRuntime = await ModelRuntime.create();

const available = await modelRuntime.getAvailable();
console.log(`Available models: ${available.length}`);
if (available.length === 0) {
  console.error("No authenticated models found. Configure ~/.pi/agent/auth.json or env API keys first.");
  process.exit(1);
}

// Prefer a well-behaved model; fall back to env override then first available
const envModel = process.env.PI_MODEL;
const preferred = [
  "anthropic/claude-sonnet-4-5",
  "anthropic/claude-haiku-4-5",
  "openai/gpt-5",
  "google/gemini-3-pro",
];
const chosen =
  (envModel
    ? available.find((m) => `${m.provider}/${m.id}` === envModel)
    : undefined) ??
  available.find((m) => preferred.some((p) => `${m.provider}/${m.id}`.startsWith(p))) ??
  available[0];
if (!chosen) {
  console.error("No usable model found.");
  process.exit(1);
}
console.log(`Using: ${chosen.provider}/${chosen.id}`);

// --- Path 1: customTools option ---
const echoTool = defineTool({
  name: "echo_support",
  label: "Echo Support",
  description: "Echoes the user's support question back (spike tool).",
  parameters: Type.Object({ question: Type.String({ description: "The question" }) }),
  execute: async (_id, params) => ({
    content: [{ type: "text", text: `ECHO: ${params.question}` }],
    details: { source: "spike" },
  }),
});

// --- Path 2: extension factory registering a tool ---
const kbTool = (pi: any) => {
  pi.registerTool({
    name: "kb_search",
    label: "KB Search",
    description: "Search the knowledge base (spike tool).",
    parameters: Type.Object({ query: Type.String({ description: "Search query" }) }),
    execute: async (_toolCallId: string, params: { query: string }) => ({
      content: [{ type: "text", text: `KB result for: ${params.query}` }],
      details: { source: "spike" },
    }),
  });
};

const loader = new DefaultResourceLoader({
  cwd: process.cwd(),
  agentDir: getAgentDir(),
  extensionFactories: [kbTool],
  systemPromptOverride: () =>
    "You are a spike assistant for a customer support system. " +
    "You have two tools: echo_support (echo a question) and kb_search (search knowledge base). " +
    "When asked, USE the appropriate tool and report its output. Be concise.",
});
await loader.reload();

const { session } = await createAgentSession({
  cwd: process.cwd(),
  model: chosen,
  thinkingLevel: "off",
  modelRuntime,
  resourceLoader: loader,
  customTools: [echoTool],
  tools: ["echo_support", "kb_search"], // ONLY custom tools — no bash/read/write
  sessionManager: SessionManager.inMemory(process.cwd()),
  settingsManager: SettingsManager.inMemory({ compaction: { enabled: false } }),
});

// Diagnostic: what tools does the agent actually see?
console.log("\n--- tools exposed to agent ---");
const exposed = session.agent.state.tools.map((t) => t.name);
console.log(exposed);

const events: string[] = [];
session.subscribe((event) => {
  const e = event as any;
  if (e.type === "message_update" && e.assistantMessageEvent?.type === "text_delta") {
    events.push(`token:${e.assistantMessageEvent.delta}`);
  }
  if (e.type === "tool_execution_start") events.push(`tool_start:${e.toolName}`);
  if (e.type === "tool_execution_end") events.push(`tool_end:${e.isError ? "err" : "ok"}`);
  if (e.type === "agent_end") events.push("agent_end");
});

await session.prompt("Use kb_search with query: 'LG TV wifi reset'");

const text = events.filter((x) => x.startsWith("token:")).map((x) => x.slice(6)).join("");
console.log("\n--- event trace (non-token) ---");
console.log(events.filter((x) => !x.startsWith("token:")).join("\n") || "(none)");
console.log("\n--- assistant text ---");
console.log(text.slice(0, 500) || "(empty)");

console.log("\n--- last messages ---");
for (const m of session.messages.slice(-6)) {
  const parts = (m as any).content?.map((p: any) =>
    p.type === "text" ? `text:"${(p.text ?? "").slice(0, 80)}"` : `[${p.type}]`
  );
  console.log(`${m.role}: ${parts?.join(" ") ?? ""}`);
}
session.dispose();
console.log("\nSPIKE OK");
