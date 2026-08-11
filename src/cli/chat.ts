/**
 * npm run chat — interactive support-agent REPL.
 *
 * Usage:
 *   npm run chat                 → interactive readline REPL
 *   npm run chat -- "question"   → one-shot demo mode (prints answer, exits)
 *
 * Streams tokens live (subscribes to text_delta), shows tool activity, and
 * prints a sources summary after each turn. Works end-to-end with the mock
 * retrieval — no DB needed.
 */

import "dotenv/config";
import readline from "node:readline/promises";
import process from "node:process";
import type { AgentSessionEvent } from "@earendil-works/pi-coding-agent";
import { createSupportRuntime } from "../runtime/session.js";
import type { SupportRuntimeImpl } from "../runtime/session.js";

const CYAN = "\x1b[36m";
const DIM = "\x1b[2m";
const GREEN = "\x1b[32m";
const YELLOW = "\x1b[33m";
const RED = "\x1b[31m";
const RESET = "\x1b[0m";

function fmtArgs(args: unknown): string {
  try {
    const s = JSON.stringify(args);
    return s && s.length > 120 ? `${s.slice(0, 117)}…` : (s ?? "");
  } catch {
    return String(args);
  }
}

/** Attach event printing; returns an unsubscribe fn. */
function attachPrinter(runtime: SupportRuntimeImpl): () => void {
  let toolDepth = 0;
  return runtime.subscribe((raw) => {
    const e = raw as AgentSessionEvent;
    switch (e.type) {
      case "turn_start":
        process.stdout.write(`\n${DIM}⟳ turn${RESET}\n`);
        break;
      case "message_update":
        if (e.assistantMessageEvent.type === "text_delta") {
          process.stdout.write(e.assistantMessageEvent.delta);
        }
        break;
      case "tool_execution_start": {
        toolDepth += 1;
        const pad = "  ".repeat(toolDepth);
        process.stdout.write(
          `\n${pad}${YELLOW}⚙ [${e.toolName}]${RESET} ${DIM}${fmtArgs(e.args)}${RESET}\n`,
        );
        break;
      }
      case "tool_execution_end": {
        process.stdout.write(
          `${"  ".repeat(toolDepth)}${e.isError ? RED : GREEN}✓ ${e.toolName} ${e.isError ? "failed" : "ok"}${RESET}\n`,
        );
        toolDepth = Math.max(0, toolDepth - 1);
        break;
      }
      case "agent_settled":
        process.stdout.write(`${DIM}— agent settled —${RESET}\n`);
        break;
      case "auto_retry_start":
        process.stdout.write(`${DIM}↻ retry ${e.attempt}/${e.maxAttempts}${RESET}\n`);
        break;
    }
  });
}

interface ToolActivity {
  tool: string;
  args: unknown;
}

function collectActivity(runtime: SupportRuntimeImpl): { done(): ToolActivity[] } {
  const calls: ToolActivity[] = [];
  const unsub = runtime.subscribe((raw) => {
    const e = raw as AgentSessionEvent;
    if (e.type === "tool_execution_start") calls.push({ tool: e.toolName, args: e.args });
  });
  return { done: () => { unsub(); return calls; } };
}

function collectSources(runtime: SupportRuntimeImpl): Array<{ title?: string; url?: string | null }> {
  const sources: Array<{ title?: string; url?: string | null }> = [];
  for (const m of runtime.getLastMessages()) {
    const msg = m as { details?: { sources?: unknown[] } };
    if (Array.isArray(msg.details?.sources)) {
      for (const s of msg.details.sources) {
        const src = s as { title?: string; url?: string | null };
        if (src.title || src.url) sources.push(src);
      }
    }
  }
  return sources;
}

async function runOne(runtime: SupportRuntimeImpl, text: string): Promise<void> {
  const activity = collectActivity(runtime);
  try {
    await runtime.promptWithBudget(text);
  } catch (err) {
    process.stdout.write(`\n${RED}✗ ${(err as Error).message}${RESET}\n`);
  }
  const calls = activity.done();
  if (calls.length > 0) {
    process.stdout.write(
      `\n${DIM}tools called: ${calls.map((c) => c.tool).join(", ")}${RESET}\n`,
    );
  }
  const sources = collectSources(runtime);
  if (sources.length > 0) {
    process.stdout.write(`\n${CYAN}sources:${RESET}\n`);
    const seen = new Set<string>();
    for (const s of sources) {
      const key = `${s.title ?? ""}|${s.url ?? ""}`;
      if (seen.has(key)) continue;
      seen.add(key);
      process.stdout.write(`${DIM}  • ${s.title ?? "(untitled)"}${s.url ? ` — ${s.url}` : ""}${RESET}\n`);
    }
  }
}

async function main(): Promise<void> {
  process.stdout.write(`${CYAN}agentic-customer-support — support agent REPL${RESET}\n`);
  process.stdout.write(
    `${DIM}retrieval: ${process.env.RETRIEVAL_MODE ?? "mock (no DB)"}`
    + ` | sql: ${process.env.SQL_MODE ?? "mock"}`
    + ` | web: ${process.env.TAVILY_API_KEY ? "tavily" : "duckduckgo"}${RESET}\n`,
  );
  process.stdout.write(`${DIM}commands: exit / quit / bye · /reset (new session)${RESET}\n\n`);

  let runtime: SupportRuntimeImpl | null = null;
  let unsubscribe: (() => void) | null = null;

  const start = async (): Promise<SupportRuntimeImpl> => {
    const r = await createSupportRuntime();
    unsubscribe?.();
    unsubscribe = attachPrinter(r);
    runtime = r;
    process.stdout.write(`${GREEN}▶ ready (model: ${r.modelLabel})${RESET}\n`);
    return r;
  };

  let rt = await start();

  const args = process.argv.slice(2);
  if (args.length > 0) {
    // one-shot demo mode
    await runOne(rt, args.join(" "));
    rt.dispose();
    return;
  }

  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  const onSigint = () => {
    process.stdout.write(`\n${DIM}bye${RESET}\n`);
    runtime?.dispose();
    process.exit(0);
  };
  process.on("SIGINT", onSigint);

  // Handle one user line: returns false when the loop should exit.
  const handleLine = async (raw: string): Promise<boolean> => {
    const text = raw.trim();
    if (!text) return true;
    if (/^(exit|quit|bye)$/i.test(text)) return false;
    if (text === "/reset") {
      rt.dispose();
      process.stdout.write(`${DIM}— new session —${RESET}\n`);
      rt = await start();
      return true;
    }
    await runOne(rt, text);
    process.stdout.write(`\n\n`);
    return true;
  };

  if (!process.stdin.isTTY) {
    // Piped input (e.g. `printf ... | npm run chat`): read everything first,
    // because readline closes the interface as soon as the pipe hits EOF.
    const { readFileSync } = await import("node:fs");
    const all = readFileSync(0, "utf8");
    for (const line of all.split("\n")) {
      if (!(await handleLine(line))) break;
    }
  } else {
    for (;;) {
      let line: string;
      try {
        line = await rl.question(`${GREEN}you>${RESET} `);
      } catch {
        break; // stdin closed (EOF / Ctrl-D)
      }
      if (line === undefined) break;
      if (!(await handleLine(line))) break;
    }
  }

  process.off("SIGINT", onSigint);
  rl.close();
  rt.dispose();
  process.stdout.write(`${DIM}bye${RESET}\n`);
}

main().catch((err) => {
  process.stderr.write(`${RED}fatal:${RESET} ${(err as Error).stack ?? err}\n`);
  process.exit(1);
});
