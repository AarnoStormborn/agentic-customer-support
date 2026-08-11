/**
 * src/streaming/bridge.ts — SDK event → SSE event mapper (§3.4 of the design doc).
 *
 * One bridge per chat turn. It subscribes to the SupportRuntime's raw SDK events and
 * pushes typed SSE events (§2.3 schema) into a sink — in practice the ChatRegistry
 * bound to that chat id. This is the ONLY module that maps SDK events to SSE events
 * (architecture rule 4).
 */
import type { SupportRuntime } from "../runtime/mock.js";
import type { SSEEventType } from "./registry.js";

/** §2.3 payload shapes. */
export type TurnStartPayload = {
  chatId: string;
  turnIndex: number;
  ts: number;
};
export type TokenPayload = { chatId: string; turnIndex: number; delta: string };
export type ToolStartPayload = {
  chatId: string;
  turnIndex: number;
  toolCallId: string;
  toolName: string;
  args: Record<string, unknown>;
};
export type ToolUpdatePayload = {
  chatId: string;
  turnIndex: number;
  toolCallId: string;
  partial: string;
};
export type ToolEndPayload = {
  chatId: string;
  turnIndex: number;
  toolCallId: string;
  toolName: string;
  isError: boolean;
  durationMs: number;
  summary: string | null;
};
export type TurnEndPayload = { chatId: string; turnIndex: number; ts: number };
export type DonePayload = {
  chatId: string;
  conversationId: string;
  turnIndex: number;
  message: string;
  sources: SourceRef[];
  usage: { inputTokens?: number; outputTokens?: number; totalCostUsd?: number };
};
export type ErrorPayload = {
  chatId: string;
  code: ErrorCode;
  message: string;
  retryable: boolean;
};
export type QueueUpdatePayload = { chatId: string; steering: number; followUp: number };

/** "turn_timeout" | "canceled" | "provider_error" | "guardrail_blocked" | "internal" */
export type ErrorCode =
  | "turn_timeout"
  | "canceled"
  | "provider_error"
  | "guardrail_blocked"
  | "internal";

export interface SourceRef {
  type: "kb" | "sql" | "web";
  title?: string;
  docName?: string;
  sectionPath?: string;
  page?: number;
  url?: string | null;
  score?: number | null;
  row?: Record<string, unknown>;
}

/** What the bridge emits into — bound to a chat id by the caller. */
export interface ChatEventSink {
  emit(event: SSEEventType, data: unknown): void;
}

export interface BridgeContext {
  chatId: string;
  conversationId: string;
}

/** Any raw SDK event that isn't one of the mapped types is ignored. */
export function attachBridge(
  session: SupportRuntime,
  sink: ChatEventSink,
  ctx: BridgeContext,
): () => void {
  let turnIndex = 0;

  return session.subscribe((raw: unknown) => {
    const event = raw as { type?: string } & Record<string, unknown>;
    if (typeof event !== "object" || event === null || !event.type) return;

    switch (event.type) {
      case "turn_start": {
        turnIndex = asNumber(event.turnIndex, 1);
        sink.emit("turn_start", {
          chatId: ctx.chatId,
          turnIndex,
          ts: Date.now(),
        } satisfies TurnStartPayload);
        break;
      }

      case "message_update": {
        const m = event.assistantMessageEvent as
          | { type?: string; delta?: string }
          | undefined;
        if (!m?.type || typeof m.delta !== "string") break;
        if (m.type === "text_delta") {
          sink.emit("token", {
            chatId: ctx.chatId,
            turnIndex,
            delta: m.delta,
          } satisfies TokenPayload);
        } else if (m.type === "thinking_delta") {
          sink.emit("thinking", {
            chatId: ctx.chatId,
            turnIndex,
            delta: m.delta,
          } satisfies TokenPayload);
        }
        break;
      }

      case "tool_execution_start":
        sink.emit("tool_start", {
          chatId: ctx.chatId,
          turnIndex,
          toolCallId: String(event.toolCallId ?? ""),
          toolName: String(event.toolName ?? "unknown"),
          args: (event.args as Record<string, unknown>) ?? {},
        } satisfies ToolStartPayload);
        break;

      case "tool_execution_update":
        sink.emit("tool_update", {
          chatId: ctx.chatId,
          turnIndex,
          toolCallId: String(event.toolCallId ?? ""),
          partial: String(event.partialResult ?? ""),
        } satisfies ToolUpdatePayload);
        break;

      case "tool_execution_end":
        sink.emit("tool_end", {
          chatId: ctx.chatId,
          turnIndex,
          toolCallId: String(event.toolCallId ?? ""),
          toolName: String(event.toolName ?? "unknown"),
          isError: Boolean(event.isError),
          durationMs: asNumber(event.durationMs, 0),
          summary: typeof event.summary === "string" ? event.summary : null,
        } satisfies ToolEndPayload);
        break;

      case "turn_end":
        sink.emit("turn_end", {
          chatId: ctx.chatId,
          turnIndex,
          ts: Date.now(),
        } satisfies TurnEndPayload);
        break;

      case "queue_update":
        sink.emit("queue_update", {
          chatId: ctx.chatId,
          steering: asNumber(event.steering, 0),
          followUp: asNumber(event.followUp, 0),
        } satisfies QueueUpdatePayload);
        break;

      case "auto_retry_start":
        sink.emit("retry_start", { chatId: ctx.chatId, turnIndex });
        break;

      case "auto_retry_end":
        sink.emit("retry_end", { chatId: ctx.chatId, turnIndex });
        break;

      case "agent_settled":
        sink.emit("done", {
          chatId: ctx.chatId,
          conversationId: ctx.conversationId,
          turnIndex,
          message: extractFinalMessage(session),
          // Mock attaches `sources` to agent_settled; real runtime should too
          // (see CONTRACT-NOTES.md). Fall back to [] to keep the schema valid.
          sources: Array.isArray(event.sources)
            ? (event.sources as SourceRef[])
            : [],
          usage: (event.usage as DonePayload["usage"]) ?? {},
        } satisfies DonePayload);
        break;
    }
  });
}

/**
 * Map a `prompt()` rejection / runtime error to the §2.3 `error` payload. The SDK
 * surfaces an error with a `.code` (e.g. "canceled" from the mock on abort); anything
 * else is "internal" unless it smells like a provider timeout.
 */
export function mapPromptError(err: unknown): {
  code: ErrorCode;
  message: string;
  retryable: boolean;
} {
  const e = err as { code?: string; message?: string } | undefined;
  const message = e?.message ?? "Agent turn failed";
  const code = e?.code;
  if (code === "canceled" || code === "turn_timeout" || code === "provider_error" || code === "guardrail_blocked") {
    return { code, message, retryable: code === "provider_error" };
  }
  return { code: "internal", message, retryable: false };
}

/** Best-effort final assistant text from `session.getLastMessages()`. */
function extractFinalMessage(session: SupportRuntime): string {
  const messages = session.getLastMessages();
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const m = messages[i] as { role?: string; content?: unknown } | undefined;
    if (!m || m.role !== "assistant") continue;
    if (typeof m.content === "string") return m.content;
    if (Array.isArray(m.content)) {
      const text = m.content
        .map((block) => {
          const b = block as { type?: string; text?: string };
          return b?.type === "text" && typeof b.text === "string" ? b.text : "";
        })
        .join("");
      if (text) return text;
    }
  }
  return "";
}

function asNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}
