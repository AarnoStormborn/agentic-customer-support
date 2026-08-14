/**
 * stores/chatStore.ts — chat state + SSE reducer.
 *
 * The reducer core (`reduceSseEvent`) is a pure function: given a state and one
 * SSE event, return the next state. Tests exercise it directly (fast, no
 * network); the zustand actions wrap it with `set()`.
 *
 * Streaming model: one optimistic user bubble + one assistant bubble per send.
 * `token` events append to the open streaming bubble; `done` replaces its text
 * with the authoritative final message and attaches sources[].
 */
import { create } from "zustand";
import { api } from "../lib/api";
import { useSettingsStore } from "./settingsStore";
import type {
  ChatHistoryResponse,
  DoneUsage,
  SdkMessage,
  SourceRef,
  SseEventMap,
  SseEventType,
} from "../lib/types";
import { truncate } from "../lib/format";

export type MessageStatus = "pending" | "sent" | "streaming" | "done" | "error" | "cancelled";

export interface ChatMessage {
  /** Local client id (optimistic bubbles) */
  id: string;
  role: "user" | "assistant";
  text: string;
  status: MessageStatus;
  turnIndex?: number;
  sources?: SourceRef[];
  usage?: DoneUsage;
  error?: { code: string; message: string; retryable: boolean };
  createdAt: number;
}

export interface ToolActivity {
  toolCallId: string;
  toolName: string;
  args: Record<string, unknown>;
  status: "running" | "done" | "error";
  turnIndex: number;
  startedAt: number;
  durationMs?: number;
  summary?: string | null;
  partial?: string;
}

export interface ChatErrorInfo {
  code: string;
  message: string;
  retryable: boolean;
}

let uid = 0;
function nextId(): string {
  uid += 1;
  return `m${Date.now().toString(36)}_${uid}`;
}

/** Data-only slice (no actions) — what the pure reducer operates on. */
export interface ChatData {
  activeChatId: string | null;
  conversationId: string | null;
  messages: ChatMessage[];
  toolActivities: ToolActivity[];
  statusLine: string | null;
  isStreaming: boolean;
  /** Set when we should open an SSE stream for activeChatId. */
  shouldStream: boolean;
  connectionState: "idle" | "connecting" | "open" | "reconnecting" | "closed";
  lastError: ChatErrorInfo | null;
  /** Composer prefill set from Tickets/Manuals routes ("Ask the agent…"). */
  pendingDraft: string | null;
}

export interface ChatState extends ChatData {
  // actions
  newSession: () => void;
  /** Optimistic part of send — pure state, no fetch. */
  startTurn: (chatId: string | null, conversationId: string | null, userText: string) => void;
  send: (text: string, opts?: { ticketId?: number }) => Promise<void>;
  openSession: (chatId: string, conversationId: string, running: boolean) => Promise<void>;
  stopStreaming: () => Promise<void>;
  steer: (text: string) => Promise<void>;
  setConnectionState: (s: ChatState["connectionState"]) => void;
  applySseEvent: (event: SseEventType, data: SseEventMap[SseEventType]) => void;
  appendTokens: (delta: string) => void;
  appendThinking: (delta: string) => void;
  prefillDraft: (text: string) => void;
  consumeDraft: () => string | null;
}

// ---------------------------------------------------------------------------
// Pure reducer (exported for unit tests)
// ---------------------------------------------------------------------------

export function reduceSseEvent(
  state: ChatData,
  event: SseEventType,
  data: unknown,
): ChatData {
  switch (event) {
    case "turn_start":
      return { ...state, statusLine: null };

    case "token": {
      const d = data as SseEventMap["token"];
      return appendText(state, d.delta, d.turnIndex);
    }

    case "thinking":
      return { ...state }; // thinking deltas are not shown as chat text (v1)

    case "tool_start":
      return addTool(state, data as SseEventMap["tool_start"]);

    case "tool_update": {
      const d = data as SseEventMap["tool_update"];
      return patchTool(state, d.toolCallId, { partial: d.partial });
    }

    case "tool_end":
      return endTool(state, data as SseEventMap["tool_end"]);

    case "turn_end":
      return { ...state, statusLine: null };

    case "done":
      return finalize(state, data as SseEventMap["done"]);

    case "error":
      return markError(state, data as SseEventMap["error"]);

    case "retry_start":
      return { ...state, statusLine: "Provider hiccup — retrying…" };

    case "retry_end":
      return { ...state, statusLine: null };

    case "queue_update":
      return { ...state };

    default:
      return state;
  }
}

function appendText(state: ChatData, delta: string, turnIndex?: number): ChatData {
  if (!delta) return state;
  const messages = [...state.messages];
  const last = messages[messages.length - 1];

  if (last && last.role === "assistant" && last.status === "streaming") {
    messages[messages.length - 1] = {
      ...last,
      text: last.text + delta,
      turnIndex: turnIndex ?? last.turnIndex,
    };
  } else {
    // New burst (e.g. after a tool round) → continue into a fresh bubble.
    messages.push(newMessage("assistant", delta, { status: "streaming", turnIndex }));
  }
  return { ...state, messages };
}

function addTool(
  state: ChatData,
  data: SseEventMap["tool_start"],
): ChatData {
  const activity: ToolActivity = {
    toolCallId: data.toolCallId,
    toolName: data.toolName,
    args: data.args ?? {},
    status: "running",
    turnIndex: data.turnIndex,
    startedAt: Date.now(),
  };
  const toolActivities = [activity, ...state.toolActivities.filter(
    (t) => t.toolCallId !== data.toolCallId,
  )];
  return { ...state, toolActivities, statusLine: labelForTool(data.toolName, data.args) };
}

function patchTool(
  state: ChatData,
  toolCallId: string,
  patch: Partial<ToolActivity>,
): ChatData {
  const toolActivities = state.toolActivities.map((t) =>
    t.toolCallId === toolCallId ? { ...t, ...patch } : t,
  );
  return { ...state, toolActivities };
}

function endTool(state: ChatData, data: SseEventMap["tool_end"]): ChatData {
  const toolActivities = state.toolActivities.map((t) =>
    t.toolCallId === data.toolCallId
      ? {
          ...t,
          status: data.isError ? ("error" as const) : ("done" as const),
          durationMs: data.durationMs,
          summary: data.summary,
        }
      : t,
  );
  // Clear the status line when no tool is running anymore.
  const anyRunning = toolActivities.some((t) => t.status === "running");
  return { ...state, toolActivities, statusLine: anyRunning ? state.statusLine : null };
}

function finalize(state: ChatData, data: SseEventMap["done"]): ChatData {
  const messages = [...state.messages];
  const last = messages[messages.length - 1];
  if (last && last.role === "assistant") {
    messages[messages.length - 1] = {
      ...last,
      // Authoritative final text (streamed tokens may be partial on reconnect).
      text: data.message || last.text,
      status: "done",
      sources: data.sources ?? [],
      usage: data.usage ?? {},
      turnIndex: data.turnIndex ?? last.turnIndex,
    };
  }
  return {
    ...state,
    messages,
    isStreaming: false,
    shouldStream: false,
    statusLine: null,
    connectionState: "closed",
    conversationId: data.conversationId || state.conversationId,
  };
}

function markError(state: ChatData, data: SseEventMap["error"]): ChatData {
  const messages = [...state.messages];
  const last = messages[messages.length - 1];
  if (last && last.role === "assistant" && last.status === "streaming") {
    messages[messages.length - 1] = {
      ...last,
      status: data.code === "canceled" ? "cancelled" : "error",
      error: { code: data.code, message: data.message, retryable: data.retryable },
    };
  }
  return {
    ...state,
    messages,
    isStreaming: false,
    shouldStream: false,
    statusLine: null,
    connectionState: "closed",
    lastError: { code: data.code, message: data.message, retryable: data.retryable },
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function newMessage(
  role: "user" | "assistant",
  text: string,
  extra: Partial<ChatMessage> = {},
): ChatMessage {
  return {
    id: nextId(),
    role,
    text,
    status: role === "user" ? "sent" : "streaming",
    createdAt: Date.now(),
    ...extra,
  };
}

function labelForTool(toolName: string, args: Record<string, unknown>): string {
  if (toolName === "route_to_agent") {
    const agent = String(args?.agent ?? "");
    if (agent.startsWith("rag")) return "RAG agent: searching manuals…";
    if (agent.startsWith("sql")) return "SQL agent: querying tickets…";
    if (agent.startsWith("web")) return "Web agent: fetching results…";
    return `Sub-agent (${agent}): working…`;
  }
  if (toolName === "kb_search") return "RAG agent: searching manuals…";
  if (toolName === "tickets_query") return "SQL agent: querying tickets…";
  if (toolName === "web_search") return "Web agent: fetching results…";
  return `${toolName}: working…`;
}

/** Map pi SDK history messages to client ChatMessages (GET /chat/:id/history). */
export function toClientMessages(history: SdkMessage[]): ChatMessage[] {
  const out: ChatMessage[] = [];
  for (const m of history) {
    const text =
      typeof m.content === "string"
        ? m.content
        : Array.isArray(m.content)
          ? m.content
              .filter((b) => b.type === "text" && typeof b.text === "string")
              .map((b) => b.text as string)
              .join("")
          : "";
    if (!text) continue;
    out.push({
      id: nextId(),
      role: m.role === "user" ? "user" : "assistant",
      text,
      status: "done",
      createdAt: Date.now(),
    });
  }
  return out;
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

export const useChatStore = create<ChatState>()((set, get) => ({
  activeChatId: null,
  conversationId: null,
  messages: [],
  toolActivities: [],
  statusLine: null,
  isStreaming: false,
  shouldStream: false,
  connectionState: "idle",
  lastError: null,
  pendingDraft: null,

  newSession: () =>
    set({
      activeChatId: null,
      conversationId: null,
      messages: [],
      toolActivities: [],
      statusLine: null,
      isStreaming: false,
      shouldStream: false,
      connectionState: "idle",
      lastError: null,
    }),

  startTurn: (chatId, conversationId, userText) =>
    set((s) => ({
      activeChatId: chatId ?? s.activeChatId,
      conversationId: conversationId ?? s.conversationId,
      messages: [
        ...s.messages,
        newMessage("user", userText),
        newMessage("assistant", ""),
      ],
      isStreaming: true,
      shouldStream: true,
      connectionState: "connecting",
      statusLine: "Sending…",
      lastError: null,
    })),

  send: async (text, opts) => {
    const { conversationId, activeChatId } = get();
    // Follow-up messages reuse the same conversationId; new chats start fresh.
    const convId = activeChatId ? conversationId : undefined;
    get().startTurn(activeChatId, convId ?? null, text);

    try {
      const res = await api.startChat({
        message: text,
        conversationId: convId ?? undefined,
        ticketId: opts?.ticketId,
        retrieval: { ...useSettingsStore.getState().retrieval },
      });
      set({
        activeChatId: res.chatId,
        conversationId: res.conversationId,
        connectionState: "connecting",
        lastError: null,
      });
    } catch (err) {
      set((s) => {
        const messages = [...s.messages];
        const last = messages[messages.length - 1];
        if (last && last.role === "assistant") {
          messages[messages.length - 1] = {
            ...last,
            status: "error",
            error: {
              code: "request_failed",
              message: err instanceof Error ? err.message : "Request failed",
              retryable: true,
            },
          };
        }
        return {
          messages,
          isStreaming: false,
          shouldStream: false,
          connectionState: "closed",
          lastError: {
            code: "request_failed",
            message: err instanceof Error ? err.message : "Request failed",
            retryable: true,
          },
        };
      });
    }
  },

  openSession: async (chatId, conversationId, running) => {
    set((s) => ({
      ...s,
      activeChatId: chatId,
      conversationId,
      messages: [],
      toolActivities: [],
      statusLine: null,
      isStreaming: running,
      shouldStream: running,
      connectionState: running ? "connecting" : "idle",
      lastError: null,
    }));
    try {
      const h: ChatHistoryResponse = await api.chatHistory(chatId);
      set((s) => ({
        ...s,
        messages: toClientMessages(h.messages),
        conversationId: h.conversationId || conversationId,
      }));
    } catch (err) {
      set({
        lastError: {
          code: "history_failed",
          message: err instanceof Error ? err.message : "Failed to load history",
          retryable: true,
        },
      });
    }
  },

  stopStreaming: async () => {
    const { activeChatId } = get();
    if (activeChatId) {
      // Fire-and-forget cancel; the backend closes the SSE with error canceled.
      void api.cancelChat(activeChatId).catch(() => {});
    }
    // Mark locally cancelled immediately (don't wait on the network).
    set((s) => {
      const messages = [...s.messages];
      const last = messages[messages.length - 1];
      if (last && last.role === "assistant" && last.status === "streaming") {
        messages[messages.length - 1] = { ...last, status: "cancelled" };
      }
      return {
        messages,
        isStreaming: false,
        shouldStream: false,
        statusLine: null,
        lastError: null,
      };
    });
  },

  steer: async (text) => {
    const { activeChatId } = get();
    if (!activeChatId) return;
    try {
      await api.steerChat(activeChatId, text);
    } catch {
      // steer is best-effort; surface nothing unless there's an active error
    }
  },

  setConnectionState: (connectionState) => set({ connectionState }),

  applySseEvent: (event, data) => set((s) => reduceSseEvent(s, event, data)),

  // rAF-batched token flush (see useChatStream) — one store update per frame.
  appendTokens: (delta) => set((s) => appendText(s, delta)),

  appendThinking: (delta) => {
    // v1: thinking is tracked but not rendered as chat text.
    void delta;
  },

  prefillDraft: (text) => set({ pendingDraft: text }),

  consumeDraft: () => {
    const draft = get().pendingDraft;
    if (draft !== null) set({ pendingDraft: null });
    return draft;
  },
}));

export function sessionPreview(messages: ChatMessage[]): string {
  const firstUser = messages.find((m) => m.role === "user");
  return firstUser ? truncate(firstUser.text, 60) : "New session";
}
