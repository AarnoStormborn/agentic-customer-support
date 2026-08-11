/**
 * chatStore.test.ts — streaming reducer logic: optimistic bubbles, token
 * append, tool activity, done finalize, error handling (no network).
 */
import { beforeEach, describe, expect, it } from "vitest";
import { reduceSseEvent, useChatStore } from "./chatStore";
import type { ChatData, ToolActivity } from "./chatStore";

function freshState(): ChatData {
  return {
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
  };
}

const token = (delta: string, turnIndex = 1) => ({ chatId: "c", turnIndex, delta });
const done = {
  chatId: "c",
  conversationId: "conv_1",
  turnIndex: 1,
  message: "Full answer.",
  sources: [{ type: "kb", title: "lg-manual.pdf", score: 0.91 }],
  usage: { inputTokens: 10, outputTokens: 5 },
};
const err = { chatId: "c", code: "provider_error" as const, message: "boom", retryable: true };

describe("chatStore streaming reducer", () => {
  beforeEach(() => {
    useChatStore.setState(freshState());
  });

  it("startTurn adds optimistic user + empty assistant bubbles", () => {
    useChatStore.getState().startTurn(null, null, "my tv won't connect");
    const { messages, isStreaming, shouldStream } = useChatStore.getState();

    expect(messages).toHaveLength(2);
    expect(messages[0]).toMatchObject({ role: "user", text: "my tv won't connect", status: "sent" });
    expect(messages[1]).toMatchObject({ role: "assistant", text: "", status: "streaming" });
    expect(isStreaming).toBe(true);
    expect(shouldStream).toBe(true);
  });

  it("token events append to the streaming bubble", () => {
    let state = freshState();
    state = reduceSseEvent(state, "token", token("Let me "));
    state = reduceSseEvent(state, "token", token("check"));

    expect(state.messages).toHaveLength(1);
    expect(state.messages[0]).toMatchObject({
      role: "assistant",
      text: "Let me check",
      status: "streaming",
    });
  });

  it("tokens continue the same bubble after a tool round", () => {
    let state = freshState();
    state = reduceSseEvent(state, "token", token("Searching…", 1));
    state = reduceSseEvent(state, "tool_start", {
      chatId: "c", turnIndex: 1, toolCallId: "t1", toolName: "kb_search", args: { query: "wifi" },
    });
    state = reduceSseEvent(state, "tool_end", {
      chatId: "c", turnIndex: 1, toolCallId: "t1", toolName: "kb_search",
      isError: false, durationMs: 5, summary: "2 chunks",
    });
    // Turn 2 tokens belong to the same user request → same bubble.
    state = reduceSseEvent(state, "token", token("Here's the fix", 2));

    expect(state.messages).toHaveLength(1);
    expect(state.messages[0]?.text).toBe("Searching…Here's the fix");
  });

  it("tool_start/end update the activity feed + status line", () => {
    let state = freshState();
    state = reduceSseEvent(state, "tool_start", {
      chatId: "c", turnIndex: 1, toolCallId: "t1", toolName: "route_to_agent",
      args: { agent: "rag", query: "wifi" },
    });
    expect(state.statusLine).toBe("RAG agent: searching manuals…");
    expect(state.toolActivities[0]).toMatchObject({
      toolCallId: "t1", toolName: "route_to_agent", status: "running",
    });

    state = reduceSseEvent(state, "tool_end", {
      chatId: "c", turnIndex: 1, toolCallId: "t1", toolName: "route_to_agent",
      isError: false, durationMs: 412, summary: "3 chunks retrieved",
    });
    const t = state.toolActivities[0] as ToolActivity;
    expect(t.status).toBe("done");
    expect(t.durationMs).toBe(412);
    // No tool running anymore → status line cleared.
    expect(state.statusLine).toBeNull();
  });

  it("tool errors mark the card as error", () => {
    let state = freshState();
    state = reduceSseEvent(state, "tool_start", {
      chatId: "c", turnIndex: 1, toolCallId: "t1", toolName: "tickets_query", args: {},
    });
    state = reduceSseEvent(state, "tool_end", {
      chatId: "c", turnIndex: 1, toolCallId: "t1", toolName: "tickets_query",
      isError: true, durationMs: 20, summary: null,
    });
    expect(state.toolActivities[0]?.status).toBe("error");
  });

  it("done finalizes the bubble with the authoritative message + sources", () => {
    let state = freshState();
    state = reduceSseEvent(state, "token", token("Streamed partial…"));
    state = reduceSseEvent(state, "done", done);

    const m = state.messages[0];
    expect(m?.status).toBe("done");
    expect(m?.text).toBe("Full answer.");
    expect(m?.sources).toEqual(done.sources);
    expect(m?.usage).toEqual(done.usage);
    expect(state.isStreaming).toBe(false);
    expect(state.shouldStream).toBe(false);
    expect(state.conversationId).toBe("conv_1");
  });

  it("error keeps the partial text and marks the bubble", () => {
    let state = freshState();
    state = reduceSseEvent(state, "token", token("Partial answer"));
    state = reduceSseEvent(state, "error", err);

    const m = state.messages[0];
    expect(m?.status).toBe("error");
    expect(m?.text).toBe("Partial answer");
    expect(m?.error).toEqual({ code: "provider_error", message: "boom", retryable: true });
    expect(state.isStreaming).toBe(false);
    expect(state.lastError).toMatchObject({ code: "provider_error", retryable: true });
  });

  it("error with code canceled marks the bubble cancelled", () => {
    let state = freshState();
    state = reduceSseEvent(state, "token", token("Half"));
    state = reduceSseEvent(state, "error", { ...err, code: "canceled" });
    expect(state.messages[0]?.status).toBe("cancelled");
  });

  it("stopStreaming marks locally cancelled without waiting on the network", () => {
    useChatStore.getState().startTurn(null, null, "q");
    useChatStore.getState().stopStreaming();
    const s = useChatStore.getState();
    expect(s.messages[1]?.status).toBe("cancelled");
    expect(s.isStreaming).toBe(false);
  });

  it("prefillDraft → consumeDraft round-trips once", () => {
    const store = useChatStore.getState();
    store.prefillDraft("Ask about chunk 3");
    expect(useChatStore.getState().pendingDraft).toBe("Ask about chunk 3");
    expect(useChatStore.getState().consumeDraft()).toBe("Ask about chunk 3");
    expect(useChatStore.getState().pendingDraft).toBeNull();
  });
});
