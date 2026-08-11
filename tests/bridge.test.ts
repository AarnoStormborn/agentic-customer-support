/**
 * Bridge tests — SDK event → SSE payload mapping (design §3.4 / §2.3 schema).
 */
import { describe, it, expect, vi } from "vitest";
import { attachBridge, mapPromptError, type ChatEventSink } from "../src/streaming/bridge.js";

function makeHarness() {
  const emitted: { type: string; data: unknown }[] = [];
  const sink: ChatEventSink = {
    emit: vi.fn((type: string, data: unknown) => emitted.push({ type, data })),
  };
  let handler: ((e: unknown) => void) | null = null;
  const session = {
    subscribe: vi.fn((fn: (e: unknown) => void) => {
      handler = fn;
      return () => {
        handler = null;
      };
    }),
    getLastMessages: vi.fn(() => []),
  };
  const detach = attachBridge(session as never, sink, {
    chatId: "chat_1",
    conversationId: "conv_1",
  });
  return { emitted, sink, fire: (e: unknown) => handler?.(e), detach, session };
}

describe("attachBridge (SDK → SSE mapping)", () => {
  it("maps turn_start", () => {
    const h = makeHarness();
    h.fire({ type: "turn_start", turnIndex: 1 });
    expect(h.emitted[0]!.type).toBe("turn_start");
    expect(h.emitted[0]!.data).toMatchObject({ chatId: "chat_1", turnIndex: 1 });
  });

  it("maps text_delta to token and thinking_delta to thinking", () => {
    const h = makeHarness();
    h.fire({ type: "message_update", assistantMessageEvent: { type: "text_delta", delta: "Hel" } });
    h.fire({ type: "message_update", assistantMessageEvent: { type: "thinking_delta", delta: "hmm" } });
    expect(h.emitted[0]).toMatchObject({ type: "token", data: { delta: "Hel" } });
    expect(h.emitted[1]).toMatchObject({ type: "thinking", data: { delta: "hmm" } });
  });

  it("maps tool_execution_start/end", () => {
    const h = makeHarness();
    h.fire({ type: "tool_execution_start", toolCallId: "c1", toolName: "kb_search", args: { query: "x" } });
    h.fire({ type: "tool_execution_end", toolCallId: "c1", toolName: "kb_search", isError: false, durationMs: 5 });
    expect(h.emitted[0]!.type).toBe("tool_start");
    expect(h.emitted[1]!.type).toBe("tool_end");
    expect(h.emitted[1]!.data).toMatchObject({ toolName: "kb_search", isError: false });
  });

  it("maps agent_settled to done with sources + final message", () => {
    const h = makeHarness();
    const sources = [{ type: "kb", title: "lg.pdf" }];
    h.fire({ type: "agent_settled", sources });
    const done = h.emitted[0]!;
    expect(done.type).toBe("done");
    expect((done.data as { sources: unknown[] }).sources).toEqual(sources);
    expect(done.data).toMatchObject({ chatId: "chat_1", conversationId: "conv_1" });
  });

  it("ignores unknown events", () => {
    const h = makeHarness();
    h.fire({ type: "something_new" });
    expect(h.emitted).toHaveLength(0);
  });

  it("detach stops forwarding", () => {
    const h = makeHarness();
    h.detach();
    h.fire({ type: "turn_start" });
    expect(h.emitted).toHaveLength(0);
  });
});

describe("mapPromptError", () => {
  it("maps known codes and defaults unknown errors to internal", () => {
    expect(mapPromptError(new Error("canceled")).code).toBe("internal"); // generic message → internal
    expect(mapPromptError({ code: "canceled", message: "cancelled" })).toMatchObject({
      code: "canceled",
      retryable: false,
    });
    expect(mapPromptError(new Error("ECONNRESET")).code).toBeTruthy();
  });
});
