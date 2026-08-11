/**
 * sse.test.ts — connectSse dispatch + lifecycle with a fake EventSource
 * (no network, no happy-dom EventSource dependency).
 */
import { describe, expect, it, vi } from "vitest";
import { connectSse, type EventSourceLike } from "./sse";

/** Minimal fake: records listeners, lets tests trigger events. */
class FakeEventSource implements EventSourceLike {
  readyState = 0;
  closed = false;
  listeners = new Map<string, Array<(e: { data?: string }) => void>>();
  url: string;

  constructor(url: string) {
    this.url = url;
  }

  addEventListener(type: string, listener: (e: { data?: string }) => void): void {
    const arr = this.listeners.get(type) ?? [];
    arr.push(listener);
    this.listeners.set(type, arr);
  }

  close(): void {
    this.closed = true;
    this.readyState = 2;
  }

  // --- test helpers ---
  emit(type: string, data?: string): void {
    for (const l of this.listeners.get(type) ?? []) l({ data });
  }
}

/** Wrap the fake in a real (constructible) subclass that records its instance. */
function connectWithSpy(handlers: Parameters<typeof connectSse>[1]): {
  conn: ReturnType<typeof connectSse>;
  es: FakeEventSource;
} {
  let es: FakeEventSource | null = null;
  class RecordingCtor extends FakeEventSource {
    constructor(url: string) {
      super(url);
      es = this;
    }
  }
  const conn = connectSse("/x", handlers, RecordingCtor);
  if (!es) throw new Error("connectSse did not construct an EventSource");
  return { conn, es: es! };
}

const base = {
  chatId: "chat_1",
  turnIndex: 1,
};

describe("connectSse", () => {
  it("connects to the given URL", () => {
    let url = "";
    class Ctor extends FakeEventSource {
      constructor(u: string) {
        super(u);
        url = u;
      }
    }
    connectSse("/api/chat/x/events", { onEvent: () => {} }, Ctor);
    expect(url).toBe("/api/chat/x/events");
  });

  it("dispatches parsed JSON by event type", () => {
    const onEvent = vi.fn();
    const { es } = connectWithSpy({ onEvent });

    es.emit("token", JSON.stringify({ ...base, delta: "Hello" }));
    expect(onEvent).toHaveBeenCalledWith("token", { ...base, delta: "Hello" });

    es.emit("tool_start", JSON.stringify({ ...base, toolCallId: "t1", toolName: "kb_search", args: {} }));
    expect(onEvent).toHaveBeenCalledWith("tool_start", {
      ...base,
      toolCallId: "t1",
      toolName: "kb_search",
      args: {},
    });
  });

  it("calls onError for malformed JSON without killing the stream", () => {
    const onError = vi.fn();
    const onEvent = vi.fn();
    const { es } = connectWithSpy({ onEvent, onError });

    es.emit("token", "not-json{");
    expect(onError).toHaveBeenCalledTimes(1);
    expect(onEvent).not.toHaveBeenCalled();

    // Stream still alive: next valid event dispatches fine.
    es.emit("token", JSON.stringify({ ...base, delta: "ok" }));
    expect(onEvent).toHaveBeenCalledWith("token", { ...base, delta: "ok" });
  });

  it("reports open → reconnecting → closed state transitions", () => {
    const onStateChange = vi.fn();
    const { es } = connectWithSpy({ onEvent: () => {}, onStateChange });

    es.emit("open");
    expect(onStateChange).toHaveBeenLastCalledWith("open");

    es.readyState = 0; // dropped connection → EventSource will retry
    es.emit("error");
    expect(onStateChange).toHaveBeenLastCalledWith("reconnecting");

    es.readyState = 2; // server closed
    es.emit("error");
    expect(onStateChange).toHaveBeenLastCalledWith("closed");
  });

  it("close() stops delivery and marks the connection closed", () => {
    const onEvent = vi.fn();
    const onStateChange = vi.fn();
    const { conn, es } = connectWithSpy({ onEvent, onStateChange });

    conn.close();
    expect(es.closed).toBe(true);
    expect(onStateChange).toHaveBeenLastCalledWith("closed");

    es.emit("token", JSON.stringify({ ...base, delta: "late" }));
    expect(onEvent).not.toHaveBeenCalled();
  });

  it("ignores events after an explicit server close (readyState 2)", () => {
    const onEvent = vi.fn();
    const { es } = connectWithSpy({ onEvent });

    es.readyState = 2;
    es.emit("error");
    es.emit("token", JSON.stringify({ ...base, delta: "late" }));
    expect(onEvent).not.toHaveBeenCalled();
  });
});
