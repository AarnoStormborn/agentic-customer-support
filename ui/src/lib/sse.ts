/**
 * lib/sse.ts — thin EventSource wrapper for the chat event stream.
 *
 * Why EventSource (not fetch-stream / WebSocket) for tokens:
 *   - auto-reconnect with Last-Event-ID replay (the backend replays buffered
 *     events from the id, so a dropped connection never loses tokens)
 *   - typed `event:` names → we dispatch straight into the chat store
 *
 * Caveat: EventSource can't send POST bodies or custom headers. If the backend
 * ever needs an auth token, swap the internals for fetch + ReadableStream +
 * eventsource-parser behind the same `connectSse` interface.
 *
 * The constructor is injectable so unit tests can fake EventSource without
 * network (see sse.test.ts).
 */

import type { SseEventMap, SseEventType } from "./types";

/** All event types the backend can emit (src/streaming/registry.ts). */
const EVENT_TYPES: readonly SseEventType[] = [
  "turn_start",
  "token",
  "thinking",
  "tool_start",
  "tool_update",
  "tool_end",
  "turn_end",
  "done",
  "error",
  "retry_start",
  "retry_end",
  "queue_update",
];

export type ConnectionState = "connecting" | "open" | "reconnecting" | "closed";

export interface SseHandlers {
  /** Called once per parsed SSE event, dispatched by its `event:` name. */
  onEvent: (type: SseEventType, data: SseEventMap[SseEventType]) => void;
  /** Connection lifecycle (for the TopBar pill / offline states). */
  onStateChange?: (state: ConnectionState) => void;
  /** Malformed JSON payload (the stream is still alive). */
  onError?: (error: unknown) => void;
}

/** Structural subset of the browser EventSource API — enough for fakes. */
export interface EventSourceLike {
  readonly readyState: number;
  addEventListener(type: string, listener: (event: { data?: string }) => void): void;
  close(): void;
}

export type EventSourceCtor = new (url: string) => EventSourceLike;

export interface SseConnection {
  close(): void;
}

// Native EventSource readyState values (browser spec).
const READY_CONNECTING = 0;
const READY_CLOSED = 2;

export function connectSse(
  url: string,
  handlers: SseHandlers,
  ctor: EventSourceCtor = EventSource,
): SseConnection {
  let closed = false;
  const source = new ctor(url);

  const dispatch = (type: SseEventType, raw?: string) => {
    if (closed) return;
    if (raw === undefined) return; // connection-level error event, not a server payload
    let data: unknown;
    try {
      data = JSON.parse(raw);
    } catch (err) {
      handlers.onError?.(err);
      return;
    }
    handlers.onEvent(type, data as SseEventMap[SseEventType]);
  };

  // Typed events: the backend always sends an `event:` name, so each listener
  // receives only its own payloads. The generic `message` listener is a
  // fallback for unnamed events (unused today).
  for (const type of EVENT_TYPES) {
    source.addEventListener(type, (e) => dispatch(type, e.data));
  }
  source.addEventListener("message", (e) => dispatch("token", e.data));

  source.addEventListener("open", () => {
    if (!closed) handlers.onStateChange?.("open");
  });
  source.addEventListener("error", () => {
    if (closed) return;
    // readyState 0 (CONNECTING) = dropped connection, EventSource will retry
    // with Last-Event-ID. 2 (CLOSED) = server ended the stream for good.
    if (source.readyState === READY_CONNECTING) {
      handlers.onStateChange?.("reconnecting");
    } else if (source.readyState === READY_CLOSED) {
      closed = true;
      handlers.onStateChange?.("closed");
    }
  });

  return {
    close() {
      closed = true;
      source.close();
      handlers.onStateChange?.("closed");
    },
  };
}
