/**
 * src/streaming/sse.ts — GET /api/chat/:id/events route handler (@fastify/sse).
 *
 * `sse: "only"` mode: the handler streams exclusively. Flow:
 *   1. Commit the SSE response immediately (`reply.sse.sendHeaders()` + keepAlive) so
 *      the connection stays open even when nothing is buffered yet.
 *   2. Replay buffered events after the client's Last-Event-ID (ring buffer).
 *   3. Subscribe for live events until a terminal `done` / `error`.
 *
 * Why not `reply.sse.replay()`? The plugin's helper only fires when a Last-Event-ID
 * header exists (reconnects). First-time clients get nothing — and if the ring is
 * empty the response would close immediately (headers never committed). Manual replay
 * covers both cases uniformly.
 */
import type { FastifyReply, FastifyRequest } from "fastify";
import type { ChatRegistry, SSEEnvelope } from "./registry.js";
import { IpConnectionCounter } from "./limits.js";

/** Shared per-IP cap across all SSE endpoints (and reused by the WS handler). */
export const sseConnectionCounter = new IpConnectionCounter();

export interface SseRouteParams {
  id: string;
}

const TERMINAL_EVENTS = new Set(["done", "error"]);

export function createSseHandler(registry: ChatRegistry) {
  return async function sseHandler(
    request: FastifyRequest<{ Params: SseRouteParams }>,
    reply: FastifyReply,
  ): Promise<void> {
    const { id: chatId } = request.params;

    const turn = registry.get(chatId);
    if (!turn) {
      await reply.code(404).send({ error: "chat_not_found", message: `No chat with id ${chatId}` });
      return;
    }

    // Per-IP connection cap (rate-limit is disabled on this long-lived route).
    if (!sseConnectionCounter.tryAcquire(request.ip)) {
      await reply
        .code(429)
        .send({ error: "too_many_connections", message: "Too many concurrent SSE connections" });
      return;
    }

    // Commit the response as SSE now — this pins the connection open, starts the
    // heartbeat, and lets us send even if nothing is buffered yet.
    reply.sse.keepAlive();
    reply.header("X-Accel-Buffering", "no");
    reply.sse.sendHeaders();

    // 1) Replay: buffered events after the client's Last-Event-ID (0 = from start).
    const lastEventId = request.headers["last-event-id"];
    const afterId = lastEventId ? Number(lastEventId) : 0;
    if (Number.isFinite(afterId)) {
      for (const env of registry.replay(chatId, afterId)) {
        await sendEvent(reply, env);
      }
    }

    // 2) Live: subscribe for anything emitted after the replay snapshot.
    const unsubscribe = registry.subscribe(chatId, (env) => {
      void sendEvent(reply, env);
      if (TERMINAL_EVENTS.has(env.event)) {
        // Give the terminal event a moment to flush, then close the stream.
        setTimeout(() => void reply.sse.close(), 100);
      }
    });

    reply.sse.onClose(() => {
      unsubscribe();
      sseConnectionCounter.release(request.ip);
    });

    // 3) If the turn already finished while we were replaying, close after the flush.
    const last = turn.ring[turn.ring.length - 1];
    if (last && TERMINAL_EVENTS.has(last.event)) {
      setTimeout(() => void reply.sse.close(), 150);
    }
  };
}

async function sendEvent(reply: FastifyReply, env: SSEEnvelope): Promise<void> {
  await reply.sse.send({
    id: String(env.id),
    event: env.event,
    data: env.data,
  });
}
