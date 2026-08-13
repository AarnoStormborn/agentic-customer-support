/**
 * src/streaming/websocket.ts — WS /api/chat/:id route handler (@fastify/websocket).
 *
 * Full-duplex channel: the client receives the same event feed as SSE (as JSON frames
 * `{ id, event, data }`) and can send control frames:
 *   { type: "steer", text: "..." }  → session.steer(text)
 *   { type: "cancel" }              → session.abort()
 */
import type { FastifyRequest } from "fastify";
import type { WebSocket } from "ws";
import type { ChatRegistry } from "./registry.js";
import { sseConnectionCounter } from "./sse.js";

export interface WsRouteParams {
  id: string;
}

interface InboundFrame {
  type?: unknown;
  text?: unknown;
}

export function createWsHandler(registry: ChatRegistry) {
  return (socket: WebSocket, request: FastifyRequest<{ Params: WsRouteParams }>): void => {
    const { id: chatId } = request.params;

    const turn = registry.get(chatId);
    if (!turn) {
      socket.close(4404, "chat_not_found");
      return;
    }

    // Per-IP connection cap (shared with SSE).
    if (!sseConnectionCounter.tryAcquire(request.ip)) {
      socket.close(4429, "too_many_connections");
      return;
    }

    // Send the buffered history first so a late WS client sees the full turn.
    for (const env of registry.replay(chatId, 0)) {
      socket.send(JSON.stringify({ id: env.id, event: env.event, data: env.data }));
    }

    const unsubscribe = registry.subscribe(chatId, (env) => {
      socket.send(JSON.stringify({ id: env.id, event: env.event, data: env.data }));
    });

    socket.on("message", (raw) => {
      let frame: InboundFrame;
      try {
        frame = JSON.parse(raw.toString()) as InboundFrame;
      } catch {
        socket.send(
          JSON.stringify({ event: "error", data: { code: "invalid_frame", message: "Frame must be JSON" } }),
        );
        return;
      }
      switch (frame.type) {
        case "steer":
          if (turn.session && typeof frame.text === "string" && frame.text.length > 0) {
            void turn.session.steer(frame.text);
          }
          break;
        case "cancel":
          if (turn.session) void turn.session.abort();
          break;
        default:
          socket.send(
            JSON.stringify({
              event: "error",
              data: { code: "invalid_frame", message: `Unknown frame type: ${String(frame.type)}` },
            }),
          );
      }
    });

    socket.on("close", () => {
      unsubscribe();
      sseConnectionCounter.release(request.ip);
    });
  };
}
