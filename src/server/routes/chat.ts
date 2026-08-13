/**
 * src/server/routes/chat.ts — chat endpoints.
 *
 *   POST /api/chat                create a chat turn (rate-limited 10/min/IP)
 *   GET  /api/chat/:id/events     SSE stream (sse: "only", replay + live)
 *   POST /api/chat/:id/steer      queue a steering message (rate-limited)
 *   POST /api/chat/:id/cancel     abort the running turn (rate-limited)
 *   WS   /api/chat/:id            full-duplex: same event feed + steer/cancel frames
 *
 * Start is decoupled from streaming (§2.1): POST returns immediately with the chat id;
 * the turn runs in the background and its events buffer in the registry so SSE clients
 * can connect late and replay via Last-Event-ID.
 */
import type { FastifyPluginAsync } from "fastify";
import { randomUUID } from "node:crypto";
import { env } from "../../config/env.js";
import {
  ChatRegistry,
  type ChatTurn,
  type SSEEventType,
} from "../../streaming/registry.js";
import { attachBridge, mapPromptError } from "../../streaming/bridge.js";
import { saveTurn } from "../../streaming/persist.js";
import { createSseHandler } from "../../streaming/sse.js";
import { createWsHandler } from "../../streaming/websocket.js";
import type { SupportRuntime } from "../../runtime/index.js";

export interface ChatRouteOptions {
  registry: ChatRegistry;
  createRuntime: (opts?: { chatId?: string; model?: string; initialMessages?: unknown[] }) => Promise<SupportRuntime>;
}

interface ChatBody {
  message?: string;
  /** Design §2.2 field; `sessionId` is accepted as an alias (task spec). */
  conversationId?: string;
  sessionId?: string;
  ticketId?: number;
  metadata?: Record<string, unknown>;
}

interface SteerBody {
  text?: string;
}

function newId(prefix: "chat" | "conv"): string {
  return `${prefix}_${randomUUID().replace(/-/g, "").slice(0, 20)}`;
}

/**
 * Run one turn in the background: attach the bridge, prompt the session, surface
 * failures as `error` SSE events, and detach the bridge when done.
 */
async function runTurn(
  registry: ChatRegistry,
  turn: ChatTurn,
  message: string,
): Promise<void> {
  // Rehydrated (historical) chats have no live session — never run them.
  if (!turn.session) return;

  const sink = {
    emit(event: SSEEventType, data: unknown): void {
      registry.emit(turn.chatId, event, data);
    },
  };
  const detachBridge = attachBridge(turn.session, sink, {
    chatId: turn.chatId,
    conversationId: turn.conversationId,
  });
  turn.detachBridge = detachBridge;

  try {
    const messagesBefore = turn.session!.getLastMessages().length;
    await turn.session!.prompt(message);
    // If the agent never ran (input guardrail returned "handled"), surface it.
    if (turn.session!.getLastMessages().length === messagesBefore) {
      registry.emit(turn.chatId, "error", {
        chatId: turn.chatId,
        code: "guardrail_blocked",
        message: "Input blocked by guardrails (prompt-injection or policy violation).",
        retryable: false,
      });
      registry.mark(turn.chatId, "error");
      return;
    }
    if (turn.status === "running") registry.mark(turn.chatId, "done");
  } catch (err) {
    const { code, message: msg, retryable } = mapPromptError(err);
    registry.emit(turn.chatId, "error", {
      chatId: turn.chatId,
      code,
      message: msg,
      retryable,
    });
    registry.mark(turn.chatId, code === "canceled" ? "canceled" : "error");
  } finally {
    turn.detachBridge?.();
    turn.detachBridge = null;
    // Durability: snapshot the conversation + write it to Postgres (best-effort).
    turn.messages = turn.session?.getLastMessages() ?? turn.messages;
    turn.messageCount = turn.messages.length;
    void saveTurn(turn);
  }
}

const chatBodySchema = {
  type: "object",
  additionalProperties: false,
  required: ["message"],
  properties: {
    message: { type: "string", minLength: 1, maxLength: 4000 },
    conversationId: { type: "string" },
    sessionId: { type: "string" },
    ticketId: { type: "number" },
    metadata: { type: "object", additionalProperties: true },
  },
} as const;

export const chatRoutes: FastifyPluginAsync<ChatRouteOptions> = async (app, opts) => {
  const { registry, createRuntime } = opts;
  const sseHandler = createSseHandler(registry);
  const wsHandler = createWsHandler(registry);

  // --- POST /api/chat — create a chat turn ---
  app.post<{ Body: ChatBody }>(
    "/api/chat",
    {
      schema: { body: chatBodySchema },
      config: { rateLimit: { max: env.RATE_CHAT_MAX, timeWindow: "1 minute" } },
    },
    async (request, reply) => {
      const body = request.body;
      const message = body.message ?? "";
      if (!message.trim()) {
        return reply.code(400).send({ error: "invalid_body", message: "message is required" });
      }

      const chatId = newId("chat");
      const conversationId = body.conversationId ?? body.sessionId ?? newId("conv");

      // Resume: a follow-up in an existing conversation seeds the agent with the
      // prior history (from the store or the live registry) so the model has context.
      const prior = registry.getByConversation(conversationId);
      const initialMessages = prior ? [...prior.messages] : undefined;

      const session = await createRuntime({
        chatId,
        model: env.PI_MODEL || undefined,
        initialMessages,
      });
      const turn = registry.create({ chatId, conversationId, session });

      // Fire-and-forget the turn; POST returns immediately (§2.1).
      void runTurn(registry, turn, message).catch((err) => {
        app.log.error({ chatId, err }, "background turn crashed");
      });

      return reply.code(201).send({
        chatId,
        conversationId,
        eventsUrl: `/api/chat/${chatId}/events`,
        status: "started",
      });
    },
  );

  // --- GET /api/chat/:id/events — SSE stream (replay + live) ---
  app.get(
    "/api/chat/:id/events",
    { sse: "only", config: { rateLimit: false } },
    sseHandler,
  );

  // --- POST /api/chat/:id/steer ---
  app.post<{ Params: { id: string }; Body: SteerBody }>(
    "/api/chat/:id/steer",
    {
      schema: {
        body: {
          type: "object",
          additionalProperties: false,
          required: ["text"],
          properties: { text: { type: "string", minLength: 1, maxLength: 2000 } },
        },
      },
      config: { rateLimit: { max: env.RATE_CHAT_MAX, timeWindow: "1 minute" } },
    },
    async (request, reply) => {
      const turn = registry.get(request.params.id);
      if (!turn) {
        return reply.code(404).send({ error: "chat_not_found", message: `No chat with id ${request.params.id}` });
      }
      if (!turn.session) {
        return reply.code(409).send({ error: "no_live_turn", message: "This chat has no running turn (historical session)." });
      }
      await turn.session.steer(request.body.text ?? "");
      return reply.code(202).send({ queued: true });
    },
  );

  // --- POST /api/chat/:id/cancel ---
  app.post<{ Params: { id: string } }>(
    "/api/chat/:id/cancel",
    { config: { rateLimit: { max: env.RATE_CHAT_MAX, timeWindow: "1 minute" } } },
    async (request, reply) => {
      const turn = registry.get(request.params.id);
      if (!turn) {
        return reply.code(404).send({ error: "chat_not_found", message: `No chat with id ${request.params.id}` });
      }
      if (!turn.session) {
        return reply.code(409).send({ error: "no_live_turn", message: "This chat has no running turn (historical session)." });
      }
      await turn.session.abort();
      return reply.code(202).send({ cancelled: true });
    },
  );

  // --- WS /api/chat/:id — full-duplex channel ---
  app.get("/api/chat/:id", { websocket: true, config: { rateLimit: false } }, wsHandler);
};
