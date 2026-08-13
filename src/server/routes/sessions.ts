/**
 * src/server/routes/sessions.ts — chat/session listing + history + delete.
 *
 *   GET    /api/sessions            sidebar history (registry snapshot)
 *   GET    /api/chat/:id/history    message history for one chat (rehydrate on open)
 *   DELETE /api/sessions/:id        close a chat (detach bridge + dispose session)
 *
 * NOTE: the design doc wanted GET /api/chat/:id for history, but that path is
 * taken by the WebSocket route (chat.ts) — history lives at /:id/history.
 */
import type { FastifyPluginAsync } from "fastify";
import type { ChatRegistry } from "../../streaming/registry.js";
import { deleteChat } from "../../streaming/persist.js";

export interface SessionRouteOptions {
  registry: ChatRegistry;
}

export const sessionRoutes: FastifyPluginAsync<SessionRouteOptions> = async (app, opts) => {
  const { registry } = opts;

  app.get("/api/sessions", async () => {
    const sessions = registry.list().map((s) => ({
      ...s,
      // First meaningful user/assistant text as a sidebar preview.
      preview: previewFrom(s.chatId, registry),
    }));
    return { sessions };
  });

  app.get<{ Params: { id: string } }>("/api/chat/:id/history", async (request, reply) => {
    const turn = registry.get(request.params.id);
    if (!turn) {
      return reply.code(404).send({ error: "chat_not_found", message: `No chat with id ${request.params.id}` });
    }
    return {
      chatId: turn.chatId,
      conversationId: turn.conversationId,
      status: turn.status,
      createdAt: turn.createdAt,
      messages: turn.messages,
    };
  });

  app.delete<{ Params: { id: string } }>("/api/sessions/:id", async (request, reply) => {
    const removed = registry.remove(request.params.id);
    if (!removed) {
      return reply.code(404).send({ error: "chat_not_found", message: `No chat with id ${request.params.id}` });
    }
    void deleteChat(request.params.id); // best-effort: purge from the store too
    return { deleted: true };
  });
};

function previewFrom(chatId: string, registry: ChatRegistry): string {
  const turn = registry.get(chatId);
  if (!turn) return "";
  const messages = turn.messages as { role?: string; content?: unknown[] }[];
  // Prefer the most recent user question; fall back to any text-bearing message.
  const candidates = [...messages].reverse();
  const userText = candidates.find((m) => m.role === "user" && textOf(m));
  const anyText = candidates.find((m) => textOf(m));
  return (textOf(userText ?? anyText ?? {}) ?? "").slice(0, 120);
}

function textOf(m: { content?: unknown[] } | undefined): string | null {
  if (!m || !Array.isArray(m.content)) return null;
  const text = m.content
    .filter((p) => (p as { type?: string }).type === "text")
    .map((p) => (p as { text?: string }).text ?? "")
    .join(" ")
    .trim();
  return text || null;
}
