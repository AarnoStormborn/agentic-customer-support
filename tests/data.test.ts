/**
 * Data/session endpoint tests (Phase 5b.1 backend additions).
 *
 * Sessions/history tests use an injected mock runtime (no DB needed).
 * Ticket/manual tests hit the live Postgres and are skipped unless ACS_TEST_DB=1.
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import type { FastifyInstance } from "fastify";
import { buildApp } from "../src/server/app.js";
import type { SupportRuntime } from "../src/runtime/index.js";

function makeSessionRuntime(messages: unknown[] = []): SupportRuntime {
  return {
    async prompt() {},
    async steer() {},
    async abort() {},
    subscribe() {
      return () => {};
    },
    getLastMessages: () => messages,
    dispose() {},
  };
}

const fakeQueue = { add: async () => ({ id: "job-1" }) } as never;

describe("sessions & history endpoints (mock runtime)", () => {
  let app: FastifyInstance;

  beforeAll(async () => {
    app = await buildApp({
      createRuntime: async () =>
        makeSessionRuntime([
          { role: "user", content: [{ type: "text", text: "reset my lg tv wifi" }] },
          { role: "assistant", content: [{ type: "text", text: "Press Settings > Network…" }] },
        ]),
      taskQueue: fakeQueue,
    });
    await app.ready();
  });

  afterAll(async () => {
    await app.close();
  });

  it("GET /api/sessions lists chats with previews", async () => {
    const created = await app.inject({
      method: "POST",
      url: "/api/chat",
      payload: { message: "reset my lg tv wifi" },
    });
    expect(created.statusCode).toBe(201);
    const chatId = created.json().chatId;

    const res = await app.inject({ method: "GET", url: "/api/sessions" });
    expect(res.statusCode).toBe(200);
    const { sessions } = res.json();
    const mine = sessions.find((s: { chatId: string }) => s.chatId === chatId);
    expect(mine).toBeTruthy();
    expect(mine.preview).toContain("reset my lg tv wifi");
  });

  it("GET /api/chat/:id/history returns messages", async () => {
    const created = await app.inject({
      method: "POST",
      url: "/api/chat",
      payload: { message: "hi" },
    });
    const { chatId } = created.json();

    const res = await app.inject({ method: "GET", url: `/api/chat/${chatId}/history` });
    expect(res.statusCode).toBe(200);
    expect(res.json().messages.length).toBe(2);
  });

  it("GET /api/chat/:id/history 404s for unknown chats", async () => {
    const res = await app.inject({ method: "GET", url: "/api/chat/chat_nope/history" });
    expect(res.statusCode).toBe(404);
  });

  it("DELETE /api/sessions/:id removes the chat", async () => {
    const created = await app.inject({
      method: "POST",
      url: "/api/chat",
      payload: { message: "bye" },
    });
    const { chatId } = created.json();

    const del = await app.inject({ method: "DELETE", url: `/api/sessions/${chatId}` });
    expect(del.statusCode).toBe(200);

    const missing = await app.inject({ method: "GET", url: `/api/chat/${chatId}/history` });
    expect(missing.statusCode).toBe(404);
  });

  it("GET /api/models returns a model list", async () => {
    const res = await app.inject({ method: "GET", url: "/api/models" });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(Array.isArray(body.models)).toBe(true);
  });
});

describe("tickets & manuals endpoints (live DB)", () => {
  const run = process.env.ACS_TEST_DB === "1" ? describe : describe.skip;
  run("data endpoints", () => {
    let app: FastifyInstance;

    beforeAll(async () => {
      app = await buildApp({
        createRuntime: async () => makeSessionRuntime(),
        taskQueue: fakeQueue,
      });
      await app.ready();
    });

    afterAll(async () => {
      await app.close();
    });

    it("GET /api/tickets searches + paginates", async () => {
      const res = await app.inject({
        method: "GET",
        url: "/api/tickets?q=lg%20oled&pageSize=5&page=1",
      });
      expect(res.statusCode).toBe(200);
      const body = res.json();
      expect(body.total).toBeGreaterThan(0);
      expect(body.rows.length).toBeLessThanOrEqual(5);
      expect(body.rows[0]).toHaveProperty("ticket_id");
    });

    it("GET /api/tickets/:id returns a ticket or 404", async () => {
      const list = await app.inject({ method: "GET", url: "/api/tickets?pageSize=1" });
      const first = list.json().rows[0] as { ticket_id: number };
      const ok = await app.inject({ method: "GET", url: `/api/tickets/${first.ticket_id}` });
      expect(ok.statusCode).toBe(200);
      expect(ok.json().ticket_id).toBe(first.ticket_id);

      const missing = await app.inject({ method: "GET", url: "/api/tickets/99999999" });
      expect(missing.statusCode).toBe(404);
    });

    it("GET /api/manuals lists manuals with chunk counts", async () => {
      const res = await app.inject({ method: "GET", url: "/api/manuals" });
      expect(res.statusCode).toBe(200);
      const { manuals } = res.json();
      expect(manuals.length).toBeGreaterThan(0);
      expect(manuals[0]).toHaveProperty("doc_id");
      expect(manuals[0]).toHaveProperty("chunk_count");
    });

    it("GET /api/manuals/:id/chunks returns ordered chunks", async () => {
      const manuals = (await app.inject({ method: "GET", url: "/api/manuals" })).json().manuals;
      const res = await app.inject({
        method: "GET",
        url: `/api/manuals/${manuals[0].doc_id}/chunks?limit=3`,
      });
      expect(res.statusCode).toBe(200);
      const { chunks } = res.json();
      expect(chunks.length).toBeGreaterThan(0);
      expect(chunks[0]).toHaveProperty("chunk_text");
    });
  });
});
