/**
 * api.test.ts — typed fetch wrappers against a mocked fetch (no network).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "./api";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

describe("api", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("startChat POSTs JSON and parses the response", async () => {
    fetchMock.mockResolvedValue(
      jsonResponse({
        chatId: "chat_abc",
        conversationId: "conv_xyz",
        eventsUrl: "/api/chat/chat_abc/events",
        status: "started",
      }, 201),
    );

    const res = await api.startChat({ message: "hello", conversationId: "conv_xyz" });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/chat",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ message: "hello", conversationId: "conv_xyz" }),
      }),
    );
    expect(res.chatId).toBe("chat_abc");
    expect(res.eventsUrl).toBe("/api/chat/chat_abc/events");
  });

  it("tickets builds the query string (omitting empty params)", async () => {
    fetchMock.mockImplementation(async () =>
      jsonResponse({ rows: [], total: 0, page: 1, pageSize: 20 }),
    );

    await api.tickets({ q: "wifi", status: "open", page: 2, pageSize: 20 });
    const url = fetchMock.mock.calls[0]?.[0] as string;
    expect(url).toBe("/api/tickets?q=wifi&status=open&page=2&pageSize=20");

    await api.tickets({ q: "", status: undefined, page: 1, pageSize: 20 });
    expect(fetchMock.mock.calls[1]?.[0]).toBe("/api/tickets?page=1&pageSize=20");
  });

  it("throws ApiError with backend error body on non-2xx", async () => {
    fetchMock.mockResolvedValue(
      jsonResponse({ error: "chat_not_found", message: "No chat with id x" }, 404),
    );

    const err = await api.chatHistory("x").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    const apiErr = err as ApiError;
    expect(apiErr.status).toBe(404);
    expect(apiErr.code).toBe("chat_not_found");
    expect(apiErr.message).toBe("No chat with id x");
  });

  it("falls back to HTTP status message when the error body is not JSON", async () => {
    fetchMock.mockResolvedValue(new Response("gateway broke", { status: 502 }));

    const err = await api.sessions().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).message).toBe("HTTP 502");
  });

  it("cancelChat and deleteSession send the right method", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ cancelled: true }));
    await api.cancelChat("chat_1");
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/chat/chat_1/cancel",
      expect.objectContaining({ method: "POST" }),
    );

    fetchMock.mockResolvedValue(jsonResponse({ deleted: true }));
    await api.deleteSession("chat_1");
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/sessions/chat_1",
      expect.objectContaining({ method: "DELETE" }),
    );
  });
});
