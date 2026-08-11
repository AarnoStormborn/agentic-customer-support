/**
 * lib/api.ts — typed fetch wrappers for the Fastify backend.
 * Base URL: VITE_API_URL (ui/.env, optional) or "" → same origin, which the
 * Vite dev server proxies to http://localhost:8000 (see vite.config.ts).
 */
import type {
  ChatHistoryResponse,
  ChatStartResponse,
  Chunk,
  HealthResponse,
  ManualSummary,
  ModelsResponse,
  SessionSummary,
  TicketRow,
  TicketsResponse,
} from "./types";

export const API_BASE = (import.meta.env.VITE_API_URL ?? "").replace(/\/+$/, "");

/** Uniform error shape for every failed request. */
export class ApiError extends Error {
  readonly status: number;
  readonly code?: string;

  constructor(status: number, message: string, code?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
  });

  if (!res.ok) {
    let message = `HTTP ${res.status}`;
    let code: string | undefined;
    try {
      const body = (await res.json()) as { error?: string; message?: string };
      code = body?.error;
      if (body?.message) message = body.message;
    } catch {
      // non-JSON error body — keep the HTTP fallback message
    }
    throw new ApiError(res.status, message, code);
  }
  return (await res.json()) as T;
}

function buildQuery(params: Record<string, string | number | undefined>): string {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") sp.set(k, String(v));
  }
  const s = sp.toString();
  return s ? `?${s}` : "";
}

export const api = {
  // --- infra ------------------------------------------------------------
  health: () => request<HealthResponse>("/health"),

  models: () => request<ModelsResponse>("/api/models"),

  // --- chat --------------------------------------------------------------
  startChat: (body: {
    message: string;
    conversationId?: string;
    ticketId?: number;
    metadata?: Record<string, unknown>;
  }) =>
    request<ChatStartResponse>("/api/chat", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  cancelChat: (chatId: string) =>
    request<{ cancelled: boolean }>(`/api/chat/${chatId}/cancel`, { method: "POST" }),

  steerChat: (chatId: string, text: string) =>
    request<{ queued: boolean }>(`/api/chat/${chatId}/steer`, {
      method: "POST",
      body: JSON.stringify({ text }),
    }),

  chatHistory: (chatId: string) =>
    request<ChatHistoryResponse>(`/api/chat/${chatId}/history`),

  // --- sessions -----------------------------------------------------------
  sessions: () => request<{ sessions: SessionSummary[] }>("/api/sessions"),

  deleteSession: (chatId: string) =>
    request<{ deleted: boolean }>(`/api/sessions/${chatId}`, { method: "DELETE" }),

  // --- tickets ------------------------------------------------------------
  tickets: (params: { q?: string; status?: string; page?: number; pageSize?: number }) =>
    request<TicketsResponse>(`/api/tickets${buildQuery(params)}`),

  ticket: (id: number) => request<TicketRow>(`/api/tickets/${id}`),

  // --- manuals ------------------------------------------------------------
  manuals: () => request<{ manuals: ManualSummary[] }>("/api/manuals"),

  manualChunks: (id: number, limit = 200) =>
    request<{ manualId: number; chunks: Chunk[] }>(
      `/api/manuals/${id}/chunks${buildQuery({ limit })}`,
    ),
};
