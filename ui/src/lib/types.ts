/**
 * lib/types.ts — mirrors the REAL backend payloads 1:1.
 * Sources of truth: src/server/routes/{chat,data,sessions,health}.ts and
 * src/streaming/bridge.ts (SSE payload shapes). Do not invent fields here —
 * if the backend changes, update this file.
 */

// --- REST payloads -------------------------------------------------------

export interface ChatStartResponse {
  chatId: string;
  conversationId: string;
  eventsUrl: string;
  status: "started";
}

export type TurnStatus = "running" | "done" | "error" | "canceled";

export interface SessionSummary {
  chatId: string;
  conversationId: string;
  status: TurnStatus;
  createdAt: number;
  finishedAt: number | null;
  messageCount: number;
  /** First meaningful user/assistant text (backend-computed) */
  preview: string;
}

export interface TicketRow {
  ticket_id: number;
  source: string;
  customer_name: string | null;
  product_purchased: string | null;
  date_of_purchase: string | null;
  ticket_type: string | null;
  ticket_priority: string | null;
  ticket_channel: string | null;
  ticket_subject: string | null;
  complaint_narrative: string | null;
  company: string | null;
  status: string;
  is_synthetic: boolean | null;
  created_at: string | null;
  [key: string]: unknown; // GET /api/tickets/:id returns SELECT *, may grow
}

export interface TicketsResponse {
  rows: TicketRow[];
  total: number;
  page: number;
  pageSize: number;
}

export interface ManualSummary {
  doc_id: number;
  doc_name: string;
  file_path: string | null;
  doc_type: string | null;
  created_at: string | null;
  chunk_count: number;
}

export interface Chunk {
  chunk_id: number;
  doc_id: number;
  chunk_index: number;
  chunk_text: string;
  section: string | null;
  heading_path: string | null;
  page_start: number | null;
  page_end: number | null;
}

export interface ModelsResponse {
  models: string[];
  default: string | null;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  uptime: number;
  deps: { postgres: "ok" | "down"; redis: "ok" | "down" };
}

// --- SSE payloads (src/streaming/bridge.ts, design §2.3) ------------------

/** A retrieval source attached to the `done` event. */
export interface SourceRef {
  type: "kb" | "sql" | "web";
  title?: string | null;
  docName?: string | null;
  sectionPath?: string | null;
  page?: number | null;
  url?: string | null;
  score?: number | null;
  /** sql sources carry the full ticket row */
  row?: Record<string, unknown>;
}

export type ErrorCode =
  | "turn_timeout"
  | "canceled"
  | "provider_error"
  | "guardrail_blocked"
  | "internal";

export interface DoneUsage {
  inputTokens?: number;
  outputTokens?: number;
  totalCostUsd?: number;
}

/** Payload per SSE event type — exactly the bridge's §2.3 schema. */
export interface SseEventMap {
  turn_start: { chatId: string; turnIndex: number; ts: number };
  token: { chatId: string; turnIndex: number; delta: string };
  thinking: { chatId: string; turnIndex: number; delta: string };
  tool_start: {
    chatId: string;
    turnIndex: number;
    toolCallId: string;
    toolName: string;
    args: Record<string, unknown>;
  };
  tool_update: { chatId: string; turnIndex: number; toolCallId: string; partial: string };
  tool_end: {
    chatId: string;
    turnIndex: number;
    toolCallId: string;
    toolName: string;
    isError: boolean;
    durationMs: number;
    summary: string | null;
  };
  turn_end: { chatId: string; turnIndex: number; ts: number };
  done: {
    chatId: string;
    conversationId: string;
    turnIndex: number;
    message: string;
    sources: SourceRef[];
    usage: DoneUsage;
  };
  error: { chatId: string; code: ErrorCode; message: string; retryable: boolean };
  retry_start: { chatId: string; turnIndex: number };
  retry_end: { chatId: string; turnIndex: number };
  queue_update: { chatId: string; steering: number; followUp: number };
}

export type SseEventType = keyof SseEventMap;

// --- Chat history (pi SDK message shape) ---------------------------------

export interface SdkMessageBlock {
  type: string;
  text?: string;
  [key: string]: unknown;
}

export interface SdkMessage {
  role: "user" | "assistant";
  content: string | SdkMessageBlock[];
  [key: string]: unknown;
}

export interface ChatHistoryResponse {
  chatId: string;
  conversationId: string;
  status: TurnStatus;
  createdAt: number;
  messages: SdkMessage[];
}

export interface CompareModeResult {
  mode: string;
  relaxed: boolean;
  queryTimeMs: number;
  top: { docName: string | null; sectionPath: string | null; score: number }[];
}
export interface CompareResponse {
  query: string;
  topK: number;
  modes: CompareModeResult[];
}
