/**
 * src/config/env.ts — env parsing + server limits.
 *
 * NOTE (api-streaming track): `src/config/` is not claimed by another track in the
 * integration contract, so this module lives here. It only parses env vars; no side
 * effects beyond loading .env. See CONTRACT-NOTES.md.
 */
import "dotenv/config";

function num(value: string | undefined, fallback: number): number {
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

export const env = {
  /** Server bind address */
  PORT: num(process.env.PORT, 8000),
  HOST: process.env.HOST ?? "0.0.0.0",
  LOG_LEVEL: process.env.LOG_LEVEL ?? "info",

  /** Data stores */
  DATABASE_URL:
    process.env.DATABASE_URL ?? "postgresql://acs:acs@localhost:5432/acs",
  REDIS_URL: process.env.REDIS_URL ?? "redis://localhost:6379",

  /** Agent runtime (used by the mock; the real runtime lives in src/runtime/) */
  PI_MODEL: process.env.PI_MODEL ?? "",
  EMBEDDING_MODEL: process.env.EMBEDDING_MODEL ?? "text-embedding-3-small",

  // --- Streaming limits (AGENTS.md: rate limiting is mandatory) ---

  /** How many SSE events to keep per chat for Last-Event-ID replay */
  RING_BUFFER_SIZE: num(process.env.RING_BUFFER_SIZE, 200),

  /** Max concurrent SSE or WS connections per client IP */
  MAX_CONNECTIONS_PER_IP: num(process.env.MAX_CONNECTIONS_PER_IP, 5),

  /** Per-IP request limits (@fastify/rate-limit, per minute) */
  RATE_CHAT_MAX: num(process.env.RATE_CHAT_MAX, 10), // POST /api/chat, steer, cancel, tasks
  RATE_READ_MAX: num(process.env.RATE_READ_MAX, 30), // GET reads (global default)

  /** Agent turn budget (ms) — enforced by the runtime; surfaced via `error` event */
  TURN_BUDGET_MS: num(process.env.TURN_BUDGET_MS, 120_000),
} as const;
