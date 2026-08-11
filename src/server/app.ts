/**
 * src/server/app.ts — Fastify instance + plugin registration.
 *
 * Plugins (AGENTS.md stack table):
 *   @fastify/cors        CORS for the React SPA
 *   @fastify/sse         SSE routes (sse: "only") with Last-Event-ID replay + heartbeat
 *   @fastify/websocket   WS /api/chat/:id
 *   @fastify/rate-limit  per-IP limits (chat 10/min, reads 30/min, §2.5)
 *   pino logging         Fastify-native (logger option, no extra dep)
 *
 * Routes get their deps (registry, runtime factory, task queue) via plugin options so
 * integration only swaps the `createRuntime` factory.
 */
import Fastify, { type FastifyInstance, type FastifyPluginAsync } from "fastify";
import cors from "@fastify/cors";
import sseModule from "@fastify/sse";
import websocket from "@fastify/websocket";
import rateLimit from "@fastify/rate-limit";
import type { Queue } from "bullmq";
import { env } from "../config/env.js";
import { ChatRegistry } from "../streaming/registry.js";
import { createTaskQueue } from "../queue/jobs.js";

// @fastify/sse is CJS with an ESM-style d.ts (`export default`); under
// verbatimModuleSyntax the default import types as the module namespace, but at
// runtime it IS the fp-wrapped plugin function (module.exports). Cast for typing.
const sse = sseModule as unknown as FastifyPluginAsync<{ heartbeatInterval?: number }>;
import { createSupportRuntime, type SupportRuntime } from "../runtime/index.js";
import { healthRoutes } from "./routes/health.js";
import { chatRoutes, type ChatRouteOptions } from "./routes/chat.js";
import { taskRoutes } from "./routes/tasks.js";
import { dataRoutes } from "./routes/data.js";
import { sessionRoutes } from "./routes/sessions.js";

export interface BuildAppDeps {
  registry: ChatRegistry;
  taskQueue: Queue;
  /** Swap point for the real runtime at integration: createSupportRuntime from runtime/index.js. */
  createRuntime: (opts?: { chatId?: string; model?: string }) => Promise<SupportRuntime>;
}

export async function buildApp(overrides: Partial<BuildAppDeps> = {}): Promise<FastifyInstance> {
  const app = Fastify({
    logger: { level: env.LOG_LEVEL },
  });

  const registry = overrides.registry ?? new ChatRegistry();
  const taskQueue = overrides.taskQueue ?? createTaskQueue();
  const createRuntime = overrides.createRuntime ?? createSupportRuntime;

  await app.register(cors, { origin: true });

  await app.register(sse, { heartbeatInterval: 15_000 });

  await app.register(websocket, {
    options: { maxPayload: 64 * 1024 },
  });

  // Global default: 30 req/min/IP for reads. Mutating/chat routes override to 10/min
  // at the route level; SSE/WS routes disable it (long-lived; connection caps instead).
  await app.register(rateLimit, {
    max: env.RATE_READ_MAX,
    timeWindow: "1 minute",
  });

  await app.register(healthRoutes);
  await app.register(dataRoutes);
  await app.register(sessionRoutes, { registry });
  await app.register(chatRoutes, { registry, createRuntime } satisfies ChatRouteOptions);
  await app.register(taskRoutes, { taskQueue });

  return app;
}
