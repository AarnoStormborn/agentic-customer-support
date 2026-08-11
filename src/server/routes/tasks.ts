/**
 * src/server/routes/tasks.ts — POST /api/tasks.
 *
 * Enqueues a background BullMQ job (ingest.document / ingest.tickets / reembed).
 * Returns 202 + taskId immediately; the worker (src/queue/worker.ts) does the work.
 */
import type { FastifyPluginAsync } from "fastify";
import type { Queue } from "bullmq";
import { env } from "../../config/env.js";
import { parseTaskType, type TaskType } from "../../queue/jobs.js";

export interface TaskRouteOptions {
  taskQueue: Queue;
}

interface TaskBody {
  type?: unknown;
  payload?: Record<string, unknown>;
}

export const taskRoutes: FastifyPluginAsync<TaskRouteOptions> = async (app, opts) => {
  const { taskQueue } = opts;

  app.post<{ Body: TaskBody }>(
    "/api/tasks",
    {
      config: { rateLimit: { max: env.RATE_CHAT_MAX, timeWindow: "1 minute" } },
    },
    async (request, reply) => {
      const body = request.body ?? {};
      const type: TaskType | null = parseTaskType(body.type);
      if (!type) {
        return reply.code(400).send({
          error: "invalid_type",
          message: "type must be one of: ingest.document, ingest.tickets, reembed",
        });
      }

      const job = await taskQueue.add(
        type,
        { type, payload: body.payload ?? {} },
        { attempts: 3, backoff: { type: "exponential", delay: 2000 } },
      );

      app.log.info({ taskId: job.id, type }, "enqueued task");
      return reply.code(202).send({ taskId: job.id, type, status: "queued" });
    },
  );
};
