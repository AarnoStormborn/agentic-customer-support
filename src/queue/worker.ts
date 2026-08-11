/**
 * src/queue/worker.ts — BullMQ worker for background tasks.
 *
 * Handlers are STUBS for now: they log + ack. The real ingest pipelines land after
 * integration with retrieval-core (src/retrieval/ingest.ts owns the actual work —
 * per the integration contract, that track exports ingestTickets/ingestManuals).
 */
import { Worker, type Job } from "bullmq";
import type { FastifyBaseLogger } from "fastify";
import { QUEUE_NAME, redisConnection, type TaskPayload } from "./jobs.js";
import {
  handleIngestDocument,
  handleIngestTickets,
  handleReembed,
} from "./handlers.js";

export interface WorkerHandlers {
  ingestDocument?(payload: TaskPayload & { type: "ingest.document" }): Promise<unknown>;
  ingestTickets?(payload: TaskPayload & { type: "ingest.tickets" }): Promise<unknown>;
  reembed?(payload: TaskPayload & { type: "reembed" }): Promise<unknown>;
}

/** Default handlers = the real ingest pipelines (Phase 5b.2). */
export const defaultHandlers: WorkerHandlers = {
  ingestDocument: async (task) => handleIngestDocument(task.payload),
  ingestTickets: async (task) => handleIngestTickets(task.payload),
  reembed: async (task) => handleReembed(task.payload),
};

/**
 * Start the worker. Defaults to the real ingest handlers; tests inject stubs.
 * Returns the Worker so the caller can `close()` it on shutdown.
 */
export function startWorker(logger: FastifyBaseLogger, handlers: WorkerHandlers = defaultHandlers): Worker {
  const worker = new Worker(
    QUEUE_NAME,
    async (job: Job) => {
      const task = job.data as TaskPayload;
      logger.info({ jobId: job.id, type: task.type }, "processing task");

      switch (task.type) {
        case "ingest.document":
          if (handlers.ingestDocument) return handlers.ingestDocument(task);
          logger.warn({ jobId: job.id }, "ingest.document handler missing");
          return { handled: false };

        case "ingest.tickets":
          if (handlers.ingestTickets) return handlers.ingestTickets(task);
          logger.warn({ jobId: job.id }, "ingest.tickets handler missing");
          return { handled: false };

        case "reembed":
          if (handlers.reembed) return handlers.reembed(task);
          logger.warn({ jobId: job.id }, "reembed handler missing");
          return { handled: false };

        default:
          logger.warn({ jobId: job.id, type: (task as { type?: string })?.type }, "unknown task type");
          return { handled: false };
      }
    },
    { connection: redisConnection() },
  );

  worker.on("completed", (job) => logger.info({ jobId: job.id, name: job.name }, "task completed"));
  worker.on("failed", (job, err) =>
    logger.error({ jobId: job?.id, name: job?.name, err }, "task failed"),
  );
  worker.on("error", (err) => logger.error({ err }, "worker error (is Redis up?)"));

  return worker;
}
