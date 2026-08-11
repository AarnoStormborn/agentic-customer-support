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

export interface WorkerHandlers {
  ingestDocument?(payload: TaskPayload & { type: "ingest.document" }): Promise<unknown>;
  ingestTickets?(payload: TaskPayload & { type: "ingest.tickets" }): Promise<unknown>;
  reembed?(payload: TaskPayload & { type: "reembed" }): Promise<unknown>;
}

/**
 * Start the worker. Stub handlers log + ack; replace via `handlers` at integration.
 * Returns the Worker so the caller can `close()` it on shutdown.
 */
export function startWorker(logger: FastifyBaseLogger, handlers: WorkerHandlers = {}): Worker {
  const worker = new Worker(
    QUEUE_NAME,
    async (job: Job) => {
      const task = job.data as TaskPayload;
      logger.info({ jobId: job.id, type: task.type }, "processing task");

      switch (task.type) {
        case "ingest.document":
          if (handlers.ingestDocument) return handlers.ingestDocument(task);
          // Stub: real handler lands after integration (retrieval-core ingest).
          logger.info(
            { jobId: job.id, path: task.payload.path },
            "[stub] ingest.document would chunk + embed + upsert this manual",
          );
          return { handled: true, stub: "ingest.document" };

        case "ingest.tickets":
          if (handlers.ingestTickets) return handlers.ingestTickets(task);
          logger.info(
            { jobId: job.id, source: task.payload.source },
            "[stub] ingest.tickets would load this dataset into tickets",
          );
          return { handled: true, stub: "ingest.tickets" };

        case "reembed":
          if (handlers.reembed) return handlers.reembed(task);
          logger.info(
            { jobId: job.id, model: task.payload.model ?? "target" },
            "[stub] reembed would re-embed stale chunks",
          );
          return { handled: true, stub: "reembed" };

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
