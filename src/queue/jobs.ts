/**
 * src/queue/jobs.ts — BullMQ job types + queue factory.
 *
 * Job types (§4.7 of the design doc):
 *   ingest.document — PDF/manual → chunk → embed → upsert documents + chunks
 *   ingest.tickets  — CSV (suraj520 / CFPB / comcast) → tickets table
 *   reembed         — chunks whose embedding_model differs from the target → re-embed
 */
import { Queue } from "bullmq";
import { Redis } from "ioredis";
import { env } from "../config/env.js";

export const QUEUE_NAME = "acs-tasks";

export type TaskType = "ingest.document" | "ingest.tickets" | "reembed";

export interface IngestDocumentPayload {
  /** Path to the manual PDF (or directory). */
  path: string;
  docName?: string;
  docSource?: "pdf" | "csv";
}

export interface IngestTicketsPayload {
  source: "suraj520" | "cfpb" | "comcast";
  /** Optional CSV path; defaults to the dataset provisioned by retrieval-core. */
  filePath?: string;
}

export interface ReembedPayload {
  /** Target embedding model; chunks with a different model are re-embedded. */
  model?: string;
  /** Optional explicit chunk ids to re-embed. */
  chunkIds?: number[];
  limit?: number;
}

/** Discriminated union keyed by job name. */
export type TaskPayload =
  | { type: "ingest.document"; payload: IngestDocumentPayload }
  | { type: "ingest.tickets"; payload: IngestTicketsPayload }
  | { type: "reembed"; payload: ReembedPayload };

/** Validate an incoming POST /api/tasks body before enqueueing. */
export function parseTaskType(type: unknown): TaskType | null {
  return type === "ingest.document" || type === "ingest.tickets" || type === "reembed"
    ? type
    : null;
}

/**
 * BullMQ connection rule: the ioredis instance used by Queue/Worker must set
 * `maxRetriesPerRequest: null` (BullMQ needs to issue blocking commands without
 * ioredis retrying them and failing).
 */
export function redisConnection(): Redis {
  return new Redis(env.REDIS_URL, { maxRetriesPerRequest: null });
}

/** Factory for the shared tasks queue. Enqueue with `queue.add(jobName, payload)`. */
export function createTaskQueue(): Queue {
  return new Queue(QUEUE_NAME, { connection: redisConnection() });
}
