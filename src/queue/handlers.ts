/**
 * src/queue/handlers.ts — real BullMQ job handlers (Phase 5b.2).
 *
 * Replaces the worker stubs. The heavy lifting lives in retrieval-core
 * (src/retrieval/ingest.ts); these handlers adapt job payloads → ingest calls
 * and return IngestSummary-shaped results for the worker's completed event.
 *
 * Exported separately from worker.ts so they are unit-testable without Redis.
 */
import { mkdtemp, rm, symlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { stat } from "node:fs/promises";
import { ingestTickets, ingestManuals } from "../retrieval/ingest.js";
import { embedTexts, EMBEDDING_MODEL } from "../retrieval/embed.js";
import { getPool } from "../db/pool.js";
import type {
  IngestDocumentPayload,
  IngestTicketsPayload,
  ReembedPayload,
} from "./jobs.js";

const REEMBED_BATCH = 100;

export interface IngestDocumentResult {
  handled: true;
  source: string;
  docs?: number;
  chunks?: number;
  dryRun: boolean;
}

/** ingest.document — manual PDF (file) or directory of PDFs → chunks → embeddings. */
export async function handleIngestDocument(payload: IngestDocumentPayload): Promise<IngestDocumentResult> {
  const target = resolve(payload.path);
  const info = await stat(target);

  let dir = target;
  let cleanup: (() => Promise<void>) | null = null;

  if (info.isFile()) {
    // Single PDF: ingest via a temp dir with a symlink so ingestManuals' dir
    // contract is preserved (it lists *.pdf).
    const temp = await mkdtemp(join(tmpdir(), "acs-manual-"));
    const name = payload.docName ?? `manual-${Date.now()}.pdf`;
    await symlink(target, join(temp, name.endsWith(".pdf") ? name : `${name}.pdf`));
    dir = temp;
    cleanup = () => rm(temp, { recursive: true, force: true });
  } else if (!info.isDirectory()) {
    throw new Error(`ingest.document path is neither file nor directory: ${target}`);
  }

  try {
    const summary = await ingestManuals(dir);
    return { handled: true, source: summary.source, docs: summary.docs, chunks: summary.chunks, dryRun: summary.dryRun };
  } finally {
    await cleanup?.();
  }
}

/** ingest.tickets — suraj520 / cfpb / comcast → tickets table (upsert). */
export async function handleIngestTickets(payload: IngestTicketsPayload): Promise<{ handled: true; source: string; rowsRead: number; rowsInserted: number; rowsUpdated: number; dryRun: boolean }> {
  const summary = await ingestTickets(payload.source, { dryRun: payload.filePath === undefined ? false : false });
  return {
    handled: true,
    source: summary.source,
    rowsRead: summary.rowsRead,
    rowsInserted: summary.rowsInserted,
    rowsUpdated: summary.rowsUpdated,
    dryRun: summary.dryRun,
  };
}

/** reembed — re-embed chunk text for stale/missing embeddings (batched). */
export async function handleReembed(payload: ReembedPayload = {}): Promise<{
  handled: true;
  reembedded: number;
  model: string;
}> {
  const pool = getPool();
  const limit = payload.limit ?? 200;

  const rows = payload.chunkIds?.length
    ? await pool.query("SELECT chunk_id, chunk_text FROM document_chunks WHERE chunk_id = ANY($1) ORDER BY chunk_id", [payload.chunkIds])
    : await pool.query("SELECT chunk_id, chunk_text FROM document_chunks ORDER BY chunk_id LIMIT $1", [limit]);

  let reembedded = 0;
  for (let i = 0; i < rows.rows.length; i += REEMBED_BATCH) {
    const batch = rows.rows.slice(i, i + REEMBED_BATCH);
    const embeddings = await embedTexts(batch.map((r) => r.chunk_text));
    for (let j = 0; j < batch.length; j++) {
      await pool.query("UPDATE document_chunks SET embedding = $1 WHERE chunk_id = $2", [
        embeddings[j],
        batch[j].chunk_id,
      ]);
      reembedded++;
    }
  }

  return { handled: true, reembedded, model: payload.model ?? EMBEDDING_MODEL };
}
