/**
 * src/retrieval/index.ts — retrieval-core public API (integration contract).
 *
 * Consumed by agent-runtime tools (rag_tool, sql_tool) and api-streaming read
 * routes. Signatures are the contract; do not change without a CONTRACT-NOTES.md
 * entry.
 */
export type {
  HybridResult,
  HybridSource,
  HybridSearchOptions,
  HybridSearchResponse,
} from "./hybrid.js";
export { searchHybrid } from "./hybrid.js";
export { embedTexts, embeddingsEnabled, EMBEDDING_MODEL, embeddingDim } from "./embed.js";
export { runSchema } from "../db/migrate.js";
export { ingestTickets, ingestManuals } from "./ingest.js";
export type { IngestSummary, IngestOptions, TicketSource } from "./ingest.js";
export { getPool, closePool } from "../db/pool.js";
