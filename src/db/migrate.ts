/**
 * src/db/migrate.ts — applies src/db/schema.sql.
 * DDL lives in a committed .sql file; this module is the only way it runs.
 * Agent tools never execute DDL (AGENTS.md: SQL is read-only allowlist).
 */
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { getPool } from "./pool.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

/**
 * Apply the schema idempotently (all statements use IF NOT EXISTS).
 * Runs as one multi-statement query (no parameters → pg simple-query protocol).
 */
export async function runSchema(): Promise<void> {
  const sql = await readFile(join(__dirname, "schema.sql"), "utf8");
  const pool = getPool();
  await pool.query(sql);
  await reconcileEmbeddingDim(pool);
}

/**
 * Keep the document_chunks.embedding column + HNSW index at the ACTIVE backend's
 * dimension. schema.sql pins vector(1536) for OpenAI; an offline backend (Ollama
 * nomic-embed-text, 768) or EMBEDDING_DIM override reconciles here. After a dim
 * change, re-run `npm run ingest -- --manuals` to re-embed (upserts replace
 * vectors; the old ones are truncated by the cast).
 */
async function reconcileEmbeddingDim(pool: import("pg").Pool): Promise<void> {
  const { embeddingDim } = await import("../retrieval/embed.js");
  const dim = embeddingDim();
  const { rows } = await pool.query(
    `SELECT atttypmod FROM pg_attribute
     WHERE attrelid = 'document_chunks'::regclass AND attname = 'embedding' AND NOT attisdropped`,
  );
  const current = rows[0] ? Number(rows[0].atttypmod) : null; // pgvector: atttypmod == dim
  if (current === dim) return;

  console.log(`[migrate] embedding dim ${current ?? "?"} → ${dim} (recreating column + HNSW index)`);
  await pool.query("DROP INDEX IF EXISTS document_chunks_embedding_hnsw_idx");
  // pgvector <0.7 can't resize via cast — drop + re-add the column (chunks are
  // re-embedded by `npm run ingest -- --manuals` afterwards).
  await pool.query("ALTER TABLE document_chunks DROP COLUMN IF EXISTS embedding");
  await pool.query(`ALTER TABLE document_chunks ADD COLUMN embedding vector(${dim})`);
  await pool.query(
    `CREATE INDEX document_chunks_embedding_hnsw_idx ON document_chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128)`,
  );
}

/** CLI entry: `tsx src/db/migrate.ts` applies the schema and reports extensions. */
const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  runSchema()
    .then(async () => {
      const { rows } = await getPool().query(
        "SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector','pg_trgm') ORDER BY 1",
      );
      console.log("[migrate] schema applied. extensions:", rows.map((r) => `${r.extname}@${r.extversion}`).join(", "));
      const tables = await getPool().query(
        "SELECT tablename FROM pg_tables WHERE schemaname='public' AND tablename IN ('tickets','documents','document_chunks') ORDER BY 1",
      );
      console.log("[migrate] tables:", tables.rows.map((r) => r.tablename).join(", "));
      await poolEnd();
    })
    .catch((err) => {
      console.error("[migrate] failed:", err.message);
      process.exitCode = 1;
    });
}

async function poolEnd(): Promise<void> {
  const { closePool } = await import("./pool.js");
  await closePool();
}
