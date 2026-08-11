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
