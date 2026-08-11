/**
 * src/db/pool.ts — Postgres async connection pool (v1 lesson #3: async pools at
 * module scope, never a sync engine inside async tools).
 *
 * Single module-scope singleton so every caller (ingest, hybrid search, migrate)
 * shares one pool. Idle-client errors are logged, never swallowed.
 *
 * Uses pg.Pool — connections are acquired per-query and returned automatically;
 * the pool keeps the event loop free (v1 used sqlalchemy.create_engine inside
 * async tools, blocking the loop).
 */
import pg from "pg";
import "dotenv/config";

const DEFAULT_DATABASE_URL = "postgresql://acs:acs@localhost:5432/acs";

let pool: pg.Pool | null = null;

/** Get the shared pg.Pool for DATABASE_URL (created lazily on first use). */
export function getPool(): pg.Pool {
  if (!pool) {
    pool = new pg.Pool({
      connectionString: process.env.DATABASE_URL ?? DEFAULT_DATABASE_URL,
      max: 10,
    });
    // Without this handler an idle client error crashes the process.
    pool.on("error", (err) => {
      console.error("[db] idle client error:", err.message);
    });
  }
  return pool;
}

/** Close the pool (used by CLIs so the process can exit cleanly). */
export async function closePool(): Promise<void> {
  if (pool) {
    await pool.end();
    pool = null;
  }
}
