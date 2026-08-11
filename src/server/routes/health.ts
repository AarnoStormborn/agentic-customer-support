/**
 * src/server/routes/health.ts — GET /health.
 *
 * Liveness/readiness: tries a Postgres `SELECT 1` and a Redis PING, reports each as
 * "ok"/"down", and returns 200 either way so load balancers can see partial state.
 */
import type { FastifyPluginAsync } from "fastify";
import { Pool, type PoolClient } from "pg";
import { Redis } from "ioredis";
import { env } from "../../config/env.js";

const PG_CONNECT_TIMEOUT_MS = 1500;
const REDIS_CONNECT_TIMEOUT_MS = 1500;

// Own local pool (src/db/pool.ts is owned by retrieval-core — see CONTRACT-NOTES.md).
let pool: Pool | null = null;
function healthPool(): Pool {
  if (!pool) {
    pool = new Pool({
      connectionString: env.DATABASE_URL,
      connectionTimeoutMillis: PG_CONNECT_TIMEOUT_MS,
      idleTimeoutMillis: 2000,
      max: 2,
    });
    pool.on("error", () => {
      // Idle-client errors must not crash the server.
    });
  }
  return pool;
}

async function checkPostgres(): Promise<boolean> {
  let client: PoolClient | undefined;
  try {
    client = await healthPool().connect();
    await client.query("SELECT 1");
    return true;
  } catch {
    return false;
  } finally {
    client?.release();
  }
}

async function checkRedis(): Promise<boolean> {
  const client = new Redis(env.REDIS_URL, {
    connectTimeout: REDIS_CONNECT_TIMEOUT_MS,
    maxRetriesPerRequest: 1,
  });
  try {
    const pong = await client.ping();
    return pong === "PONG";
  } catch {
    return false;
  } finally {
    client.disconnect();
  }
}

export const healthRoutes: FastifyPluginAsync = async (app) => {
  app.get(
    "/health",
    // No rate limit: health checks shouldn't trip alerts when the LB polls.
    { config: { rateLimit: false } },
    async () => {
      const [postgres, redis] = await Promise.all([checkPostgres(), checkRedis()]);
      return {
        status: postgres && redis ? "ok" : "degraded",
        uptime: process.uptime(),
        deps: {
          postgres: postgres ? "ok" : "down",
          redis: redis ? "ok" : "down",
        },
      };
    },
  );
};
