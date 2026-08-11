/**
 * src/server/index.ts — boot: Fastify listen + BullMQ worker start.
 * Run with `npm run dev` (tsx watch) on PORT/HOST from env.
 */
import { env } from "../config/env.js";
import { buildApp } from "./app.js";
import { startWorker } from "../queue/worker.js";

async function main(): Promise<void> {
  const app = await buildApp();
  const worker = startWorker(app.log);

  const shutdown = async (signal: string): Promise<void> => {
    app.log.info({ signal }, "shutting down");
    await worker.close();
    await app.close();
    process.exit(0);
  };
  process.on("SIGINT", () => void shutdown("SIGINT"));
  process.on("SIGTERM", () => void shutdown("SIGTERM"));

  try {
    await app.listen({ port: env.PORT, host: env.HOST });
  } catch (err) {
    app.log.error(err);
    await worker.close();
    process.exit(1);
  }
}

void main();
