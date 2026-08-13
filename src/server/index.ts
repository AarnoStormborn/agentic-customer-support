/**
 * src/server/index.ts — boot: Fastify listen + BullMQ worker start.
 * Run with `npm run dev` (tsx watch) on PORT/HOST from env.
 */
import { env } from "../config/env.js";
import { buildApp } from "./app.js";
import { startWorker } from "../queue/worker.js";
import { ChatRegistry } from "../streaming/registry.js";
import { loadRecentChats } from "../streaming/persist.js";

async function main(): Promise<void> {
  // One registry for live turns + rehydrated history (persisted chats).
  const registry = new ChatRegistry();
  const app = await buildApp({ registry });
  const worker = startWorker(app.log);

  // Rehydrate recent conversations from Postgres so the sidebar + history
  // survive restarts (best-effort; a missing/empty store is fine).
  try {
    const stored = await loadRecentChats();
    for (const chat of stored) registry.hydrate(chat);
    app.log.info({ rehydrated: stored.length }, "rehydrated chats from store");
  } catch (err) {
    app.log.warn({ err }, "chat rehydration skipped (store unavailable?)");
  }

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
