/**
 * src/streaming/persist.ts — chat/session persistence (best-effort, Postgres).
 *
 * The live registry is in-memory (fast path); this module is the durability
 * layer. Every finished turn is written to `chats` + `chat_messages`; on boot
 * the registry rehydrates from `loadRecentChats`.
 *
 * All writes are best-effort: a DB failure logs and is swallowed so the chat
 * API keeps working (unit CI jobs have no Postgres). Round-trip correctness is
 * covered by DB-guarded tests (ACS_TEST_DB=1).
 */
import { getPool } from "../db/pool.js";
import type { ChatTurn, TurnStatus } from "./registry.js";

const MAX_CHATS_LOAD = 100;
const MAX_MESSAGES_PER_CHAT = 200;

export interface StoredChat {
  chatId: string;
  conversationId: string;
  status: TurnStatus;
  createdAt: number;
  finishedAt: number | null;
  messageCount: number;
  messages: unknown[];
}

/** Serialize a live turn's messages into DB rows. */
function serializeMessages(messages: unknown[]): { role: string; content: unknown }[] {
  return messages.slice(0, MAX_MESSAGES_PER_CHAT).map((m) => {
    const msg = m as { role?: string; content?: unknown };
    return { role: msg.role ?? "assistant", content: msg.content ?? { type: "text", text: "" } };
  });
}

/** Upsert a turn's chat row + replace its messages (idempotent per chatId). */
export async function saveTurn(turn: ChatTurn): Promise<void> {
  const pool = getPool();
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    await client.query(
      `INSERT INTO chats (chat_id, conversation_id, status, created_at, finished_at, message_count)
       VALUES ($1, $2, $3, to_timestamp($4 / 1000.0), $5::timestamptz, $6)
       ON CONFLICT (chat_id) DO UPDATE SET
         status = EXCLUDED.status,
         finished_at = EXCLUDED.finished_at,
         message_count = EXCLUDED.message_count`,
      [
        turn.chatId,
        turn.conversationId,
        turn.status,
        turn.createdAt,
        turn.finishedAt ? new Date(turn.finishedAt).toISOString() : null,
        turn.messageCount,
      ],
    );
    await client.query("DELETE FROM chat_messages WHERE chat_id = $1", [turn.chatId]);
    for (const [i, m] of serializeMessages(turn.messages).entries()) {
      await client.query(
        `INSERT INTO chat_messages (chat_id, role, content, turn_index)
         VALUES ($1, $2, $3::jsonb, $4)`,
        [turn.chatId, m.role, JSON.stringify(m.content), i],
      );
    }
    await client.query("COMMIT");
  } catch (err) {
    await client.query("ROLLBACK").catch(() => {});
    // Best-effort: never let persistence break the live chat.
    console.error(`[persist] saveTurn(${turn.chatId}) failed:`, (err as Error).message);
  } finally {
    client.release();
  }
}

/** Load the most recent chats (with messages) for boot hydration. */
export async function loadRecentChats(limit: number = MAX_CHATS_LOAD): Promise<StoredChat[]> {
  const pool = getPool();
  const chats = await pool.query(
    `SELECT chat_id, conversation_id, status,
            extract(epoch from created_at)::bigint * 1000 AS created_at,
            extract(epoch from finished_at)::bigint * 1000 AS finished_at,
            message_count
     FROM chats
     ORDER BY created_at DESC
     LIMIT $1`,
    [limit],
  );
  if (chats.rows.length === 0) return [];

  const ids = chats.rows.map((r) => r.chat_id);
  const messages = await pool.query(
    `SELECT chat_id, role, content, turn_index
     FROM chat_messages
     WHERE chat_id = ANY($1::text[])
     ORDER BY chat_id, turn_index`,
    [ids],
  );

  const byChat = new Map<string, unknown[]>();
  for (const row of messages.rows) {
    const list = byChat.get(row.chat_id) ?? [];
    list.push({ role: row.role, content: row.content });
    byChat.set(row.chat_id, list);
  }

  return chats.rows.map((r) => ({
    chatId: r.chat_id,
    conversationId: r.conversation_id,
    status: r.status as TurnStatus,
    createdAt: Number(r.created_at),
    finishedAt: r.finished_at ? Number(r.finished_at) : null,
    messageCount: r.message_count,
    messages: byChat.get(r.chat_id) ?? [],
  }));
}

/** Remove a chat + its messages (best-effort). */
export async function deleteChat(chatId: string): Promise<void> {
  try {
    const pool = getPool();
    await pool.query("DELETE FROM chats WHERE chat_id = $1", [chatId]);
  } catch (err) {
    console.error(`[persist] deleteChat(${chatId}) failed:`, (err as Error).message);
  }
}
