/**
 * Session persistence tests (Phase 5b.7) — saveTurn/loadRecentChats/deleteChat
 * round-trip against the live Postgres (ACS_TEST_DB=1).
 */
import { describe, it, expect, beforeEach } from "vitest";
import { saveTurn, loadRecentChats, deleteChat } from "../src/streaming/persist.js";
import type { ChatTurn } from "../src/streaming/registry.js";

function fakeTurn(overrides: Partial<ChatTurn> = {}): ChatTurn {
  const now = Date.now();
  return {
    chatId: overrides.chatId ?? "chat_test_1",
    conversationId: overrides.conversationId ?? "conv_test_1",
    session: null,
    messages: overrides.messages ?? [
      { role: "user", content: [{ type: "text", text: "reset my lg tv wifi" }] },
      { role: "assistant", content: [{ type: "text", text: "Press Settings > Network…" }] },
    ],
    status: overrides.status ?? "done",
    subscribers: new Set(),
    ring: [],
    seq: 0,
    createdAt: overrides.createdAt ?? now,
    finishedAt: overrides.finishedAt ?? now + 1000,
    messageCount: overrides.messages?.length ?? 2,
    detachBridge: null,
  };
}

describe("persist (live DB)", () => {
  const run = process.env.ACS_TEST_DB === "1" ? describe : describe.skip;

  run("round-trip", () => {
    beforeEach(async () => {
      const { getPool } = await import("../src/db/pool.js");
      await getPool().query("DELETE FROM chats WHERE chat_id LIKE 'chat_test_%'");
    });

    it("saveTurn → loadRecentChats returns the same chat + messages", async () => {
      const turn = fakeTurn();
      await saveTurn(turn);

      const loaded = await loadRecentChats(50);
      const mine = loaded.find((c) => c.chatId === turn.chatId);
      expect(mine).toBeTruthy();
      expect(mine!.status).toBe("done");
      expect(mine!.conversationId).toBe("conv_test_1");
      expect(mine!.messageCount).toBe(2);
      expect(mine!.messages).toHaveLength(2);
      expect(mine!.messages[0]).toMatchObject({ role: "user" });
      expect((mine!.messages[1] as { content: unknown[] }).content[0]).toMatchObject({
        type: "text",
        text: "Press Settings > Network…",
      });
    });

    it("re-saving a chat replaces its messages (idempotent)", async () => {
      const turn = fakeTurn();
      await saveTurn(turn);
      turn.messages = [{ role: "user", content: [{ type: "text", text: "only one" }] }];
      turn.messageCount = 1;
      await saveTurn(turn);

      const loaded = await loadRecentChats(50);
      const mine = loaded.find((c) => c.chatId === turn.chatId);
      expect(mine!.messages).toHaveLength(1);
      expect((mine!.messages[0] as { content: { text: string }[] }).content[0].text).toBe("only one");
    });

    it("deleteChat removes the chat and its messages", async () => {
      const turn = fakeTurn({ chatId: "chat_test_del" });
      await saveTurn(turn);
      await deleteChat(turn.chatId);

      const loaded = await loadRecentChats(50);
      expect(loaded.find((c) => c.chatId === turn.chatId)).toBeUndefined();
      const { getPool } = await import("../src/db/pool.js");
      const msgs = await getPool().query("SELECT count(*)::int AS n FROM chat_messages WHERE chat_id = $1", [turn.chatId]);
      expect(msgs.rows[0].n).toBe(0);
    });
  });
});
