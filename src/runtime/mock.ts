/**
 * src/runtime/mock.ts — LOCAL MOCK of the runtime contract (api-streaming track).
 *
 * The real implementation (`createSupportRuntime` in `src/runtime/index.ts`) is built
 * by the agent-runtime track. This file re-declares the contract interface locally and
 * provides a fake session that emits a believable SDK-style event sequence so the
 * server can stream end-to-end WITHOUT the real agent.
 *
 * Integration: swap `import { createSupportRuntime } from "../../runtime/mock.js"` in
 * `src/server/routes/chat.ts` (or pass a different factory to `buildApp`) for the real
 * module. Signatures are identical — verified by `tsc --noEmit` after merge.
 *
 * Event names mirror the pi SDK events the bridge maps (§3.4 of the design doc):
 *   turn_start · message_update (text_delta / thinking_delta) · tool_execution_start ·
 *   tool_execution_update · tool_execution_end · queue_update · turn_end · agent_settled
 * plus auto_retry_start / auto_retry_end (never emitted by the mock, but supported).
 */

/** Event delivered to `subscribe()` listeners — raw SDK event shape. */
export interface MockRuntimeEvent {
  type: string;
  [key: string]: unknown;
}

/** The runtime contract (mirror of `src/runtime/index.ts` in the integration contract). */
export interface SupportRuntime {
  prompt(text: string, opts?: { images?: unknown[] }): Promise<void>;
  steer(text: string): Promise<void>;
  abort(): Promise<void>;
  /** Subscribe to raw SDK events. Returns an unsubscribe function. */
  subscribe(fn: (event: unknown) => void): () => void;
  getLastMessages(): unknown[];
  dispose(): void;
}

export interface CreateSupportRuntimeOptions {
  model?: string; // "provider/model" from PI_MODEL
  chatId?: string;
  sessionDir?: string; // undefined = in-memory
}

export interface MockMessage {
  role: "user" | "assistant";
  content: string;
}

const TOKEN_DELAY_MS = 35;
const TOOL_DELAY_MS = 150;

function randomId(prefix: string): string {
  return `${prefix}_${Date.now().toString(36)}${Math.random()
    .toString(36)
    .slice(2, 8)}`;
}

/** Build the canned answer, echoing the user's question and folding in steers. */
function buildAnswer(userText: string, steerTexts: string[]): string[] {
  const paragraphs = [
    `Here's what I found for "${userText.slice(0, 80)}":`,
    "1. On your LG Smart TV, open Settings → General → Network.",
    "2. Choose Wi-Fi Connection, select your network, and re-enter the password.",
    "3. If it still fails, power-cycle the router and retry (our KB shows a 74% success rate for this fix).",
  ];
  for (const steer of steerTexts) {
    paragraphs.push(`Also noted from your steering message: ${steer}`);
  }
  // Split into word-level deltas so the SSE stream shows tokens arriving.
  return paragraphs.join("\n\n").split(/(?<=\s)/);
}

const MOCK_SOURCES: Record<string, unknown>[] = [
  {
    type: "kb",
    title: "LG OLED TV User Guide — Network Settings",
    docName: "lg-oled-user-guide.pdf",
    sectionPath: "4.2 Wi-Fi Connection",
    page: 42,
    score: 0.91,
    url: null,
  },
  {
    type: "sql",
    title: "ticket #10293",
    row: {
      id: 10293,
      ticket_type: "Technical issue",
      ticket_priority: "High",
      ticket_status: "Open",
    },
    score: null,
  },
];

/**
 * Create a fake support session that, on `prompt()`, emits the full SDK event sequence
 * (turn_start → tokens → tool call → tokens → turn_end → agent_settled with sources).
 */
export async function createSupportRuntime(
  opts: CreateSupportRuntimeOptions = {},
): Promise<SupportRuntime> {
  const chatId = opts.chatId ?? randomId("mock_chat");
  const listeners = new Set<(event: MockRuntimeEvent) => void>();

  let disposed = false;
  let running = false;
  let aborted = false;
  let turnIndex = 0;
  let lastMessages: MockMessage[] = [];
  const steeringQueue: string[] = [];

  const emit = (event: MockRuntimeEvent): void => {
    if (disposed) return;
    for (const fn of [...listeners]) {
      try {
        fn(event);
      } catch (err) {
        // A failing subscriber must not kill the mock's turn loop.
        console.error(`[mock:${chatId}] subscriber error:`, err);
      }
    }
  };

  const delay = (ms: number): Promise<void> =>
    new Promise((resolve) => setTimeout(resolve, ms));

  async function runTurn(text: string, steerTexts: string[]): Promise<void> {
    turnIndex += 1;
    const ti = turnIndex;

    emit({ type: "turn_start", turnIndex: ti });

    // Thinking delta first (optional extra event the UI can render).
    emit({
      type: "message_update",
      assistantMessageEvent: { type: "thinking_delta", delta: "Searching knowledge base…" },
    });

    // Opening tokens.
    for (const chunk of ["Let me ", "look that ", "up for you.\n\n"]) {
      if (aborted) throw abortError();
      emit({
        type: "message_update",
        assistantMessageEvent: { type: "text_delta", delta: chunk },
      });
      await delay(TOKEN_DELAY_MS);
    }

    // Tool call: kb_search (the specialist sub-agent in the real system).
    const toolCallId = randomId("call");
    emit({
      type: "tool_execution_start",
      toolCallId,
      toolName: "kb_search",
      args: { query: text.slice(0, 80) },
    });
    emit({
      type: "tool_execution_update",
      toolCallId,
      partialResult: "Querying document_chunks (hybrid RRF)…",
    });
    await delay(TOOL_DELAY_MS);
    if (aborted) throw abortError();
    emit({
      type: "tool_execution_end",
      toolCallId,
      toolName: "kb_search",
      isError: false,
      durationMs: TOOL_DELAY_MS,
      summary: "3 chunks retrieved from knowledge base",
    });

    // Answer tokens.
    for (const chunk of buildAnswer(text, steerTexts)) {
      if (aborted) throw abortError();
      emit({
        type: "message_update",
        assistantMessageEvent: { type: "text_delta", delta: chunk },
      });
      await delay(TOKEN_DELAY_MS);
    }

    emit({ type: "queue_update", steering: 0, followUp: 0 });
    emit({ type: "turn_end", turnIndex: ti });

    const fullAnswer = buildAnswer(text, steerTexts).join("");
    lastMessages = [{ role: "user", content: text }, { role: "assistant", content: fullAnswer }];

    // agent_settled carries `sources` in the mock. NOTE: the bridge reads
    // event.sources when present — see CONTRACT-NOTES.md (the real runtime should
    // attach sources here too, or the bridge reads session state instead).
    emit({
      type: "agent_settled",
      turnIndex: ti,
      sources: MOCK_SOURCES,
      usage: { inputTokens: 2140, outputTokens: 318, totalCostUsd: 0.0042 },
    });
  }

  function abortError(): Error {
    const err = new Error("Turn canceled by user") as Error & { code: string };
    err.code = "canceled";
    return err;
  }

  return {
    async prompt(text: string): Promise<void> {
      if (disposed) throw new Error("runtime disposed");
      if (running) throw new Error("a turn is already running on this session");
      running = true;
      aborted = false;
      const steers = steeringQueue.splice(0);
      try {
        await runTurn(text, steers);
      } catch (err) {
        if (aborted) throw abortError();
        throw err;
      } finally {
        running = false;
      }
    },

    async steer(text: string): Promise<void> {
      steeringQueue.push(text);
      // Emit queue depth so clients can render the steering state (§2.3 queue_update).
      emit({ type: "queue_update", steering: steeringQueue.length, followUp: 0 });
    },

    async abort(): Promise<void> {
      aborted = true;
    },

    subscribe(fn: (event: unknown) => void): () => void {
      const wrapped = fn as (event: MockRuntimeEvent) => void;
      listeners.add(wrapped);
      return () => listeners.delete(wrapped);
    },

    getLastMessages(): unknown[] {
      return lastMessages;
    },

    dispose(): void {
      disposed = true;
      listeners.clear();
      steeringQueue.length = 0;
    },
  };
}
