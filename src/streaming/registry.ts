/**
 * src/streaming/registry.ts — ChatRegistry: per-chat state for streaming.
 *
 * One ChatTurn per chat id:
 *   - `session`: the SupportRuntime (mock today, real agent after integration)
 *   - `subscribers`: SSE / WebSocket clients listening for live events
 *   - `ring`: ring buffer (last ~200 events) so a reconnecting SSE client can
 *     replay missed events via Last-Event-ID
 *
 * All event→client streaming goes through this registry + the bridge (architecture
 * rule 4: one bridge module owns the SDK→SSE mapping).
 */
import type { SupportRuntime } from "../runtime/mock.js";
import { env } from "../config/env.js";

/** SSE event types — exactly the §2.3 schema (plus the optional extras). */
export type SSEEventType =
  | "turn_start"
  | "token"
  | "thinking"
  | "tool_start"
  | "tool_update"
  | "tool_end"
  | "turn_end"
  | "done"
  | "error"
  | "retry_start"
  | "retry_end"
  | "queue_update";

/** One event as stored in the ring buffer / delivered to subscribers. */
export interface SSEEnvelope {
  /** Monotonic per-chat sequence number → the SSE `id:` field for Last-Event-ID. */
  id: number;
  event: SSEEventType;
  /** Always a JSON-serializable object (§2.3: `data` is a JSON object). */
  data: unknown;
}

export type TurnStatus = "running" | "done" | "error" | "canceled";

export type Subscriber = (env: SSEEnvelope) => void;

export interface ChatTurn {
  chatId: string;
  conversationId: string;
  session: SupportRuntime;
  status: TurnStatus;
  subscribers: Set<Subscriber>;
  /** Circular buffer of recent events (cap: env.RING_BUFFER_SIZE). */
  ring: SSEEnvelope[];
  /** Next SSE sequence id for this chat. */
  seq: number;
  createdAt: number;
  finishedAt: number | null;
  /** Detach function for the session's bridge subscription (set by the chat route). */
  detachBridge: (() => void) | null;
}

export interface CreateTurnParams {
  chatId: string;
  conversationId: string;
  session: SupportRuntime;
}

export class ChatRegistry {
  private readonly turns = new Map<string, ChatTurn>();

  create(params: CreateTurnParams): ChatTurn {
    const turn: ChatTurn = {
      chatId: params.chatId,
      conversationId: params.conversationId,
      session: params.session,
      status: "running",
      subscribers: new Set(),
      ring: [],
      seq: 0,
      createdAt: Date.now(),
      finishedAt: null,
      detachBridge: null,
    };
    this.turns.set(params.chatId, turn);
    return turn;
  }

  get(chatId: string): ChatTurn | undefined {
    return this.turns.get(chatId);
  }

  has(chatId: string): boolean {
    return this.turns.has(chatId);
  }

  /** Number of live subscribers across all chats (for debugging / caps). */
  get subscriberCount(): number {
    let n = 0;
    for (const turn of this.turns.values()) n += turn.subscribers.size;
    return n;
  }

  mark(chatId: string, status: TurnStatus): void {
    const turn = this.turns.get(chatId);
    if (!turn) return;
    turn.status = status;
    turn.finishedAt = Date.now();
  }

  /** Assign the next seq, append to the ring buffer, and fan out to subscribers. */
  emit(chatId: string, event: SSEEventType, data: unknown): void {
    const turn = this.turns.get(chatId);
    if (!turn) return;
    turn.seq += 1;
    const envelope: SSEEnvelope = { id: turn.seq, event, data };
    turn.ring.push(envelope);
    if (turn.ring.length > env.RING_BUFFER_SIZE) {
      turn.ring.splice(0, turn.ring.length - env.RING_BUFFER_SIZE);
    }
    for (const subscriber of [...turn.subscribers]) {
      try {
        subscriber(envelope);
      } catch (err) {
        console.error(`[registry:${chatId}] subscriber error:`, err);
      }
    }
  }

  /** Subscribe to live events. Returns an unsubscribe function. */
  subscribe(chatId: string, subscriber: Subscriber): () => void {
    const turn = this.turns.get(chatId);
    if (!turn) throw new Error(`unknown chat: ${chatId}`);
    turn.subscribers.add(subscriber);
    return () => {
      turn.subscribers.delete(subscriber);
    };
  }

  /** Events with id > `afterId` (for Last-Event-ID replay). */
  replay(chatId: string, afterId: number): SSEEnvelope[] {
    const turn = this.turns.get(chatId);
    if (!turn) return [];
    return turn.ring.filter((env) => env.id > afterId);
  }

  remove(chatId: string): void {
    this.turns.delete(chatId);
  }
}
