/**
 * Sources collector — pure logic extracted from SupportRuntimeImpl.subscribe
 * so it can be unit-tested without a live pi SDK session.
 *
 * Watches an SDK event stream, accumulates `tool_execution_end` result
 * `details.sources`, and attaches them (deduped + capped) to the `agent_settled`
 * event that the api-streaming bridge reads to build the SSE `done` payload.
 */
import { MAX_DONE_SOURCES } from "../config/limits.js";

export interface EnrichedEvent {
  type: string;
  [key: string]: unknown;
}

export interface SourceLike {
  type?: unknown;
  title?: unknown;
  row?: { ticket_id?: unknown };
}

/** Stable identity for dedupe (kb: title, sql: title + row.ticket_id, ...). */
export function sourceKey(s: SourceLike): string {
  const rowId = s.row?.ticket_id;
  return `${String(s.type ?? "")}:${String(s.title ?? "")}:${String(rowId ?? "")}`;
}

/**
 * Build an event enricher. `onSettled(sources)` is called exactly once when an
 * `agent_settled` event passes through, with the sources collected since the
 * last `agent_start`. The handler's return value is what `handle()` returns for
 * that event (pass the event through otherwise).
 */
export function createSourceEnricher<T>(
  onSettled: (sources: SourceLike[]) => T,
): {
  handle(event: EnrichedEvent): T | EnrichedEvent;
} {
  let collected: SourceLike[] = [];
  let seen = new Set<string>();

  return {
    handle(event: EnrichedEvent): T | EnrichedEvent {
      if (event.type === "tool_execution_end") {
        const details = (event as { result?: { details?: { sources?: SourceLike[] } } }).result?.details;
        if (Array.isArray(details?.sources)) {
          for (const s of details.sources) {
            const k = sourceKey(s);
            if (k && !seen.has(k)) {
              seen.add(k);
              collected.push(s);
            }
          }
          if (collected.length > MAX_DONE_SOURCES) collected = collected.slice(0, MAX_DONE_SOURCES);
        }
        return event;
      }
      if (event.type === "agent_start") {
        collected = [];
        seen = new Set();
        return event;
      }
      if (event.type === "agent_settled") {
        const sources = collected;
        collected = [];
        seen = new Set();
        return onSettled(sources);
      }
      return event;
    },
  };
}
