/**
 * hooks/useChatStream.ts — opens the SSE stream for the active chat and feeds
 * events into chatStore.
 *
 * Token batching: `token`/`thinking` deltas arrive many times per second, so we
 * coalesce them per animation frame and flush ONE store update per frame (the
 * ui.md §3.4 requirement) instead of re-rendering per token.
 */
import { useEffect, useRef } from "react";
import { connectSse } from "../lib/sse";
import type { SseEventMap } from "../lib/types";
import { useChatStore } from "../stores/chatStore";

export function useChatStream(): void {
  const activeChatId = useChatStore((s) => s.activeChatId);
  const shouldStream = useChatStore((s) => s.shouldStream);

  // rAF batching buffers
  const tokenBuf = useRef("");
  const rafId = useRef<number | null>(null);

  const flush = () => {
    rafId.current = null;
    if (tokenBuf.current) {
      const delta = tokenBuf.current;
      tokenBuf.current = "";
      useChatStore.getState().appendTokens(delta);
    }
  };

  const scheduleFlush = () => {
    if (rafId.current === null) {
      rafId.current = requestAnimationFrame(flush);
    }
  };

  useEffect(() => {
    // A pending rAF from a previous chat must not leak into the next one.
    if (rafId.current !== null) {
      cancelAnimationFrame(rafId.current);
      rafId.current = null;
      tokenBuf.current = "";
    }
  }, [activeChatId]);

  useEffect(() => {
    if (!activeChatId || !shouldStream) return;

    const store = useChatStore.getState();
    store.setConnectionState("connecting");

    const conn = connectSse(`/api/chat/${activeChatId}/events`, {
      onEvent: (event, data) => {
        if (event === "token" || event === "thinking") {
          const d = data as SseEventMap["token"];
          tokenBuf.current += d.delta;
          scheduleFlush();
          return;
        }
        // Non-token events flush any pending tokens first, then apply directly.
        flush();
        store.applySseEvent(event, data);
      },
      onStateChange: (state) => useChatStore.getState().setConnectionState(state),
    });

    return () => {
      if (rafId.current !== null) cancelAnimationFrame(rafId.current);
      rafId.current = null;
      tokenBuf.current = "";
      conn.close();
    };
  }, [activeChatId, shouldStream]);
}
