/**
 * chat/Composer.tsx — auto-growing textarea, Enter=send / Shift+Enter=newline,
 * Stop button while streaming (POST /api/chat/:id/cancel). Consumes the
 * chatStore `pendingDraft` prefill from Tickets/Manuals ("Ask the agent…").
 */
import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { ArrowUp, Square } from "lucide-react";
import { useChatStore } from "../../stores/chatStore";
import { Button } from "../common/Button";

export function Composer() {
  const isStreaming = useChatStore((s) => s.isStreaming);
  const send = useChatStore((s) => s.send);
  const stopStreaming = useChatStore((s) => s.stopStreaming);
  const consumeDraft = useChatStore((s) => s.consumeDraft);

  const [text, setText] = useState("");
  const taRef = useRef<HTMLTextAreaElement>(null);

  // Prefill from other routes ("Ask about this chunk/ticket").
  useEffect(() => {
    const draft = consumeDraft();
    if (draft) {
      setText(draft);
      taRef.current?.focus();
    }
  }, [consumeDraft]);

  // Auto-grow up to ~8 lines.
  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`;
  }, [text]);

  const submit = () => {
    const trimmed = text.trim();
    if (!trimmed || isStreaming) return;
    setText("");
    void send(trimmed);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="border-t border-border bg-surface/60 p-3">
      <div className="mx-auto flex max-w-3xl items-end gap-2">
        <div className="relative flex-1 rounded-xl border border-border bg-surface focus-within:border-accent">
          <textarea
            ref={taRef}
            rows={1}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={
              isStreaming ? "Agent is working…" : "Ask about manuals, tickets, or the web…"
            }
            disabled={isStreaming}
            className="max-h-40 w-full resize-none bg-transparent px-3.5 py-2.5 text-sm text-text placeholder:text-text-dim focus:outline-none"
          />
        </div>

        {isStreaming ? (
          <Button variant="danger" onClick={() => void stopStreaming()} aria-label="Stop">
            <Square size={14} /> Stop
          </Button>
        ) : (
          <Button
            variant="primary"
            onClick={submit}
            disabled={!text.trim()}
            aria-label="Send"
            className="px-3"
          >
            <ArrowUp size={16} />
          </Button>
        )}
      </div>
      <p className="mx-auto mt-1.5 max-w-3xl text-[11px] text-text-dim/70">
        Enter to send · Shift+Enter for a new line · follow-ups continue the same conversation
      </p>
    </div>
  );
}
