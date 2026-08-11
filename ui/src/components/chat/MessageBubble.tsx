/**
 * chat/MessageBubble.tsx — one message. User: right-aligned accent bubble.
 * Assistant: surface card with markdown, streaming caret, sources + error tag.
 */
import { AlertTriangle, Ban, User } from "lucide-react";
import type { ChatMessage } from "../../stores/chatStore";
import { Markdown } from "../common/Markdown";
import { Badge } from "../common/Badge";
import { SourceRefCard } from "./SourcesPanel";

export function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="flex max-w-[85%] items-end gap-2">
          <div className="rounded-2xl rounded-br-sm bg-accent px-4 py-2.5 text-[15px] leading-relaxed text-white shadow-panel">
            {message.text}
          </div>
          <div className="mb-0.5 rounded-full bg-surface-2 p-1 text-text-dim">
            <User size={14} />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="w-full max-w-[92%]">
        <div
          className={`rounded-2xl rounded-bl-sm border bg-surface px-4 py-3 shadow-panel ${
            message.status === "cancelled" ? "opacity-60" : "border-border"
          }`}
        >
          {message.text ? (
            <Markdown
              onCitationClick={(n) => {
                document
                  .getElementById(`sources-${message.id}`)
                  ?.scrollIntoView({ behavior: "smooth", block: "nearest" });
                void n;
              }}
            >
              {message.text}
            </Markdown>
          ) : (
            message.status === "streaming" && (
              <div className="flex items-center gap-2 py-1 text-sm text-text-dim">
                Thinking <span className="animate-caret">▍</span>
              </div>
            )
          )}

          {message.status === "streaming" && message.text && (
            <span className="animate-caret ml-0.5" aria-label="streaming">
              ▍
            </span>
          )}

          {message.status === "error" && message.error && (
            <div className="mt-2 flex items-start gap-2 rounded-lg border border-danger/30 bg-danger/10 px-3 py-2">
              <AlertTriangle size={14} className="mt-0.5 shrink-0 text-danger" />
              <div className="text-xs">
                <p className="font-medium text-danger">Stream stopped</p>
                <p className="mt-0.5 text-text-dim">{message.error.message}</p>
              </div>
            </div>
          )}

          {message.status === "cancelled" && (
            <div className="mt-2 flex items-center gap-1.5 text-xs text-text-dim">
              <Ban size={12} /> Cancelled
            </div>
          )}

          {message.status === "done" && message.usage && (
            <p className="mt-2 text-[11px] text-text-dim/70">
              {message.usage.inputTokens ?? "—"} in · {message.usage.outputTokens ?? "—"} out
              {message.usage.totalCostUsd !== undefined &&
                ` · $${message.usage.totalCostUsd.toFixed(4)}`}
            </p>
          )}
        </div>

        {message.sources && message.sources.length > 0 && (
          <div id={`sources-${message.id}`} className="mt-2 space-y-1.5">
            <p className="flex items-center gap-1.5 px-1 text-[11px] font-medium uppercase tracking-wide text-text-dim">
              Sources
              <Badge tone="accent">{message.sources.length}</Badge>
            </p>
            {message.sources.map((s, i) => (
              <SourceRefCard key={i} source={s} index={i + 1} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
