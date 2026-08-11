/**
 * layout/ContextPanel.tsx — right rail: ToolActivityFeed + Sources of the
 * latest answered assistant message. Collapsible; on small screens it becomes
 * a fixed overlay toggled from the chat header.
 */
import { Activity, BookOpenCheck } from "lucide-react";
import { X } from "lucide-react";
import { useChatStore } from "../../stores/chatStore";
import { ToolActivityFeed } from "../chat/ToolActivityFeed";
import { SourceRefCard } from "../chat/SourcesPanel";

export function ContextPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const messages = useChatStore((s) => s.messages);
  const isStreaming = useChatStore((s) => s.isStreaming);

  // Sources of the most recent assistant message that has them.
  const lastWithSources = [...messages].reverse().find((m) => m.sources && m.sources.length > 0);

  const body = (
    <div className="flex h-full flex-col gap-5 overflow-y-auto p-4">
      <section>
        <h3 className="mb-2 flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-text-dim">
          <Activity size={13} /> Tool activity
          {isStreaming && <span className="ml-auto h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />}
        </h3>
        <ToolActivityFeed />
      </section>

      <section>
        <h3 className="mb-2 flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-text-dim">
          <BookOpenCheck size={13} /> Sources
        </h3>
        {lastWithSources ? (
          <div className="space-y-1.5">
            {lastWithSources.sources?.map((s, i) => (
              <SourceRefCard key={i} source={s} index={i + 1} />
            ))}
          </div>
        ) : (
          <p className="text-xs text-text-dim">
            Sources from retrieved manuals and tickets appear here when the agent answers.
          </p>
        )}
      </section>
    </div>
  );

  return (
    <>
      {/* Desktop: static right rail ≥ xl */}
      <aside className="hidden w-[320px] shrink-0 border-l border-border bg-surface xl:block">
        {body}
      </aside>

      {/* Small screens: fixed drawer when opened */}
      {open && (
        <div className="fixed inset-0 z-40 flex justify-end xl:hidden">
          <div className="absolute inset-0 bg-black/50" onClick={onClose} />
          <aside className="relative h-full w-[320px] max-w-[85vw] border-l border-border bg-surface shadow-panel">
            <button
              onClick={onClose}
              aria-label="Close context panel"
              className="absolute right-3 top-3 rounded p-1 text-text-dim hover:bg-surface-2 hover:text-text"
            >
              <X size={16} />
            </button>
            {body}
          </aside>
        </div>
      )}
    </>
  );
}
