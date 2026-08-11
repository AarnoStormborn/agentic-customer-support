/**
 * layout/SessionSidebar.tsx — history list (GET /api/sessions) grouped by day,
 * new-session button, per-item delete, active highlight. Collapsible.
 */
import { useEffect, useMemo } from "react";
import { useNavigate } from "react-router";
import { MessageSquarePlus, Trash2, X } from "lucide-react";
import { useSessionStore } from "../../stores/sessionStore";
import { useChatStore } from "../../stores/chatStore";
import { formatRelativeDay } from "../../lib/format";
import { Skeleton } from "../common/Spinner";

export function SessionSidebar({
  collapsed,
  onClose,
}: {
  collapsed: boolean;
  onClose: () => void;
}) {
  const { sessions, loading, error, fetchSessions, deleteSession } = useSessionStore();
  const activeChatId = useChatStore((s) => s.activeChatId);
  const openSession = useChatStore((s) => s.openSession);
  const newSession = useChatStore((s) => s.newSession);
  const navigate = useNavigate();

  useEffect(() => {
    void fetchSessions();
  }, [fetchSessions]);

  const grouped = useMemo(() => {
    const groups = new Map<string, typeof sessions>();
    for (const s of sessions) {
      const day = formatRelativeDay(s.createdAt);
      const arr = groups.get(day) ?? [];
      arr.push(s);
      groups.set(day, arr);
    }
    return [...groups.entries()];
  }, [sessions]);

  if (collapsed) return null;

  return (
    <aside className="flex h-full w-60 shrink-0 flex-col border-r border-border bg-surface">
      <header className="flex items-center gap-2 border-b border-border p-3">
        <button
          onClick={() => {
            newSession();
            navigate("/");
          }}
          className="flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-accent px-2 py-2 text-sm font-medium text-white hover:opacity-90"
        >
          <MessageSquarePlus size={15} /> New session
        </button>
        <button
          onClick={onClose}
          aria-label="Hide sidebar"
          className="rounded p-1.5 text-text-dim hover:bg-surface-2 hover:text-text"
        >
          <X size={15} />
        </button>
      </header>

      <div className="flex-1 overflow-y-auto p-2">
        {loading && (
          <div className="space-y-2 p-2">
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-5/6" />
            <Skeleton className="h-8 w-4/6" />
          </div>
        )}

        {error && <p className="p-3 text-xs text-danger">{error}</p>}

        {!loading && !error && sessions.length === 0 && (
          <p className="p-3 text-xs text-text-dim">No sessions yet — start a conversation.</p>
        )}

        {grouped.map(([day, items]) => (
          <div key={day} className="mb-3">
            <p className="px-2 pb-1 text-[11px] font-medium uppercase tracking-wide text-text-dim">
              {day}
            </p>
            <ul className="space-y-0.5">
              {items.map((s) => (
                <li key={s.chatId} className="group relative">
                  <button
                    onClick={() => {
                      void openSession(s.chatId, s.conversationId, s.status === "running");
                      navigate("/");
                    }}
                    className={`flex w-full items-start gap-2 rounded-lg px-2 py-2 text-left transition-colors ${
                      s.chatId === activeChatId
                        ? "bg-accent/15 text-text"
                        : "text-text-dim hover:bg-surface-2 hover:text-text"
                    }`}
                  >
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-[13px]">
                        {s.preview || "New session"}
                      </span>
                      <span className="mt-0.5 flex items-center gap-1.5 text-[11px] opacity-70">
                        {s.status === "running" && (
                          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-ok" />
                        )}
                        {s.status}
                        {s.messageCount > 0 && ` · ${s.messageCount} msgs`}
                      </span>
                    </span>
                  </button>
                  <button
                    onClick={() => void deleteSession(s.chatId)}
                    aria-label={`Delete ${s.preview || s.chatId}`}
                    className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded p-1 text-text-dim opacity-0 transition-opacity hover:text-danger group-hover:opacity-100"
                  >
                    <Trash2 size={13} />
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <footer className="border-t border-border p-3">
        <p className="text-[11px] leading-relaxed text-text-dim">
          Sessions live in the backend registry (in-memory — they reset on server restart).
        </p>
      </footer>
    </aside>
  );
}
