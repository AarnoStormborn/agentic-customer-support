/**
 * chat/ToolActivityFeed.tsx — the "watch the agent think" stack (ui.md §2.2):
 * one card per tool call, running → done/error, with elapsed ms + summary.
 * Rendered in the ContextPanel; also used inline in chat views.
 */
import { BookOpen, Check, Database, Globe, Loader2, Network, X } from "lucide-react";
import { useState } from "react";
import { useChatStore, type ToolActivity } from "../../stores/chatStore";
import { formatDuration, summarizeArgs } from "../../lib/format";

const TOOL_ICONS: Record<string, typeof BookOpen> = {
  route_to_agent: Network,
  kb_search: BookOpen,
  tickets_query: Database,
  web_search: Globe,
};

export function ToolActivityFeed() {
  const activities = useChatStore((s) => s.toolActivities);
  const [expanded, setExpanded] = useState(false);

  if (activities.length === 0) {
    return (
      <p className="px-1 text-xs text-text-dim">
        No tool activity yet — send a message and watch the agent work.
      </p>
    );
  }

  const shown = expanded ? activities : activities.slice(0, 6);

  return (
    <div className="space-y-1.5">
      {shown.map((t) => (
        <ToolActivityCard key={t.toolCallId} activity={t} />
      ))}
      {activities.length > 6 && (
        <button
          onClick={() => setExpanded((e) => !e)}
          className="w-full rounded py-1 text-center text-[11px] text-text-dim hover:text-text"
        >
          {expanded ? "Show fewer" : `Show all ${activities.length}`}
        </button>
      )}
    </div>
  );
}

export function ToolActivityCard({ activity }: { activity: ToolActivity }) {
  const Icon = TOOL_ICONS[activity.toolName] ?? Loader2;
  const argsText = summarizeArgs(activity.args);
  const running = activity.status === "running";

  return (
    <div
      data-testid="tool-card"
      className="flex items-start gap-2.5 rounded-lg border border-border bg-surface px-3 py-2"
    >
      <Icon
        size={14}
        className={`mt-0.5 shrink-0 ${
          activity.status === "error"
            ? "text-danger"
            : activity.status === "done"
              ? "text-ok"
              : "animate-pulse text-accent"
        }`}
      />
      <div className="min-w-0 flex-1">
        <div className="flex items-center justify-between gap-2">
          <p className="truncate font-mono text-xs text-text">{activity.toolName}</p>
          {running ? (
            <span className="flex shrink-0 items-center gap-1 text-[11px] text-accent">
              <Loader2 size={10} className="animate-spin" /> running
            </span>
          ) : activity.status === "error" ? (
            <span className="flex shrink-0 items-center gap-1 text-[11px] text-danger">
              <X size={10} /> error
            </span>
          ) : (
            <span className="flex shrink-0 items-center gap-1 text-[11px] text-ok">
              <Check size={10} /> {formatDuration(activity.durationMs ?? 0)}
            </span>
          )}
        </div>
        {argsText && <p className="mt-0.5 truncate text-[11px] text-text-dim">{argsText}</p>}
        {activity.summary && (
          <p className="mt-0.5 truncate text-[11px] text-text-dim/80">{activity.summary}</p>
        )}
      </div>
    </div>
  );
}
