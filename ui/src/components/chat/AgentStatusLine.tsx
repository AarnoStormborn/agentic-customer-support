/**
 * chat/AgentStatusLine.tsx — the "agent is working" indicator above the
 * streaming bubble, driven by the chatStore statusLine (tool activity).
 */
import { Bot } from "lucide-react";
import { useChatStore } from "../../stores/chatStore";
import { Spinner } from "../common/Spinner";

export function AgentStatusLine() {
  const statusLine = useChatStore((s) => s.statusLine);
  const connectionState = useChatStore((s) => s.connectionState);
  const messages = useChatStore((s) => s.messages);
  const last = messages[messages.length - 1];

  const label =
    statusLine ??
    (last?.text ? "Agent is working…" : "Agent is thinking…");

  return (
    <div className="flex items-center gap-2 px-1 text-xs text-text-dim" data-testid="agent-status">
      {connectionState === "reconnecting" ? (
        <span className="text-warn">Reconnecting to stream…</span>
      ) : (
        <>
          <Spinner size={12} className="text-accent" />
          <Bot size={12} />
          <span>{label}</span>
        </>
      )}
    </div>
  );
}
