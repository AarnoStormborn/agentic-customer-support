/**
 * routes/chat.tsx — the core screen (ui.md §2.2): ChatHeader + MessageList +
 * Composer, with the ContextPanel living in AppShell for this route.
 */
import { useOutletContext } from "react-router";
import { BookOpenCheck, FlaskConical, Square } from "lucide-react";
import { useState } from "react";
import { useChatStream } from "../hooks/useChatStream";
import { useChatStore, sessionPreview } from "../stores/chatStore";
import { MessageList } from "../components/chat/MessageList";
import { Composer } from "../components/chat/Composer";
import { RetrievalCompare } from "../components/chat/RetrievalCompare";
import { Badge } from "../components/common/Badge";

export default function ChatView() {
  useChatStream();
  const messages = useChatStore((s) => s.messages);
  const isStreaming = useChatStore((s) => s.isStreaming);
  const connectionState = useChatStore((s) => s.connectionState);
  const stopStreaming = useChatStore((s) => s.stopStreaming);
  const send = useChatStore((s) => s.send);
  const [compareOpen, setCompareOpen] = useState(false);

  const { setContextOpen } = useOutletContext<{ setContextOpen: (v: boolean) => void }>();

  const title = messages.length > 0 ? sessionPreview(messages) : "New session";
  // Last user question is what the compare panel runs (best for the retrieval demo).
  const lastUserQuestion = [...messages].reverse().find((m) => m.role === "user")?.text ?? "";

  return (
    <div className="flex h-full flex-col bg-canvas">
      <header className="flex h-11 shrink-0 items-center gap-2 border-b border-border bg-surface px-4">
        <h2 className="min-w-0 flex-1 truncate text-sm font-medium text-text">{title}</h2>
        {isStreaming && <Badge tone="ok">streaming</Badge>}
        {connectionState === "reconnecting" && <Badge tone="warn">reconnecting</Badge>}
        <button
          onClick={() => setContextOpen(true)}
          className="flex items-center gap-1 rounded-md px-2 py-1 text-xs text-text-dim hover:bg-surface-2 hover:text-text xl:hidden"
        >
          <BookOpenCheck size={14} /> Sources
        </button>
        {messages.length > 0 && !isStreaming && (
          <button
            onClick={() => setCompareOpen((v) => !v)}
            title="Compare retrieval modes on the last question"
            className="flex items-center gap-1 rounded-md border border-accent/30 px-2 py-1 text-xs text-accent hover:bg-accent/10"
          >
            <FlaskConical size={13} /> {compareOpen ? "Close compare" : "Compare modes"}
          </button>
        )}
        {isStreaming && (
          <button
            onClick={() => void stopStreaming()}
            className="flex items-center gap-1 rounded-md border border-danger/30 px-2 py-1 text-xs text-danger hover:bg-danger/10"
          >
            <Square size={12} /> Cancel
          </button>
        )}
      </header>

      <MessageList
        onSuggestedPrompt={(text) => {
          void send(text);
        }}
      />
      {compareOpen && lastUserQuestion && (
        <div className="shrink-0 border-t border-border p-3">
          <RetrievalCompare query={lastUserQuestion} onClose={() => setCompareOpen(false)} />
        </div>
      )}
      <Composer />
    </div>
  );
}
