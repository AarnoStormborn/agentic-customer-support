/**
 * chat/MessageList.tsx — scrollable message list with pin-to-bottom auto-scroll
 * (pause when the user scrolls up, ui.md §2.2). Empty state = welcome hero.
 */
import { useEffect, useRef, useState, type ReactNode } from "react";
import { Sparkles } from "lucide-react";
import { useChatStore } from "../../stores/chatStore";
import { MessageBubble } from "./MessageBubble";
import { AgentStatusLine } from "./AgentStatusLine";

const PIN_THRESHOLD = 120;

export function MessageList({ onSuggestedPrompt }: { onSuggestedPrompt: (text: string) => void }) {
  const messages = useChatStore((s) => s.messages);
  const isStreaming = useChatStore((s) => s.isStreaming);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [pinned, setPinned] = useState(true);

  // Keep pinned to bottom on new messages / stream ticks.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (pinned) el.scrollTop = el.scrollHeight;
  });

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const dist = el.scrollHeight - el.scrollTop - el.clientHeight;
    setPinned(dist < PIN_THRESHOLD);
  };

  if (messages.length === 0) {
    return <WelcomeHero onSuggestedPrompt={onSuggestedPrompt} />;
  }

  return (
    <div ref={scrollRef} onScroll={onScroll} className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
      <div className="mx-auto flex max-w-3xl flex-col gap-4">
        {messages.map((m) => (
          <MessageBubble key={m.id} message={m} />
        ))}
        {isStreaming && <AgentStatusLine />}
      </div>
    </div>
  );
}

function WelcomeHero({ onSuggestedPrompt }: { onSuggestedPrompt: (text: string) => void }) {
  const prompts: ReactNode[] = [
    "My LG TV won't connect to Wi-Fi — how do I reset the network settings?",
    "Find tickets about a broken soundbar and what the common fix is.",
    "How do I pair my new remote with the OLED?",
  ];

  return (
    <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-6 px-6">
      <div className="flex flex-col items-center gap-3 text-center">
        <div className="rounded-2xl bg-accent/15 p-4 text-accent">
          <Sparkles size={28} />
        </div>
        <h1 className="text-2xl font-semibold text-text">Agentic Customer Support</h1>
        <p className="max-w-md text-sm leading-relaxed text-text-dim">
          Ask about manuals, tickets, or the web. The agent searches the knowledge base,
          runs SQL against the ticket store, and cites its sources — watch it work in the
          context panel.
        </p>
      </div>
      <div className="grid w-full max-w-2xl gap-2">
        {prompts.map((p) => (
          <button
            key={String(p)}
            onClick={() => onSuggestedPrompt(String(p))}
            className="rounded-xl border border-border bg-surface px-4 py-3 text-left text-sm text-text transition-colors hover:border-accent/60 hover:bg-surface-2"
          >
            {p}
          </button>
        ))}
      </div>
    </div>
  );
}
