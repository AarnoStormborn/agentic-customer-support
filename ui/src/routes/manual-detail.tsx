/**
 * routes/manual-detail.tsx — chunk list for one manual (GET /api/manuals/:id/chunks).
 * "Ask about this chunk" prefills the composer and routes back to chat.
 */
import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router";
import { ArrowLeft, Bot, ChevronDown, FileText } from "lucide-react";
import { api } from "../lib/api";
import type { Chunk } from "../lib/types";
import { useChatStore } from "../stores/chatStore";
import { Skeleton } from "../components/common/Spinner";
import { ErrorBanner } from "../components/common/ErrorBanner";
import { Button } from "../components/common/Button";

export default function ManualDetailView() {
  const { id } = useParams<{ id: string }>();
  const docId = Number(id);
  const [chunks, setChunks] = useState<Chunk[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [openIndex, setOpenIndex] = useState<number | null>(0);
  const navigate = useNavigate();

  useEffect(() => {
    let cancelled = false;
    setChunks(null);
    setError(null);
    api
      .manualChunks(docId, 200)
      .then((res) => {
        if (!cancelled) setChunks(res.chunks);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load chunks");
      });
    return () => {
      cancelled = true;
    };
  }, [docId]);

  const askAboutChunk = (chunk: Chunk) => {
    useChatStore.getState().prefillDraft(
      `About "${chunk.section ?? chunk.heading_path ?? `chunk ${chunk.chunk_index}`}" (${chunk.heading_path ?? "no heading"}): ${chunk.chunk_text.slice(0, 1200)}`,
    );
    navigate("/");
  };

  return (
    <div className="flex h-full flex-col gap-3 p-4">
      <header className="flex items-center gap-2">
        <Link
          to="/manuals"
          className="flex items-center gap-1 rounded-md px-2 py-1 text-xs text-text-dim hover:bg-surface-2 hover:text-text"
        >
          <ArrowLeft size={14} /> Manuals
        </Link>
        <h1 className="truncate text-lg font-semibold text-text">Manual #{docId}</h1>
        {chunks && <span className="text-xs text-text-dim">{chunks.length} chunks</span>}
      </header>

      {error && <ErrorBanner message={error} onRetry={undefined} />}

      {!chunks && !error && (
        <div className="space-y-2">
          <Skeleton className="h-16 w-full" />
          <Skeleton className="h-16 w-full" />
          <Skeleton className="h-16 w-2/3" />
        </div>
      )}

      <div className="min-h-0 flex-1 space-y-2 overflow-y-auto pb-4">
        {chunks?.map((c, i) => {
          const open = openIndex === i;
          return (
            <div key={c.chunk_id} className="rounded-xl border border-border bg-surface">
              <button
                onClick={() => setOpenIndex(open ? null : i)}
                className="flex w-full items-center gap-2 px-3 py-2.5 text-left"
              >
                <FileText size={14} className="shrink-0 text-accent" />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-[13px] font-medium text-text">
                    {c.heading_path || c.section || `Chunk ${c.chunk_index}`}
                  </p>
                  <p className="text-[11px] text-text-dim">
                    chunk #{c.chunk_index}
                    {c.page_start !== null && ` · pages ${c.page_start}–${c.page_end ?? c.page_start}`}
                  </p>
                </div>
                <ChevronDown
                  size={14}
                  className={`shrink-0 text-text-dim transition-transform ${open ? "rotate-180" : ""}`}
                />
              </button>
              {open && (
                <div className="border-t border-border px-3 py-3">
                  <p className="whitespace-pre-wrap text-[13px] leading-relaxed text-text">
                    {c.chunk_text}
                  </p>
                  <div className="mt-3 flex items-center justify-between gap-2">
                    <p className="font-mono text-[11px] text-text-dim">
                      {c.section ?? "section: —"} · id {c.chunk_id}
                    </p>
                    <Button size="sm" variant="primary" onClick={() => askAboutChunk(c)}>
                      <Bot size={13} /> Ask about this chunk
                    </Button>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
