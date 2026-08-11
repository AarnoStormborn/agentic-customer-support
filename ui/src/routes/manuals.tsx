/**
 * routes/manuals.tsx — knowledge-base browser: manual list (GET /api/manuals)
 * with client-side filtering, linking into the chunk detail route.
 */
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router";
import { BookOpen, ChevronRight, Search } from "lucide-react";
import { api } from "../lib/api";
import type { ManualSummary } from "../lib/types";
import { useDebounce } from "../hooks/useDebounce";
import { Skeleton } from "../components/common/Spinner";
import { ErrorBanner } from "../components/common/ErrorBanner";
import { formatCount } from "../lib/format";

export default function ManualsView() {
  const [manuals, setManuals] = useState<ManualSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const debouncedQ = useDebounce(q, 200);

  useEffect(() => {
    let cancelled = false;
    api
      .manuals()
      .then((res) => {
        if (!cancelled) setManuals(res.manuals);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load manuals");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const filtered = useMemo(() => {
    if (!manuals) return [];
    const needle = debouncedQ.trim().toLowerCase();
    if (!needle) return manuals;
    return manuals.filter(
      (m) =>
        m.doc_name.toLowerCase().includes(needle) ||
        (m.doc_type ?? "").toLowerCase().includes(needle) ||
        (m.file_path ?? "").toLowerCase().includes(needle),
    );
  }, [manuals, debouncedQ]);

  return (
    <div className="flex h-full flex-col gap-3 p-4">
      <header className="flex items-center gap-2">
        <BookOpen size={18} className="text-accent" />
        <h1 className="text-lg font-semibold text-text">Knowledge base</h1>
        <p className="text-xs text-text-dim">Ingested manuals, chunked for retrieval</p>
      </header>

      <div className="relative max-w-sm">
        <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-dim" />
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Filter manuals…"
          className="h-9 w-full rounded-lg border border-border bg-surface pl-9 pr-3 text-sm text-text placeholder:text-text-dim focus:border-accent focus:outline-none"
        />
      </div>

      {error && <ErrorBanner message={error} onRetry={() => setError(null)} />}

      {!manuals && !error && (
        <div className="grid gap-2 md:grid-cols-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-20 w-full" />
          ))}
        </div>
      )}

      {manuals && (
        <div className="min-h-0 flex-1 overflow-y-auto">
          {filtered.length === 0 ? (
            <p className="py-10 text-center text-sm text-text-dim">No manuals match.</p>
          ) : (
            <div className="grid gap-2 md:grid-cols-2">
              {filtered.map((m) => (
                <Link
                  key={m.doc_id}
                  to={`/manuals/${m.doc_id}`}
                  className="group flex items-start gap-3 rounded-xl border border-border bg-surface p-3 transition-colors hover:border-accent/60 hover:bg-surface-2"
                >
                  <div className="rounded-lg bg-accent/15 p-2 text-accent">
                    <BookOpen size={16} />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium text-text">{m.doc_name}</p>
                    <p className="mt-0.5 truncate font-mono text-[11px] text-text-dim">
                      {m.file_path ?? m.doc_type ?? "—"}
                    </p>
                    <p className="mt-1 text-[11px] text-text-dim">
                      {formatCount(m.chunk_count)} chunks
                    </p>
                  </div>
                  <ChevronRight
                    size={15}
                    className="mt-1 shrink-0 text-text-dim opacity-0 transition-opacity group-hover:opacity-100"
                  />
                </Link>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
