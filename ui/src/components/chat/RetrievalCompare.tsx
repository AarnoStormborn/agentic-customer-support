/**
 * chat/RetrievalCompare.tsx — "Compare retrieval modes" panel.
 *
 * Runs the current question through every retrieval strategy (hybrid, vector,
 * keyword, hyde, hyde-hybrid) via POST /api/retrieval/compare and shows the
 * top results side by side — the eval comparison, live in the UI. Lets the user
 * see which mode surfaces the best chunks for their phrasing.
 */
import { useState } from "react";
import { FlaskConical, Loader2, X } from "lucide-react";
import { api } from "../../lib/api";
import type { CompareResponse } from "../../lib/types";

const MODE_LABEL: Record<string, string> = {
  hybrid: "hybrid (FTS+vector)",
  vector: "vector only",
  keyword: "keyword (FTS)",
  hyde: "hyde (hypothesis)",
  "hyde-hybrid": "hyde+hybrid",
};

export function RetrievalCompare({
  query,
  onClose,
}: {
  query: string;
  onClose: () => void;
}) {
  const [data, setData] = useState<CompareResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.compareRetrieval({ query, topK: 3 });
      setData(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "compare failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="rounded-xl border border-accent/30 bg-surface p-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="flex items-center gap-1.5 text-xs font-medium text-accent">
          <FlaskConical size={13} /> Retrieval mode comparison
        </span>
        <button onClick={onClose} className="text-text-dim hover:text-text" aria-label="Close">
          <X size={14} />
        </button>
      </div>
      <p className="mb-2 truncate text-xs text-text-dim" title={query}>
        query: “{query}”
      </p>

      {!data && !loading && (
        <button
          onClick={run}
          className="rounded-lg border border-accent bg-accent/10 px-3 py-1.5 text-xs text-accent hover:bg-accent/20"
        >
          Run all modes (may use an LLM for hyde)
        </button>
      )}
      {loading && (
        <div className="flex items-center gap-2 text-xs text-text-dim">
          <Loader2 size={13} className="animate-spin" /> comparing across modes…
        </div>
      )}
      {error && <p className="text-xs text-danger">{error}</p>}

      {data && (
        <div className="space-y-2">
          {data.modes.map((m) => (
            <div key={m.mode} className="rounded-lg bg-surface-2 p-2">
              <div className="mb-1 flex items-center gap-2">
                <span className="font-mono text-[11px] text-text">{MODE_LABEL[m.mode] ?? m.mode}</span>
                {m.relaxed && <span className="rounded bg-warn/20 px-1 text-[10px] text-warn">relaxed</span>}
                <span className="ml-auto text-[10px] text-text-dim">{m.queryTimeMs}ms</span>
              </div>
              {m.top.length === 0 ? (
                <p className="text-[11px] text-text-dim">no results</p>
              ) : (
                <ul className="space-y-0.5">
                  {m.top.map((r, i) => (
                    <li key={i} className="truncate text-[11px] text-text-dim">
                      <span className="font-mono text-text">{r.score.toFixed(3)}</span>{" "}
                      {r.docName ?? "?"}
                      {r.sectionPath ? ` › ${r.sectionPath}` : ""}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          ))}
          <p className="text-[10px] text-text-dim">
            hyde modes call an LLM to generate a hypothetical answer first — results vary run to run.
          </p>
        </div>
      )}
    </div>
  );
}
