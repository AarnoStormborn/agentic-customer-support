/**
 * tickets/TicketTable.tsx — server-side searched/paginated table
 * (GET /api/tickets?q=&status=&page=&pageSize=). Debounced input, skeleton
 * rows, row click → TicketDrawer.
 */
import { useEffect, useState } from "react";
import { Search } from "lucide-react";
import { api } from "../../lib/api";
import type { TicketRow } from "../../lib/types";
import { useDebounce } from "../../hooks/useDebounce";
import { Badge } from "../common/Badge";
import { Skeleton } from "../common/Spinner";
import { ErrorBanner } from "../common/ErrorBanner";
import { Button } from "../common/Button";

const PAGE_SIZE = 20;
const STATUSES = ["open", "pending", "resolved", "closed", "escalated"];

export function TicketTable({ onSelect }: { onSelect: (id: number) => void }) {
  const [q, setQ] = useState("");
  const [status, setStatus] = useState("");
  const [page, setPage] = useState(1);
  const [rows, setRows] = useState<TicketRow[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const debouncedQ = useDebounce(q, 350);

  useEffect(() => {
    setPage(1);
  }, [debouncedQ, status]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .tickets({ q: debouncedQ, status: status || undefined, page, pageSize: PAGE_SIZE })
      .then((res) => {
        if (cancelled) return;
        setRows(res.rows);
        setTotal(res.total);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load tickets");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [debouncedQ, status, page]);

  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <div className="flex h-full flex-col gap-3">
      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-dim" />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search subject, product, narrative…"
            className="h-9 w-full rounded-lg border border-border bg-surface pl-9 pr-3 text-sm text-text placeholder:text-text-dim focus:border-accent focus:outline-none"
          />
        </div>
        <select
          value={status}
          onChange={(e) => setStatus(e.target.value)}
          className="h-9 rounded-lg border border-border bg-surface px-2 text-sm text-text focus:border-accent focus:outline-none"
        >
          <option value="">All statuses</option>
          {STATUSES.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </div>

      <div className="min-h-0 flex-1 overflow-auto rounded-lg border border-border">
        <table className="w-full border-collapse text-sm">
          <thead className="sticky top-0 bg-surface-2 text-left text-xs uppercase tracking-wide text-text-dim">
            <tr>
              <th className="px-3 py-2 font-medium">ID</th>
              <th className="px-3 py-2 font-medium">Subject</th>
              <th className="px-3 py-2 font-medium">Product</th>
              <th className="px-3 py-2 font-medium">Status</th>
              <th className="px-3 py-2 font-medium">Priority</th>
              <th className="px-3 py-2 font-medium">Source</th>
            </tr>
          </thead>
          <tbody>
            {loading &&
              Array.from({ length: 8 }).map((_, i) => (
                <tr key={i} className="border-t border-border">
                  <td colSpan={6} className="px-3 py-2">
                    <Skeleton className="h-4 w-full" />
                  </td>
                </tr>
              ))}
            {!loading &&
              rows.map((t) => (
                <tr
                  key={t.ticket_id}
                  onClick={() => onSelect(t.ticket_id)}
                  className="cursor-pointer border-t border-border transition-colors hover:bg-surface-2"
                >
                  <td className="px-3 py-2 font-mono text-xs text-accent">
                    #{t.ticket_id}
                  </td>
                  <td className="max-w-[320px] truncate px-3 py-2 text-text">
                    {t.ticket_subject || "—"}
                  </td>
                  <td className="max-w-[160px] truncate px-3 py-2 text-text-dim">
                    {t.product_purchased || "—"}
                  </td>
                  <td className="px-3 py-2">
                    <Badge tone={t.status === "resolved" || t.status === "closed" ? "ok" : "warn"}>
                      {t.status}
                    </Badge>
                  </td>
                  <td className="px-3 py-2 text-text-dim">{t.ticket_priority || "—"}</td>
                  <td className="px-3 py-2 text-text-dim">{t.source}</td>
                </tr>
              ))}
            {!loading && rows.length === 0 && (
              <tr>
                <td colSpan={6} className="px-3 py-10 text-center text-sm text-text-dim">
                  No tickets match your filters.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {error && <ErrorBanner message={error} onRetry={() => setPage((p) => p)} />}

      <div className="flex items-center justify-between text-xs text-text-dim">
        <span>
          {total.toLocaleString()} tickets · page {page}/{pages}
        </span>
        <div className="flex gap-1.5">
          <Button size="sm" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
            Previous
          </Button>
          <Button size="sm" disabled={page >= pages} onClick={() => setPage((p) => p + 1)}>
            Next
          </Button>
        </div>
      </div>
    </div>
  );
}
