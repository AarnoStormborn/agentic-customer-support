/**
 * chat/SourcesPanel.tsx — source cards for the `done.sources[]` payload.
 *
 * Shapes (from src/tools/*):
 *   kb:  { type:"kb",  title: docName, sectionPath?, url?, score? }
 *   sql: { type:"sql", title:"ticket #id", row: {ticket_id, ticket_subject, status, …} }
 *   web: { type:"web", title, url }
 */
import { BookOpen, Database, ExternalLink, FileText } from "lucide-react";
import type { SourceRef } from "../../lib/types";
import { Badge } from "../common/Badge";
import { useState } from "react";
import { TicketDrawer } from "../tickets/TicketDrawer";

export function SourceRefCard({
  source,
  index,
}: {
  source: SourceRef;
  index: number;
}) {
  const [drawerTicket, setDrawerTicket] = useState<number | null>(null);

  const kb = source.type === "kb";
  const sql = source.type === "sql";
  const web = source.type === "web";

  const title = source.title ?? source.docName ?? "Source";
  const ticketId = sql
    ? Number(source.row?.ticket_id ?? /#(\d+)/.exec(title)?.[1] ?? 0)
    : 0;
  const score = typeof source.score === "number" ? source.score : null;

  return (
    <>
      <div
        className="group flex items-start gap-2.5 rounded-lg border border-border bg-surface px-3 py-2 transition-colors hover:border-text-dim/50"
        data-testid="source-card"
      >
        <span className="mt-0.5 shrink-0 font-mono text-[11px] text-text-dim">{index}</span>
        {kb && <BookOpen size={15} className="mt-0.5 shrink-0 text-accent" />}
        {sql && <Database size={15} className="mt-0.5 shrink-0 text-ok" />}
        {web && <ExternalLink size={15} className="mt-0.5 shrink-0 text-warn" />}

        <div className="min-w-0 flex-1">
          <div className="flex items-center justify-between gap-2">
            <p className="truncate text-[13px] font-medium text-text">{title}</p>
            {score !== null && (
              <Badge tone="accent">{score.toFixed(2)}</Badge>
            )}
          </div>
          {kb && source.sectionPath && (
            <p className="mt-0.5 truncate text-xs text-text-dim">{source.sectionPath}</p>
          )}
          {sql && source.row && (
            <p className="mt-0.5 truncate text-xs text-text-dim">
              {String(source.row.ticket_subject ?? "—")}
            </p>
          )}
          {web && source.url && (
            <a
              href={source.url}
              target="_blank"
              rel="noreferrer noopener"
              className="mt-0.5 block truncate text-xs text-accent hover:underline"
            >
              {source.url}
            </a>
          )}
        </div>

        {sql && ticketId > 0 && (
          <button
            onClick={() => setDrawerTicket(ticketId)}
            className="shrink-0 rounded px-1.5 py-0.5 text-[11px] text-accent hover:bg-accent/10"
          >
            Open
          </button>
        )}
        {kb && (
          <FileText size={13} className="mt-1 shrink-0 text-text-dim/50" />
        )}
      </div>

      {drawerTicket !== null && (
        <TicketDrawer ticketId={drawerTicket} onClose={() => setDrawerTicket(null)} />
      )}
    </>
  );
}
