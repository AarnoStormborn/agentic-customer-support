/**
 * tickets/TicketDrawer.tsx — right-side detail panel for one ticket.
 * Fetches GET /api/tickets/:id; "Ask the agent about this ticket" prefills the
 * composer and routes back to the chat (ui.md §2.3).
 */
import { useEffect, useState } from "react";
import { useNavigate } from "react-router";
import { Bot, X } from "lucide-react";
import { api } from "../../lib/api";
import type { TicketRow } from "../../lib/types";
import { useChatStore } from "../../stores/chatStore";
import { Badge } from "../common/Badge";
import { Skeleton } from "../common/Spinner";
import { ErrorBanner } from "../common/ErrorBanner";
import { Button } from "../common/Button";

export function TicketDrawer({
  ticketId,
  onClose,
}: {
  ticketId: number;
  onClose: () => void;
}) {
  const [ticket, setTicket] = useState<TicketRow | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .ticket(ticketId)
      .then((t) => {
        if (!cancelled) setTicket(t);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load ticket");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [ticketId]);

  const askAgent = () => {
    const context = ticket
      ? `Ticket #${ticket.ticket_id}: ${ticket.ticket_subject ?? "no subject"}\n` +
        `Product: ${ticket.product_purchased ?? "?"}\n` +
        `Status: ${ticket.status}\n` +
        `Customer: ${ticket.customer_name ?? "?"}\n\n` +
        `Complaint: ${ticket.complaint_narrative ?? "—"}`
      : `Ticket #${ticketId}`;
    useChatStore.getState().prefillDraft(
      `Help me with ${context.slice(0, 1800)}`,
    );
    navigate("/");
  };

  return (
    <div className="fixed inset-0 z-40 flex justify-end">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />
      <aside className="relative flex h-full w-full max-w-md flex-col border-l border-border bg-surface shadow-panel">
        <header className="flex items-center justify-between border-b border-border px-4 py-3">
          <h3 className="text-sm font-semibold text-text">
            {ticket ? `Ticket #${ticket.ticket_id}` : `Ticket #${ticketId}`}
          </h3>
          <button
            onClick={onClose}
            aria-label="Close"
            className="rounded p-1 text-text-dim hover:bg-surface-2 hover:text-text"
          >
            <X size={16} />
          </button>
        </header>

        <div className="flex-1 overflow-y-auto p-4">
          {loading && (
            <div className="space-y-3">
              <Skeleton className="h-4 w-2/3" />
              <Skeleton className="h-4 w-1/2" />
              <Skeleton className="h-24 w-full" />
            </div>
          )}
          {error && <ErrorBanner message={error} onRetry={undefined} />}

          {ticket && (
            <div className="space-y-4">
              <div className="flex flex-wrap gap-1.5">
                <Badge tone={ticket.status === "resolved" || ticket.status === "closed" ? "ok" : "warn"}>
                  {ticket.status}
                </Badge>
                {ticket.ticket_priority && <Badge tone="accent">{ticket.ticket_priority}</Badge>}
                {ticket.ticket_type && <Badge>{ticket.ticket_type}</Badge>}
                <Badge>{ticket.source}</Badge>
              </div>

              <div className="grid grid-cols-2 gap-3 text-xs">
                <Field label="Customer" value={ticket.customer_name} />
                <Field label="Product" value={ticket.product_purchased} />
                <Field label="Channel" value={ticket.ticket_channel} />
                <Field label="Company" value={ticket.company} />
                <Field label="Created" value={ticket.created_at?.slice(0, 10)} />
                <Field label="Synthetic" value={ticket.is_synthetic ? "yes" : "no"} />
              </div>

              <div>
                <p className="mb-1 text-xs font-medium text-text-dim">Subject</p>
                <p className="text-sm text-text">{ticket.ticket_subject}</p>
              </div>

              <div>
                <p className="mb-1 text-xs font-medium text-text-dim">Complaint narrative</p>
                <p className="whitespace-pre-wrap text-[13px] leading-relaxed text-text">
                  {ticket.complaint_narrative || "—"}
                </p>
              </div>
            </div>
          )}
        </div>

        <footer className="border-t border-border p-3">
          <Button variant="primary" className="w-full" onClick={askAgent} disabled={!ticket}>
            <Bot size={15} /> Ask the agent about this ticket
          </Button>
        </footer>
      </aside>
    </div>
  );
}

function Field({ label, value }: { label: string; value: string | null | undefined }) {
  return (
    <div>
      <p className="text-text-dim">{label}</p>
      <p className="truncate font-medium text-text">{value || "—"}</p>
    </div>
  );
}
