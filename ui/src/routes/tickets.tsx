/** routes/tickets.tsx — server-searched ticket table + drawer (ui.md §2.3). */
import { useState } from "react";
import { Ticket } from "lucide-react";
import { TicketTable } from "../components/tickets/TicketTable";
import { TicketDrawer } from "../components/tickets/TicketDrawer";

export default function TicketsView() {
  const [selected, setSelected] = useState<number | null>(null);

  return (
    <div className="flex h-full flex-col gap-3 p-4">
      <header className="flex items-center gap-2">
        <Ticket size={18} className="text-accent" />
        <h1 className="text-lg font-semibold text-text">Tickets</h1>
        <p className="text-xs text-text-dim">
          Server-side search over the ingested ticket store
        </p>
      </header>

      <div className="min-h-0 flex-1">
        <TicketTable onSelect={setSelected} />
      </div>

      {selected !== null && (
        <TicketDrawer ticketId={selected} onClose={() => setSelected(null)} />
      )}
    </div>
  );
}
