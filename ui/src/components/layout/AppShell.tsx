/**
 * layout/AppShell.tsx — three-region shell (ui.md §2.1): SessionSidebar +
 * TopBar + routed main area. The chat route additionally renders the
 * ContextPanel beside the outlet; other routes use the full width.
 */
import { useState } from "react";
import { Outlet, useLocation } from "react-router";
import { SessionSidebar } from "./SessionSidebar";
import { TopBar } from "./TopBar";
import { ContextPanel } from "./ContextPanel";

export function AppShell() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [contextOpen, setContextOpen] = useState(false);
  const { pathname } = useLocation();
  const isChat = pathname === "/";

  return (
    <div className="flex h-screen overflow-hidden bg-canvas text-text">
      <SessionSidebar
        collapsed={sidebarCollapsed}
        onClose={() => setSidebarCollapsed(true)}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar
          onToggleSidebar={() => setSidebarCollapsed((c) => !c)}
          sidebarCollapsed={sidebarCollapsed}
        />

        <div className="flex min-h-0 flex-1">
          <main className="min-w-0 flex-1">
            <Outlet context={{ setContextOpen }} />
          </main>
          {isChat && <ContextPanel open={contextOpen} onClose={() => setContextOpen(false)} />}
        </div>
      </div>
    </div>
  );
}
