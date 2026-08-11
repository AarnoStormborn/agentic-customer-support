/**
 * layout/TopBar.tsx — model pill (GET /api/models), backend connection pill
 * (GET /health poll), nav links, settings gear. ui.md §2.1.
 */
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router";
import { BookOpen, ChevronDown, PanelLeftClose, PanelLeftOpen, Settings, Ticket } from "lucide-react";
import { api } from "../../lib/api";
import { useHealth } from "../../hooks/useHealth";
import { useChatStore } from "../../stores/chatStore";
import { useSettingsStore } from "../../stores/settingsStore";
import { Badge } from "../common/Badge";

export function TopBar({
  onToggleSidebar,
  sidebarCollapsed,
}: {
  onToggleSidebar: () => void;
  sidebarCollapsed: boolean;
}) {
  const { status, health } = useHealth(30_000);
  const connectionState = useChatStore((s) => s.connectionState);
  const { defaultModel, setDefaultModel } = useSettingsStore();
  const [models, setModels] = useState<string[]>([]);
  const [modelMenuOpen, setModelMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api
      .models()
      .then((m) => {
        setModels(m.models);
        if (!defaultModel && m.default) setDefaultModel(m.default);
      })
      .catch(() => {});
  }, [defaultModel, setDefaultModel]);

  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setModelMenuOpen(false);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  const pill = (() => {
    if (status === "offline") return { tone: "danger" as const, label: "Backend offline" };
    if (status === "degraded") return { tone: "warn" as const, label: "Degraded" };
    if (connectionState === "reconnecting")
      return { tone: "warn" as const, label: "Reconnecting…" };
    if (connectionState === "open" || connectionState === "connecting")
      return { tone: "ok" as const, label: "Live" };
    return { tone: "ok" as const, label: "Connected" };
  })();

  const model = defaultModel ?? models[0] ?? "model";

  return (
    <header className="flex h-12 shrink-0 items-center gap-2 border-b border-border bg-surface px-3">
      <button
        onClick={onToggleSidebar}
        aria-label={sidebarCollapsed ? "Show sidebar" : "Hide sidebar"}
        className="rounded p-1.5 text-text-dim hover:bg-surface-2 hover:text-text"
      >
        {sidebarCollapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
      </button>

      <nav className="flex items-center gap-1 text-xs">
        <Link
          to="/"
          className="rounded px-2 py-1.5 text-text-dim hover:bg-surface-2 hover:text-text"
        >
          Chat
        </Link>
        <Link
          to="/tickets"
          className="flex items-center gap-1 rounded px-2 py-1.5 text-text-dim hover:bg-surface-2 hover:text-text"
        >
          <Ticket size={12} /> Tickets
        </Link>
        <Link
          to="/manuals"
          className="flex items-center gap-1 rounded px-2 py-1.5 text-text-dim hover:bg-surface-2 hover:text-text"
        >
          <BookOpen size={12} /> Manuals
        </Link>
      </nav>

      <div className="ml-auto flex items-center gap-2">
        {/* Model pill */}
        <div ref={menuRef} className="relative">
          <button
            onClick={() => setModelMenuOpen((o) => !o)}
            className="flex items-center gap-1.5 rounded-lg border border-border bg-surface-2 px-2.5 py-1.5 font-mono text-[11px] text-text hover:border-text-dim/60"
          >
            {model}
            <ChevronDown size={12} className="text-text-dim" />
          </button>
          {modelMenuOpen && (
            <div className="absolute right-0 top-full z-30 mt-1 max-h-64 w-64 overflow-y-auto rounded-lg border border-border bg-surface py-1 shadow-panel">
              <p className="px-3 py-1.5 text-[11px] uppercase tracking-wide text-text-dim">
                Available models
              </p>
              {models.length === 0 && (
                <p className="px-3 py-1.5 text-xs text-text-dim">
                  No models reported by the backend.
                </p>
              )}
              {models.map((m) => (
                <button
                  key={m}
                  onClick={() => {
                    setDefaultModel(m);
                    setModelMenuOpen(false);
                  }}
                  className={`block w-full px-3 py-1.5 text-left font-mono text-[11px] hover:bg-surface-2 ${
                    m === model ? "text-accent" : "text-text"
                  }`}
                >
                  {m} {m === model && "✓"}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Connection pill */}
        <Badge tone={pill.tone}>
          <span
            className={`h-1.5 w-1.5 rounded-full ${
              pill.tone === "ok"
                ? "bg-ok"
                : pill.tone === "warn"
                  ? "bg-warn"
                  : "bg-danger animate-pulse"
            }`}
          />
          {pill.label}
          {health && health.status === "degraded" && (
            <span className="opacity-70">
              (pg {health.deps.postgres} · redis {health.deps.redis})
            </span>
          )}
        </Badge>

        <Link
          to="/settings"
          aria-label="Settings"
          className="rounded p-1.5 text-text-dim hover:bg-surface-2 hover:text-text"
        >
          <Settings size={16} />
        </Link>
      </div>
    </header>
  );
}
