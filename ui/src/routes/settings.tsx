/**
 * routes/settings.tsx — theme / font size / model + retrieval toggles,
 * persisted to localStorage via settingsStore (ui.md §2.5).
 */
import { useEffect, useState } from "react";
import { Settings2 } from "lucide-react";
import { api, API_BASE } from "../lib/api";
import { useSettingsStore, type FontSize, type Theme } from "../stores/settingsStore";
import { Badge } from "../components/common/Badge";

function Toggle({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string;
  hint?: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-3 rounded-lg border border-border bg-surface px-3 py-2.5">
      <span>
        <span className="block text-sm text-text">{label}</span>
        {hint && <span className="block text-xs text-text-dim">{hint}</span>}
      </span>
      <button
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`relative h-5 w-9 shrink-0 rounded-full transition-colors ${
          checked ? "bg-accent" : "bg-surface-2 border border-border"
        }`}
      >
        <span
          className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-all ${
            checked ? "left-[18px]" : "left-0.5"
          }`}
        />
      </button>
    </label>
  );
}

export default function SettingsView() {
  const {
    theme,
    fontSize,
    defaultModel,
    retrieval,
    setTheme,
    setFontSize,
    setDefaultModel,
    setRetrieval,
  } = useSettingsStore();
  const [models, setModels] = useState<string[]>([]);

  useEffect(() => {
    api
      .models()
      .then((m) => setModels(m.models))
      .catch(() => {});
  }, []);

  const themeOptions: Theme[] = ["dark", "light"];
  const fontOptions: FontSize[] = ["sm", "md", "lg"];

  return (
    <div className="h-full overflow-y-auto p-4">
      <header className="mb-4 flex items-center gap-2">
        <Settings2 size={18} className="text-accent" />
        <h1 className="text-lg font-semibold text-text">Settings</h1>
        <Badge tone="neutral">stored locally</Badge>
      </header>

      <div className="max-w-2xl space-y-6">
        <section className="space-y-2">
          <h2 className="text-xs font-medium uppercase tracking-wide text-text-dim">Appearance</h2>
          <div className="flex gap-2">
            {themeOptions.map((t) => (
              <button
                key={t}
                onClick={() => setTheme(t)}
                className={`rounded-lg border px-4 py-2 text-sm capitalize transition-colors ${
                  theme === t
                    ? "border-accent bg-accent/15 text-text"
                    : "border-border bg-surface text-text-dim hover:text-text"
                }`}
              >
                {t}
              </button>
            ))}
          </div>
          <div className="flex gap-2">
            {fontOptions.map((f) => (
              <button
                key={f}
                onClick={() => setFontSize(f)}
                className={`rounded-lg border px-4 py-2 text-sm transition-colors ${
                  fontSize === f
                    ? "border-accent bg-accent/15 text-text"
                    : "border-border bg-surface text-text-dim hover:text-text"
                }`}
              >
                {f === "sm" ? "Small" : f === "lg" ? "Large" : "Medium"}
              </button>
            ))}
          </div>
        </section>

        <section className="space-y-2">
          <h2 className="text-xs font-medium uppercase tracking-wide text-text-dim">Default model</h2>
          <p className="text-xs text-text-dim">
            UI-level preference. The backend picks the runtime model itself (env
            <span className="font-mono"> PI_MODEL</span>) — the chat POST body doesn't carry a
            model field.
          </p>
          <select
            value={defaultModel ?? ""}
            onChange={(e) => setDefaultModel(e.target.value || null)}
            className="h-9 w-full rounded-lg border border-border bg-surface px-2 font-mono text-sm text-text focus:border-accent focus:outline-none"
          >
            <option value="">Backend default</option>
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </section>

        <section className="space-y-2">
          <h2 className="text-xs font-medium uppercase tracking-wide text-text-dim">
            Retrieval toggles
          </h2>
          <p className="text-xs text-text-dim">
            Preferences stored locally (v1). Wiring these into the agent's tool config is a
            future phase.
          </p>
          <Toggle
            label="SQL tickets"
            hint="Query the ticket store"
            checked={retrieval.sql}
            onChange={(v) => setRetrieval({ sql: v })}
          />
          <Toggle
            label="Vector manuals"
            hint="Search the knowledge base"
            checked={retrieval.vector}
            onChange={(v) => setRetrieval({ vector: v })}
          />
          <Toggle
            label="Web search"
            hint="Fall back to the web"
            checked={retrieval.web}
            onChange={(v) => setRetrieval({ web: v })}
          />
          <Toggle
            label="Reranker"
            hint="Re-rank hybrid results"
            checked={retrieval.reranker}
            onChange={(v) => setRetrieval({ reranker: v })}
          />
          <label className="flex items-center justify-between gap-3 rounded-lg border border-border bg-surface px-3 py-2.5">
            <span className="text-sm text-text">Top-K results</span>
            <input
              type="number"
              min={1}
              max={10}
              value={retrieval.topK}
              onChange={(e) => setRetrieval({ topK: Math.min(10, Math.max(1, Number(e.target.value) || 5)) })}
              className="h-8 w-20 rounded-md border border-border bg-canvas px-2 text-sm text-text focus:border-accent focus:outline-none"
            />
          </label>
        </section>

        <section className="space-y-2">
          <h2 className="text-xs font-medium uppercase tracking-wide text-text-dim">Connection</h2>
          <p className="rounded-lg border border-border bg-surface px-3 py-2.5 font-mono text-xs text-text-dim">
            API base: {API_BASE || "/ (Vite proxy → http://localhost:8000)"}
          </p>
        </section>
      </div>
    </div>
  );
}
