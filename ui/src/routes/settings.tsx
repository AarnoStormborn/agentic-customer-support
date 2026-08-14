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
          <h2 className="text-xs font-medium uppercase tracking-wide text-text-dim">Retrieval strategy</h2>
          <p className="text-xs text-text-dim">
            Sent with every chat message (Phase 5c). <span className="font-mono">hybrid</span> is the
            default; <span className="font-mono">hyde</span> embeds a hypothetical answer,{" "}
            <span className="font-mono">vector</span>/<span className="font-mono">keyword</span> are
            single-retriever modes, <span className="font-mono">multiQuery</span> fuses several
            paraphrases. Compare modes: <span className="font-mono">npm run eval -- --strategy &lt;mode&gt;</span>
          </p>

          <label className="flex items-center justify-between gap-3 rounded-lg border border-border bg-surface px-3 py-2.5">
            <span>
              <span className="block text-sm text-text">Mode</span>
              <span className="block text-xs text-text-dim">Retrieval pipeline for the knowledge base</span>
            </span>
            <select
              value={retrieval.mode}
              onChange={(e) => setRetrieval({ mode: e.target.value as typeof retrieval.mode })}
              className="h-8 w-36 rounded-lg border border-border bg-surface-2 px-2 font-mono text-sm text-text focus:border-accent focus:outline-none"
            >
              {(["hybrid", "vector", "keyword", "hyde", "hyde-hybrid"] as const).map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </label>

          <label className="flex items-center justify-between gap-3 rounded-lg border border-border bg-surface px-3 py-2.5">
            <span className="text-sm text-text">Top-K results</span>
            <input
              type="number"
              min={1}
              max={10}
              value={retrieval.topK}
              onChange={(e) => setRetrieval({ topK: Number(e.target.value) })}
              className="h-8 w-20 rounded-lg border border-border bg-surface-2 px-2 text-right font-mono text-sm text-text focus:border-accent focus:outline-none"
            />
          </label>

          <label className="flex items-center justify-between gap-3 rounded-lg border border-border bg-surface px-3 py-2.5">
            <span>
              <span className="block text-sm text-text">RRF constant (k)</span>
              <span className="block text-xs text-text-dim">Higher = rank position dominates fusion</span>
            </span>
            <input
              type="range"
              min={10}
              max={120}
              step={5}
              value={retrieval.rrfK}
              onChange={(e) => setRetrieval({ rrfK: Number(e.target.value) })}
              className="w-32 accent-indigo-500"
            />
            <span className="w-10 text-right font-mono text-sm text-text">{retrieval.rrfK}</span>
          </label>

          <Toggle
            label="Query relaxation"
            hint="Auto-drop unmatched FTS terms (keeps results when one word is missing)"
            checked={retrieval.relax}
            onChange={(v) => setRetrieval({ relax: v })}
          />
          <Toggle
            label="Multi-query"
            hint={`LLM paraphrases the query into ${retrieval.numVariants} variants, retrieves each, fuses`}
            checked={retrieval.multiQuery}
            onChange={(v) => setRetrieval({ multiQuery: v })}
          />
          <Toggle
            label="Query expansion"
            hint="Rule-based synonyms appended (no LLM cost)"
            checked={retrieval.queryExpansion}
            onChange={(v) => setRetrieval({ queryExpansion: v })}
          />
          <Toggle
            label="Rerank"
            hint="Cross-encoder rerank of candidates (needs COHERE_API_KEY; skipped otherwise)"
            checked={retrieval.rerank}
            onChange={(v) => setRetrieval({ rerank: v })}
          />
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
