/**
 * stores/settingsStore.ts — UI preferences, persisted to localStorage via the
 * zustand `persist` middleware.
 *
 * NOTE: v1 these are client-side only. The backend chat POST accepts a
 * `metadata` object, so a future phase can pass retrieval toggles through.
 */
import { create } from "zustand";
import { persist } from "zustand/middleware";

export type Theme = "dark" | "light";
export type FontSize = "sm" | "md" | "lg";

export interface RetrievalSettings {
  sql: boolean;
  vector: boolean;
  web: boolean;
  reranker: boolean;
  topK: number;
}

interface SettingsState {
  theme: Theme;
  fontSize: FontSize;
  /** "provider/id", selected in TopBar/Settings (UI-level; backend uses PI_MODEL) */
  defaultModel: string | null;
  retrieval: RetrievalSettings;

  setTheme: (theme: Theme) => void;
  setFontSize: (size: FontSize) => void;
  setDefaultModel: (model: string | null) => void;
  setRetrieval: (patch: Partial<RetrievalSettings>) => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      theme: "dark",
      fontSize: "md",
      defaultModel: null,
      retrieval: { sql: true, vector: true, web: true, reranker: true, topK: 5 },

      setTheme: (theme) => set({ theme }),
      setFontSize: (fontSize) => set({ fontSize }),
      setDefaultModel: (defaultModel) => set({ defaultModel }),
      setRetrieval: (patch) =>
        set((s) => ({ retrieval: { ...s.retrieval, ...patch } })),
    }),
    {
      name: "acs-ui-settings",
      version: 1,
    },
  ),
);
