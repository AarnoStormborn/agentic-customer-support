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

export type RetrievalMode = "hybrid" | "vector" | "keyword" | "hyde" | "hyde-hybrid";

export interface RetrievalSettings {
  mode: RetrievalMode;
  topK: number;
  rrfK: number;
  relax: boolean;
  multiQuery: boolean;
  numVariants: number;
  queryExpansion: boolean;
  rerank: boolean;
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
      retrieval: { mode: "hybrid", topK: 5, rrfK: 60, relax: true, multiQuery: false, numVariants: 3, queryExpansion: false, rerank: false },

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
