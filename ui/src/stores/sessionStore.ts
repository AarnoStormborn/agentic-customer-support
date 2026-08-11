/**
 * stores/sessionStore.ts — sidebar session list (GET /api/sessions).
 *
 * The active selection lives in chatStore (single source of truth for what the
 * ChatArea shows); this store only manages the history list + CRUD.
 */
import { create } from "zustand";
import { api } from "../lib/api";
import type { SessionSummary } from "../lib/types";

interface SessionState {
  sessions: SessionSummary[];
  loading: boolean;
  error: string | null;

  fetchSessions: () => Promise<void>;
  deleteSession: (chatId: string) => Promise<void>;
}

export const useSessionStore = create<SessionState>()((set, get) => ({
  sessions: [],
  loading: false,
  error: null,

  fetchSessions: async () => {
    set({ loading: true, error: null });
    try {
      const res = await api.sessions();
      set({ sessions: res.sessions, loading: false });
    } catch (err) {
      set({
        loading: false,
        error: err instanceof Error ? err.message : "Failed to load sessions",
      });
    }
  },

  deleteSession: async (chatId) => {
    try {
      await api.deleteSession(chatId);
      set({ sessions: get().sessions.filter((s) => s.chatId !== chatId) });
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : "Failed to delete session",
      });
    }
  },
}));
