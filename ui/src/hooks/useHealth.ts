/**
 * hooks/useHealth.ts — poll GET /health so the TopBar connection pill reflects
 * backend availability (and per-dependency status) without a WS channel.
 */
import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type { HealthResponse } from "../lib/types";

export type HealthStatus = "checking" | "ok" | "degraded" | "offline";

export function useHealth(intervalMs = 30_000): {
  health: HealthResponse | null;
  status: HealthStatus;
  refresh: () => void;
} {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [status, setStatus] = useState<HealthStatus>("checking");

  const check = async () => {
    try {
      const h = await api.health();
      setHealth(h);
      setStatus(h.status === "ok" ? "ok" : "degraded");
    } catch {
      setHealth(null);
      setStatus("offline");
    }
  };

  useEffect(() => {
    void check();
    const t = setInterval(() => void check(), intervalMs);
    return () => clearInterval(t);
  }, [intervalMs]);

  return { health, status, refresh: () => void check() };
}
