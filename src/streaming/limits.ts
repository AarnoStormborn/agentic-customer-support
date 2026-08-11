/**
 * src/streaming/limits.ts — per-IP connection caps for SSE and WebSocket.
 *
 * §2.5 of the design doc: SSE/WS connections are cheap to open but long-lived, so we
 * cap concurrent connections per IP (default 5, see env.MAX_CONNECTIONS_PER_IP).
 * @fastify/rate-limit only governs request frequency, not long-lived sockets, so this
 * counter is the guard for both SSE and WS.
 */
import { env } from "../config/env.js";

export class IpConnectionCounter {
  private readonly counts = new Map<string, number>();
  private readonly limit: number;

  constructor(limit: number = env.MAX_CONNECTIONS_PER_IP) {
    this.limit = limit;
  }

  /** Try to acquire a slot. Returns true if the connection is allowed. */
  tryAcquire(ip: string): boolean {
    const current = this.counts.get(ip) ?? 0;
    if (current >= this.limit) return false;
    this.counts.set(ip, current + 1);
    return true;
  }

  release(ip: string): void {
    const current = this.counts.get(ip) ?? 0;
    if (current <= 1) this.counts.delete(ip);
    else this.counts.set(ip, current - 1);
  }

  get active(): number {
    let n = 0;
    for (const count of this.counts.values()) n += count;
    return n;
  }
}
