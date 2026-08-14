/**
 * src/retrieval/expand.ts — rule-based query expansion (no LLM).
 *
 * Adds synonyms / related terms so FTS and vector search catch more matches.
 * Deterministic + free (vs multiQuery which is LLM-based and costs tokens).
 */
const SYNONYMS: Record<string, string[]> = {
  tv: ["television", "display"],
  television: ["tv", "display"],
  wifi: ["wi-fi", "wireless", "network", "internet"],
  "wi-fi": ["wifi", "wireless", "network"],
  wireless: ["wifi", "wi-fi", "network"],
  network: ["wifi", "wi-fi", "connection"],
  refund: ["reimbursement", "money back", "return"],
  reimbursement: ["refund"],
  reset: ["restart", "reboot", "factory reset"],
  restart: ["reset", "reboot"],
  reboot: ["restart", "reset"],
  disconnect: ["drop", "lose connection", "offline"],
  sound: ["audio", "no sound"],
  audio: ["sound"],
  remote: ["remote control", "controller"],
  sim: ["sim card", "nano sim", "esim"],
  "sim card": ["sim", "nano sim", "esim"],
  screen: ["display", "panel", "picture"],
  picture: ["screen", "display", "image"],
  blinking: ["flashing"],
  power: ["power on", "boot", "startup"],
  charging: ["battery", "charge"],
  battery: ["charging", "power"],
};

const re = /[a-z0-9]+(?:[-'][a-z0-9]+)*/gi;

/** Expand a query by appending synonyms for its terms (deduped, capped). */
export function expandQuery(query: string, maxExtra = 12): string {
  const words = query.toLowerCase().match(re) ?? [];
  const extra: string[] = [];
  for (const w of words) {
    for (const syn of SYNONYMS[w] ?? []) {
      if (!extra.includes(syn)) extra.push(syn);
      if (extra.length >= maxExtra) break;
    }
    if (extra.length >= maxExtra) break;
  }
  return extra.length > 0 ? `${query} ${extra.join(" ")}` : query;
}
