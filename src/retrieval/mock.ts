/**
 * Mock retrieval implementation — lets the agent runtime run with zero DB.
 *
 * A tiny keyword-overlap scorer over a hand-written "manuals" KB plus a couple of
 * ticket rows. Real searchHybrid (pgvector `<=>` + tsvector GIN + RRF, optional
 * Cohere rerank) replaces this at integration; see src/retrieval/index.ts.
 */

import type { HybridResult, HybridSearchOptions } from "./index.js";

interface MockDoc {
  docName: string;
  sectionPath: string;
  text: string;
  url: string;
}

const KB: MockDoc[] = [
  {
    docName: "LG OLED TV User Manual (65C4)",
    sectionPath: "Troubleshooting > Wi-Fi",
    text:
      "If the TV cannot connect to Wi-Fi, press and hold the Wi-Fi/Network reset option " +
      "in Settings > General > Network > Wi-Fi Connection for 5 seconds, then re-enter the " +
      "network password. Power-cycle the router for 30 seconds first. The TV supports 2.4 GHz " +
      "and 5 GHz bands; on 5 GHz networks, disable 'Auto' channel selection on the router if " +
      "the TV keeps dropping the signal.",
    url: "https://manuals.example.com/lg-oled-65c4#wifi",
  },
  {
    docName: "LG OLED TV User Manual (65C4)",
    sectionPath: "Troubleshooting > Factory Reset",
    text:
      "A full reset clears Wi-Fi credentials and all apps. Settings > General > System > " +
      "Reset to Initial Settings. This is a last resort — try the network reset under " +
      "Troubleshooting > Wi-Fi first.",
    url: "https://manuals.example.com/lg-oled-65c4#factory-reset",
  },
  {
    docName: "LG Soundbar S80QR Quick Guide",
    sectionPath: "Setup > HDMI eARC",
    text:
      "Connect the soundbar to the TV's HDMI eARC port (usually HDMI 2). In TV settings set " +
      "HDMI Deep Color to 4K and Digital Sound Output to 'Pass Through' for Dolby Atmos.",
    url: "https://manuals.example.com/lg-s80qr#earc",
  },
  {
    docName: "Samsung Side-by-Side Refrigerator Manual",
    sectionPath: "Troubleshooting > Ice Maker",
    text:
      "If the ice maker is not producing ice, check that the ice maker arm is in the ON " +
      "position, the freezer temperature is below 8 °F, and the water line has no kinks. " +
      "Replace the water filter every 6 months (part RF-4MB).",
    url: "https://manuals.example.com/samsung-fridge#ice-maker",
  },
  {
    docName: "Whirlpool Front-Load Washer Manual",
    sectionPath: "Error Codes > F5 E2",
    text:
      "F5 E2 means the door latch failed to lock. Check the door gasket for debris, clean the " +
      "striker plate, and ensure the door is fully closed. If the error persists after 3 " +
      "attempts, the door lock assembly needs replacement (part WPW10366014).",
    url: "https://manuals.example.com/whirlpool-washer#f5e2",
  },
];

const TICKET_ROWS: Record<string, unknown>[] = [
  {
    id: 10231,
    customer_name: "A. Rivera",
    product: "LG OLED TV 65C4",
    issue: "TV keeps disconnecting from home Wi-Fi",
    status: "open",
    priority: "high",
    created_at: "2026-07-28",
  },
  {
    id: 10217,
    customer_name: "J. Chen",
    product: "LG Soundbar S80QR",
    issue: "No sound over eARC",
    status: "in_progress",
    priority: "medium",
    created_at: "2026-07-25",
  },
  {
    id: 10198,
    customer_name: "M. Okafor",
    product: "Samsung Refrigerator",
    issue: "Ice maker stopped working after filter change",
    status: "open",
    priority: "medium",
    created_at: "2026-07-21",
  },
  {
    id: 10154,
    customer_name: "S. Patel",
    product: "Whirlpool Washer",
    issue: "F5 E2 door latch error",
    status: "resolved",
    priority: "low",
    created_at: "2026-07-14",
  },
];

const TOKEN_RE = /[a-z0-9]+/g;

/** Normalize hyphens so "wi-fi" tokenizes like "wifi". */
function tokenize(s: string): Set<string> {
  return new Set(s.toLowerCase().replace(/[-–—]/g, "").match(TOKEN_RE) ?? []);
}

/** Naive keyword-overlap score, tuned so "lg tv wifi reset" hits the Wi-Fi entry. */
function scoreDoc(queryTokens: Set<string>, doc: MockDoc): number {
  const textTokens = tokenize(`${doc.docName} ${doc.sectionPath} ${doc.text}`);
  let score = 0;
  for (const t of queryTokens) {
    if (textTokens.has(t)) score += 1;
  }
  // Heuristic: 'wifi'/'network'/'reset' appearing together in the Wi-Fi entry already
  // wins on overlap; boost exact section matches to keep ordering deterministic.
  if (tokenize(doc.sectionPath).has("wifi") && queryTokens.has("wifi")) score += 1.5;
  return score / (queryTokens.size + 2); // normalize, avoid div-by-zero
}

export async function searchHybrid(
  opts: HybridSearchOptions,
): Promise<{ results: HybridResult[]; queryTimeMs: number }> {
  const started = Date.now();
  const topK = opts.topK ?? 5;
  const sourceTypes = opts.sourceTypes ?? ["kb", "sql"];
  const tokens = tokenize(opts.query ?? "");

  const results: HybridResult[] = [];

  if (sourceTypes.includes("kb")) {
    const scored = KB.map((doc) => ({ doc, score: scoreDoc(tokens, doc) }))
      .filter((x) => x.score > 0)
      .sort((a, b) => b.score - a.score);
    for (const { doc, score } of scored.slice(0, topK)) {
      results.push({
        text: doc.text,
        score: Number(score.toFixed(3)),
        source: {
          type: "kb",
          docName: doc.docName,
          sectionPath: doc.sectionPath,
          url: doc.url,
        },
      });
    }
  }

  if (sourceTypes.includes("sql")) {
    // Only surface ticket rows when the query looks ticket-ish.
    const looksLikeTickets = tokens.size > 0 &&
      (["ticket", "status", "wifi", "tv", "issue", "complaint", "escalat"].some((k) => tokens.has(k)));
    if (looksLikeTickets) {
      const rowTokens = TICKET_ROWS.flatMap((r) =>
        [...tokenize(Object.values(r).join(" "))],
      );
      const rowSet = new Set(rowTokens);
      let overlap = 0;
      for (const t of tokens) if (rowSet.has(t)) overlap += 1;
      const score = overlap / (tokens.size + 2);
      if (score > 0) {
        for (const row of TICKET_ROWS.slice(0, topK)) {
          results.push({
            text: `Ticket #${String(row.id)} — ${String(row.product)}: ${String(row.issue)} [${String(row.status)}]`,
            score,
            source: { type: "sql", title: `ticket #${String(row.id)}`, row },
          });
        }
      }
    }
  }

  return { results, queryTimeMs: Date.now() - started };
}
