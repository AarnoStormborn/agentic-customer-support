/**
 * LOCAL COPY of the retrieval contract — see docs/design/integration-contract.md.
 *
 * This file is owned by the agent-runtime track as a *local interface copy + mock
 * switch* so the kb_search tool works without the database. At integration, the
 * retrieval-core track's src/retrieval/index.ts (same signatures) replaces this
 * module — tools keep working unchanged.
 *
 * Switch flag (env):
 *   RETRIEVAL_MODE=mock (default) — use src/retrieval/mock.ts (no DB needed).
 *   RETRIEVAL_MODE=real            — dynamically load the implementation module at
 *                                    RETRIEVAL_IMPL (e.g. a path to the real
 *                                    retrieval module) and use its searchHybrid.
 *                                    Falls back to mock with a warning if the
 *                                    module cannot be loaded.
 */

import { searchHybrid as mockSearchHybrid } from "./mock.js";

export interface HybridResult {
  text: string;
  source: {
    type: "kb" | "sql";
    title?: string;
    docName?: string;
    sectionPath?: string;
    page?: number;
    url?: string | null;
    row?: Record<string, unknown>; // for sql results
  };
  score: number;
}

export interface HybridSearchOptions {
  query: string;
  topK?: number; // default 5
  sourceTypes?: ("kb" | "sql")[]; // default ["kb","sql"]
  filter?: Record<string, unknown>;
}

export interface IngestSummary {
  source: "suraj520" | "cfpb" | "comcast" | "manual";
  rowsInserted: number;
  errors: number;
}

const MODE = process.env.RETRIEVAL_MODE ?? "mock";

/** searchHybrid — the only retrieval function the runtime tools consume. */
export async function searchHybrid(
  opts: HybridSearchOptions,
): Promise<{ results: HybridResult[]; queryTimeMs: number }> {
  if (MODE === "real") {
    const implPath = process.env.RETRIEVAL_IMPL;
    if (implPath) {
      try {
        // Non-literal dynamic import: resolved at runtime, not type-checked here.
        const mod: unknown = await import(implPath);
        const real = mod as { searchHybrid?: typeof searchHybrid };
        if (typeof real.searchHybrid === "function") {
          return await real.searchHybrid(opts);
        }
        console.warn("[retrieval] RETRIEVAL_IMPL loaded but has no searchHybrid — using mock.");
      } catch (err) {
        console.warn(
          `[retrieval] RETRIEVAL_MODE=real but impl '${implPath}' failed to load (${(err as Error).message}) — using mock.`,
        );
      }
    } else {
      console.warn("[retrieval] RETRIEVAL_MODE=real requires RETRIEVAL_IMPL — using mock.");
    }
  }
  return mockSearchHybrid(opts);
}

/** embedTexts — used by the ingest pipeline (real module only); mock is a no-op stub. */
export async function embedTexts(_texts: string[]): Promise<number[][]> {
  return [];
}

/** runSchema / ingestTickets / ingestManuals — owned by retrieval-core; stubs here. */
export async function runSchema(): Promise<void> {
  throw new Error("runSchema is owned by the retrieval-core track (real module).");
}
export async function ingestTickets(_source: "suraj520" | "cfpb" | "comcast"): Promise<IngestSummary> {
  throw new Error("ingestTickets is owned by the retrieval-core track (real module).");
}
export async function ingestManuals(_dir: string): Promise<IngestSummary> {
  throw new Error("ingestManuals is owned by the retrieval-core track (real module).");
}
