/**
 * src/retrieval/ingest.ts — ingest pipeline + CLI (contract: `npm run ingest`).
 *
 * Tickets (suraj520): parquet → CSV (scripts/convert-suraj520.py) → RFC4180 parse
 *   → deterministic enrichment (data-management §2.4) → batched upsert by
 *   (source, source_ticket_id) via unnest + ON CONFLICT. Idempotent: re-running
 *   converges; `--dry-run` does no DB writes / no embedding calls.
 *
 * Manuals: PDF → parsePdf → chunkDocument → embedTexts → upsert documents by
 *   file_path + chunks by (doc_id, chunk_index). Idempotent.
 *
 * All values parameterized (v1 lesson #2); batching mirrors v1's async batch
 * embeddings (lesson: what worked).
 */
import { createHash } from "node:crypto";
import { readdir, readFile, stat } from "node:fs/promises";
import { join, basename, extname } from "node:path";
import { spawnSync } from "node:child_process";
import "dotenv/config";
import { getPool } from "../db/pool.js";
import { embedTexts, embeddingsEnabled, EMBEDDING_MODEL, embeddingDim } from "./embed.js";
import { parseCsv } from "./csv.js";
import { parsePdf, chunkDocument, type DocumentChunk } from "./chunk.js";

// ---------------------------------------------------------------------------
// Types & config
// ---------------------------------------------------------------------------

export type TicketSource = "suraj520" | "cfpb" | "comcast";

export interface IngestSummary {
  source: string;
  rowsRead: number;
  rowsInserted: number;
  rowsUpdated: number;
  failures: string[];
  embeddingMode: "openai" | "hash";
  docs?: number;
  chunks?: number;
  dryRun: boolean;
}

export interface IngestOptions {
  dryRun?: boolean;
}

const SRC_DIR = "config/data";
const TICKETS_CSV = `${SRC_DIR}/tickets/suraj520.csv`;
const TICKETS_PARQUET = `${SRC_DIR}/raw/suraj520/tickets.parquet`;
const BATCH = 500;

const log = (...args: unknown[]) => console.log(`[ingest ${new Date().toISOString().slice(11, 19)}]`, ...args);
const warn = (...args: unknown[]) => console.warn(`[ingest]`, ...args);

// ---------------------------------------------------------------------------
// suraj520 mapping (data-management §2.4 — deterministic synthesis)
// ---------------------------------------------------------------------------

const md5 = (s: string) => createHash("md5").update(s).digest("hex");

/** int from md5 — deterministic across runs/machines */
const md5Int = (s: string) => parseInt(md5(s).slice(0, 8), 16);

const WORD_FIXES: Record<string, string> = {
  adobe: "Adobe", amazon: "Amazon", apple: "Apple", asus: "ASUS", autodesk: "Autodesk",
  bose: "Bose", canon: "Canon", dell: "Dell", dyson: "Dyson", fitbit: "Fitbit",
  garmin: "Garmin", google: "Google", gopro: "GoPro", hp: "HP", iphone: "iPhone",
  lenovo: "Lenovo", lg: "LG", macbook: "MacBook", microsoft: "Microsoft", nest: "Nest",
  nikon: "Nikon", nintendo: "Nintendo", philips: "Philips", playstation: "PlayStation",
  roomba: "Roomba", samsung: "Samsung", sony: "Sony", xbox: "Xbox", tv: "TV", oled: "OLED",
  xps: "XPS", rog: "ROG", pc: "PC", eos: "EOS", hdr: "HDR", d: "D", "4k": "4K",
};

/** title-case canonicalization: 'lg smart tv' → 'LG Smart TV' */
export function canonicalizeProduct(raw: string): string {
  const words = raw.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const out = words.map((w) => {
    const fix = WORD_FIXES[w];
    if (fix) return fix;
    if (/^\d+$/.test(w)) return w;
    return w.charAt(0).toUpperCase() + w.slice(1);
  });
  return out.join(" ");
}

/** 'technical issue' → 'Technical issue' (first word capitalized only, per data-management §2.4) */
const capitalizeFirst = (s: string) => s.trim().toLowerCase().replace(/^./, (c) => c.toUpperCase());

/** name from email local-part: 'john.smith42@example.com' → 'John Smith' */
function nameFromEmail(email: string, index: number): string {
  const local = email.split("@")[0] ?? "";
  const parts = local.split(/[._\-+]/).map((p) => p.replace(/\d+/g, "")).filter((p) => /[a-z]/i.test(p));
  if (parts.length === 0) return `customer_${index + 1}`;
  return parts.map((p) => p.charAt(0).toUpperCase() + p.slice(1)).join(" ");
}

/** deterministic channel: h%10 → Email 40%, Social Media/Phone/Chat 20% each */
function channelFromHash(h: number): string {
  const m = h % 10;
  if (m < 4) return "Email";
  if (m < 6) return "Social Media";
  if (m < 8) return "Phone";
  return "Chat";
}

const GENDERS = ["Male", "Female", "Other"] as const;

export interface MappedTicket {
  source: "suraj520";
  source_ticket_id: string;
  customer_name: string;
  customer_email: string;
  customer_age: number;
  customer_gender: string;
  product_purchased: string;
  date_of_purchase: string; // YYYY-MM-DD
  ticket_type: string;
  ticket_priority: string;
  ticket_channel: string;
  ticket_subject: string | null;
  complaint_narrative: string | null;
  is_synthetic: boolean;
}

/** Map + deterministically enrich one suraj520 CSV row (data-management §2.4). */
export function mapSuraj520Row(cells: string[], index: number): MappedTicket {
  const [email, product, type, subject, narrative, priority] = cells as (string | undefined)[];
  if (!email || !product) throw new Error(`row ${index + 2}: missing email/product`);
  const emailTrim = email.trim();
  const productCanon = canonicalizeProduct(product);
  const hEmail = md5Int(emailTrim);
  const hChannel = md5Int(`${emailTrim}||${type ?? ""}`);
  const hDate = md5Int(`${emailTrim}||${productCanon}`);
  const days = hDate % 730;
  const date = new Date(Date.UTC(2023, 0, 1 + days)).toISOString().slice(0, 10);
  return {
    source: "suraj520",
    source_ticket_id: md5(`${emailTrim}||${productCanon}||${narrative ?? ""}`),
    customer_name: nameFromEmail(emailTrim, index),
    customer_email: emailTrim,
    customer_age: 18 + (hEmail % 63),
    customer_gender: GENDERS[hEmail % 100 < 46 ? 0 : hEmail % 100 < 93 ? 1 : 2]!,
    product_purchased: productCanon,
    date_of_purchase: date,
    ticket_type: capitalizeFirst(type ?? ""),
    ticket_priority: capitalizeFirst(priority ?? ""),
    ticket_channel: channelFromHash(hChannel),
    ticket_subject: subject?.trim() || null,
    complaint_narrative: narrative?.trim() || null,
    is_synthetic: true,
  };
}

/** Load mapped tickets from CSV, converting the parquet first if needed. */
export async function loadSuraj520Rows(): Promise<MappedTicket[]> {
  let csv = TICKETS_CSV;
  try {
    await stat(csv);
  } catch {
    const parquet = TICKETS_PARQUET;
    await stat(parquet);
    log("tickets CSV missing — converting parquet via python3");
    const res = spawnSync("python3", ["scripts/convert-suraj520.py"], { stdio: "inherit" });
    if (res.status !== 0) throw new Error("python3 conversion failed (is pyarrow installed?)");
  }
  const raw = await readFile(csv, "utf8");
  const rows = parseCsv(raw);
  if (rows.length < 2) throw new Error(`empty CSV: ${csv}`);
  const header = rows[0]!.cells.map((c) => c.trim());
  const expected = ["Customer Email", "Product Purchased", "Ticket Type", "Ticket Subject", "Combined Text", "Ticket Priority"];
  const missing = expected.filter((c) => !header.includes(c));
  if (missing.length > 0) throw new Error(`CSV missing columns: ${missing.join(", ")}`);

  const mapped: MappedTicket[] = [];
  const seen = new Set<string>();
  for (const row of rows.slice(1)) {
    const t = mapSuraj520Row(row.cells, mapped.length);
    if (seen.has(t.source_ticket_id)) continue; // dedupe on natural key (keep first)
    seen.add(t.source_ticket_id);
    mapped.push(t);
  }
  return mapped;
}

const PRIORITIES = new Set(["Critical", "High", "Medium", "Low"]);
const CHANNELS = new Set(["Social Media", "Email", "Phone", "Chat"]);
const GENDERS_SET = new Set(["Male", "Female", "Other"]);

function validateTicket(t: MappedTicket, violations: string[]): void {
  if (!PRIORITIES.has(t.ticket_priority)) violations.push(`priority=${t.ticket_priority}`);
  if (!CHANNELS.has(t.ticket_channel)) violations.push(`channel=${t.ticket_channel}`);
  if (!GENDERS_SET.has(t.customer_gender)) violations.push(`gender=${t.customer_gender}`);
  if (!(t.customer_age >= 0 && t.customer_age <= 120)) violations.push(`age=${t.customer_age}`);
}

// ---------------------------------------------------------------------------
// ingestTickets
// ---------------------------------------------------------------------------

const TICKET_COLS = [
  "source", "source_ticket_id", "customer_name", "customer_email", "customer_age",
  "customer_gender", "product_purchased", "date_of_purchase", "ticket_type",
  "ticket_priority", "ticket_channel", "ticket_subject", "complaint_narrative", "is_synthetic",
] as const;

const TICKET_TS = [
  "$1::text[]", "$2::text[]", "$3::text[]", "$4::text[]", "$5::int[]",
  "$6::text[]", "$7::text[]", "$8::date[]", "$9::text[]", "$10::text[]",
  "$11::text[]", "$12::text[]", "$13::text[]", "$14::bool[]",
] as const;

export async function ingestTickets(source: TicketSource, opts: IngestOptions = {}): Promise<IngestSummary> {
  const dryRun = opts.dryRun ?? false;
  if (source !== "suraj520") {
    throw new Error(`ingestTickets("${source}") not implemented yet — provision ${source} raw data and add a mapper (contract API preserved)`);
  }
  if (!dryRun) await preflight("tickets");

  const tickets = await loadSuraj520Rows();
  log(`suraj520: ${tickets.length} mapped rows (${embeddingsEnabled() ? "openai" : "hash-fallback"} embeddings; dim ${embeddingDim()})`);

  const violations: string[] = [];
  for (const t of tickets) validateTicket(t, violations);
  if (violations.length > 0) {
    const sample = violations.slice(0, 10).join(", ");
    throw new Error(`enum/check violations (${violations.length}): ${sample} — fix mapper before commit`);
  }

  if (dryRun) {
    log("--- dry-run: first 5 mapped rows ---");
    for (const t of tickets.slice(0, 5)) log(JSON.stringify(t));
    const channels: Record<string, number> = {};
    for (const t of tickets) channels[t.ticket_channel] = (channels[t.ticket_channel] ?? 0) + 1;
    log("channel distribution:", JSON.stringify(channels));
    log("no DB writes performed");
    return { source, rowsRead: tickets.length, rowsInserted: 0, rowsUpdated: 0, failures: [], embeddingMode: embeddingsEnabled() ? "openai" : "hash", dryRun };
  }

  const pool = getPool();
  const updateCols = TICKET_COLS.slice(1).map((c) => `${c} = EXCLUDED.${c}`).join(", ");
  let inserted = 0;
  let updated = 0;

  for (let i = 0; i < tickets.length; i += BATCH) {
    const batch = tickets.slice(i, i + BATCH);
    const cols = TICKET_COLS.map((c) => batch.map((t) => t[c]));
    const res = await pool.query(
      `INSERT INTO tickets (${TICKET_COLS.join(", ")})
       SELECT * FROM unnest(${TICKET_TS.join(", ")})
         AS u(${TICKET_COLS.join(", ")})
       ON CONFLICT (source, source_ticket_id) DO UPDATE SET ${updateCols}, updated_at = now()
       RETURNING (xmax = 0) AS inserted`,
      cols,
    );
    for (const r of res.rows) {
      if (r.inserted) inserted++;
      else updated++;
    }
    log(`tickets batch ${i + 1}-${i + batch.length}/${tickets.length} (inserted=${inserted}, updated=${updated})`);
  }

  log(`done: ${tickets.length} rows (${inserted} inserted, ${updated} updated)`);
  return { source, rowsRead: tickets.length, rowsInserted: inserted, rowsUpdated: updated, failures: [], embeddingMode: embeddingsEnabled() ? "openai" : "hash", dryRun };
}

// ---------------------------------------------------------------------------
// ingestManuals
// ---------------------------------------------------------------------------

const CHUNK_COLS = ["doc_id", "chunk_index", "chunk_text", "page_start", "page_end", "section", "heading_path", "embedding"] as const;

export interface ManualsOptions extends IngestOptions {
  only?: string; // filename filter, e.g. "lg_oled_55b9pla.pdf"
}

export async function ingestManuals(dir: string, opts: ManualsOptions = {}): Promise<IngestSummary> {
  const dryRun = opts.dryRun ?? false;
  if (!dryRun) await preflight("document_chunks");

  const files = (await readdir(dir)).filter((f) => extname(f).toLowerCase() === ".pdf").sort();
  const targets = opts.only ? files.filter((f) => f === opts.only) : files;
  if (targets.length === 0) throw new Error(`no PDFs in ${dir}${opts.only ? ` matching "${opts.only}"` : ""}`);

  const failures: string[] = [];
  let chunksTotal = 0;
  let docsTotal = 0;

  for (const file of targets) {
    const path = join(dir, file);
    try {
      log(`parsing ${file} …`);
      const parsed = await parsePdf(path);
      const chunks = chunkDocument(parsed);
      log(`  ${parsed.pageCount} pages, ${parsed.totalChars} chars → ${chunks.length} chunks`);

      if (dryRun) {
        log("  --- dry-run sample chunks (3) ---");
        for (const c of chunks.slice(0, 3)) {
          log(`  [${c.chunkIndex}] section=${c.section ?? "—"} path=${c.headingPath ?? "—"} pages=${c.pageStart ?? "?"}-${c.pageEnd ?? "?"} chars=${c.text.length}`);
          log(`    ${c.text.slice(0, 140).replace(/\n/g, " ")}…`);
        }
        docsTotal++;
        chunksTotal += chunks.length;
        continue;
      }

      const docId = await upsertDocument({ ...parsed, filePath: file });
      const embeddings = await embedTexts(chunks.map((c) => c.text));
      await upsertChunks(docId, chunks, embeddings);
      docsTotal++;
      chunksTotal += chunks.length;
      log(`  committed doc_id=${docId} chunks=${chunks.length} (${embeddingsEnabled() ? "openai" : "hash"} embeddings)`);
    } catch (err) {
      failures.push(`${file}: ${(err as Error).message}`);
      warn(`FAILED ${file}:`, (err as Error).message);
    }
  }

  log(`done: ${docsTotal} docs, ${chunksTotal} chunks${failures.length > 0 ? `, ${failures.length} failures` : ""}`);
  return { source: `manuals:${dir}`, rowsRead: targets.length, rowsInserted: docsTotal, rowsUpdated: 0, failures, embeddingMode: embeddingsEnabled() ? "openai" : "hash", docs: docsTotal, chunks: chunksTotal, dryRun };
}

async function upsertDocument(parsed: { docName: string; filePath: string; pageCount: number; totalChars: number }): Promise<number> {
  const { rows } = await getPool().query(
    `INSERT INTO documents (doc_name, file_path, doc_type, page_count, total_chars)
     VALUES ($1, $2, 'pdf', $3, $4)
     ON CONFLICT (file_path) DO UPDATE SET
       doc_name = EXCLUDED.doc_name, doc_type = EXCLUDED.doc_type,
       page_count = EXCLUDED.page_count, total_chars = EXCLUDED.total_chars
     RETURNING doc_id`,
    [parsed.docName, parsed.filePath, parsed.pageCount, parsed.totalChars],
  );
  return rows[0]!.doc_id as number;
}

async function upsertChunks(docId: number, chunks: DocumentChunk[], embeddings: number[][]): Promise<void> {
  if (chunks.length === 0) return;
  const pool = getPool();
  for (let i = 0; i < chunks.length; i += BATCH) {
    const batch = chunks.slice(i, i + BATCH);
    const embs = embeddings.slice(i, i + BATCH);
    const cols = [
      batch.map(() => docId),
      batch.map((c) => c.chunkIndex),
      batch.map((c) => c.text),
      batch.map((c) => c.pageStart),
      batch.map((c) => c.pageEnd),
      batch.map((c) => c.section),
      batch.map((c) => c.headingPath),
      embs.map((e) => `[${e.join(",")}]`),
    ];
    await pool.query(
      `INSERT INTO document_chunks (${CHUNK_COLS.join(", ")})
       SELECT d, i, t, ps, pe, s, h, e::vector
       FROM unnest($1::bigint[], $2::int[], $3::text[], $4::int[], $5::int[], $6::text[], $7::text[], $8::text[])
         AS u(d, i, t, ps, pe, s, h, e)
       ON CONFLICT (doc_id, chunk_index) DO UPDATE SET
         chunk_text = EXCLUDED.chunk_text, page_start = EXCLUDED.page_start,
         page_end = EXCLUDED.page_end, section = EXCLUDED.section,
         heading_path = EXCLUDED.heading_path, embedding = EXCLUDED.embedding`,
      cols,
    );
  }
}

async function preflight(table: "tickets" | "document_chunks"): Promise<void> {
  const { rows } = await getPool().query(
    "SELECT to_regclass($1) AS t",
    [table === "tickets" ? "public.tickets" : "public.document_chunks"],
  );
  if (!rows[0]!.t) {
    throw new Error(`table ${table} missing — run \`npm run db:migrate\` first (applies src/db/schema.sql)`);
  }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

function usage(): string {
  return `usage:
  npm run ingest -- --all [--dry-run]                       # tickets + manuals (default)
  npm run ingest -- --tickets [--source suraj520] [--dry-run]
  npm run ingest -- --manuals [--dir config/data/manuals] [--only file.pdf] [--dry-run]`;
}

const isMain = process.argv[1] && import.meta.url === new URL(`file://${process.argv[1]}`).href;

if (isMain) {
  const argv = process.argv.slice(2);
  const flag = (name: string): string | undefined => {
    const i = argv.indexOf(name);
    return i >= 0 ? argv[i + 1] : undefined;
  };
  const has = (name: string) => argv.includes(name);
  const dryRun = has("--dry-run");
  const mode = has("--tickets") ? "tickets" : has("--manuals") ? "manuals" : "all";

  (async () => {
    try {
      log(`embedding backend: ${embeddingsEnabled() ? `openai (${EMBEDDING_MODEL})` : "hash-fallback (OPENAI_API_KEY unset)"} dim=${embeddingDim()}`);
      const summaries: IngestSummary[] = [];
      if (mode === "tickets" || mode === "all") {
        summaries.push(await ingestTickets((flag("--source") as TicketSource) ?? "suraj520", { dryRun }));
      }
      if (mode === "manuals" || mode === "all") {
        summaries.push(await ingestManuals(flag("--dir") ?? "config/data/manuals", { dryRun, only: flag("--only") }));
      }
      log("summary:", JSON.stringify(summaries, null, 2));
      const failed = summaries.some((s) => s.failures.length > 0);
      if (failed) process.exitCode = 1;
    } catch (err) {
      console.error("[ingest] fatal:", (err as Error).message);
      console.error(usage());
      process.exitCode = 1;
    } finally {
      const { closePool } = await import("../db/pool.js");
      await closePool();
    }
  })();
}
