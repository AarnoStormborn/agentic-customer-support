/**
 * src/retrieval/ingest-cfpb.ts — CFPB Consumer Complaint full-dump ingest.
 *
 * Owner decision (docs/design/data-management.md §3.3): ingest the FULL dump
 * (~2.9M rows) as the project's large-data training ground. The 1.4 GB zip is
 * provisioned by scripts/provision-data.sh; this module streams the CSV
 * (csv-parse, handles quoted narratives with embedded newlines), maps rows to
 * the tickets schema, and bulk-inserts in batches.
 *
 * PII: CFPB publishes no names/emails — identity columns stay NULL.
 */
import { createReadStream } from "node:fs";
import { parse } from "csv-parse";
import { getPool } from "../db/pool.js";

/** Column name → tickets field. Keys are the CFPB CSV headers. */
const COLUMN_MAP: Record<string, keyof MappedCfpbRow> = {
  "Complaint ID": "source_ticket_id",
  "Date received": "date_of_purchase",
  Product: "product_purchased",
  "Sub-product": "ticket_subject",
  Issue: "ticket_type",
  Company: "company",
  State: "state",
  "ZIP code": "zip_code",
  "Submitted via": "ticket_channel",
  "Consumer complaint narrative": "complaint_narrative",
  "Company response to consumer": "company_response",
};

export interface MappedCfpbRow {
  source_ticket_id: string;
  date_of_purchase: string | null;
  product_purchased: string;
  ticket_subject: string | null;
  ticket_type: string;
  company: string | null;
  state: string | null;
  zip_code: string | null;
  ticket_channel: string | null;
  complaint_narrative: string | null;
  company_response: string | null;
}

/** Map one raw CFPB record (keyed by header name) to a tickets row. */
export function mapCfpbRecord(rec: Record<string, string>): MappedCfpbRow {
  const pick = (col: string): string | null => {
    const v = rec[col]?.trim();
    return v && v.length > 0 ? v : null;
  };

  return {
    source_ticket_id: pick("Complaint ID") ?? "",
    date_of_purchase: pick("Date received"),
    product_purchased: pick("Product") ?? "Unknown",
    ticket_subject: pick("Sub-product"),
    ticket_type: pick("Issue") ?? "General inquiry",
    company: pick("Company"),
    state: pick("State"),
    zip_code: pick("ZIP code"),
    ticket_channel: pick("Submitted via"),
    complaint_narrative: pick("Consumer complaint narrative"),
    company_response: pick("Company response to consumer"),
  };
}

const BATCH = 2000;

/** Stream the CFPB CSV into tickets (batched upsert). Returns a summary. */
export async function ingestCfpbCsv(csvPath: string): Promise<{
  source: string;
  rowsRead: number;
  rowsInserted: number;
  rowsUpdated: number;
  failures: string[];
  dryRun: boolean;
}> {
  const pool = getPool();
  const failures: string[] = [];
  let rowsRead = 0;
  let inserted = 0;
  let updated = 0;
  let batch: MappedCfpbRow[] = [];

  const flush = async (): Promise<void> => {
    if (batch.length === 0) return;
    const res = await pool.query(
      `INSERT INTO tickets (source, source_ticket_id, product_purchased, date_of_purchase,
                            ticket_type, ticket_priority, ticket_channel, ticket_subject,
                            complaint_narrative, company, state, zip_code, is_synthetic)
       SELECT * FROM unnest(
         $1::text[], $2::text[], $3::text[], $4::date[], $5::text[], $6::text[],
         $7::text[], $8::text[], $9::text[], $10::text[], $11::text[], $12::text[], $13::bool[]
       ) AS u(source, source_ticket_id, product_purchased, date_of_purchase, ticket_type,
              ticket_priority, ticket_channel, ticket_subject, complaint_narrative, company,
              state, zip_code, is_synthetic)
       ON CONFLICT (source, source_ticket_id) DO UPDATE SET
         product_purchased = EXCLUDED.product_purchased,
         date_of_purchase = EXCLUDED.date_of_purchase,
         ticket_type = EXCLUDED.ticket_type,
         ticket_channel = EXCLUDED.ticket_channel,
         ticket_subject = EXCLUDED.ticket_subject,
         complaint_narrative = EXCLUDED.complaint_narrative,
         company = EXCLUDED.company,
         state = EXCLUDED.state,
         zip_code = EXCLUDED.zip_code,
         updated_at = now()
       RETURNING (xmax = 0) AS inserted`,
      [
        batch.map(() => "cfpb"),
        batch.map((r) => r.source_ticket_id),
        batch.map((r) => r.product_purchased),
        batch.map((r) => r.date_of_purchase),
        batch.map((r) => r.ticket_type),
        batch.map(() => null), // CFPB has no priority
        batch.map((r) => r.ticket_channel),
        batch.map((r) => r.ticket_subject),
        batch.map((r) => r.complaint_narrative),
        batch.map((r) => r.company),
        batch.map((r) => r.state),
        batch.map((r) => r.zip_code),
        batch.map(() => false),
      ],
    );
    for (const row of res.rows) {
      if (row.inserted) inserted++;
      else updated++;
    }
    batch = [];
  };

  const parser = createReadStream(csvPath).pipe(
    parse({ columns: true, skip_empty_lines: true, relax_column_count: true }),
  );

  for await (const record of parser) {
    rowsRead++;
    try {
      const row = mapCfpbRecord(record as Record<string, string>);
      if (!row.source_ticket_id) {
        failures.push(`row ${rowsRead}: missing Complaint ID — skipped`);
        continue;
      }
      batch.push(row);
      if (batch.length >= BATCH) await flush();
    } catch (err) {
      failures.push(`row ${rowsRead}: ${(err as Error).message}`);
    }
    if (rowsRead % 250_000 === 0) {
      console.log(`[cfpb] ${rowsRead} rows read (inserted=${inserted}, updated=${updated})…`);
    }
  }
  await flush();

  return {
    source: "cfpb",
    rowsRead,
    rowsInserted: inserted,
    rowsUpdated: updated,
    failures,
    dryRun: false,
  };
}
