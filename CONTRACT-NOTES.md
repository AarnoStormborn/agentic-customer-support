# CONTRACT-NOTES — retrieval-core track deviations & notes

Orchestrator + other tracks: read before integration. The **interface** in
`docs/design/integration-contract.md` is unchanged; these are implementation notes.

## 1. Embedding fallback (NO OPENAI_API_KEY found in env / .env / ~/.pi/agent/auth.json)

- `src/retrieval/embed.ts` uses the **OpenAI API when `OPENAI_API_KEY` is set**
  (batched 100, retry ×5 backoff on 429/5xx). Otherwise it emits **deterministic
  feature-hash embeddings** (FNV-1a hashing trick, word+bigram+trigram tokens,
  L2-normalized, dim 1536). Deterministic across runs/machines → ingest stays
  idempotent; cosine similarity approximates token overlap.
- All verification in this track used the fallback (`embeddingMode: "hash"`).
  To switch to real embeddings: set `OPENAI_API_KEY`, `npm run ingest -- --all`
  (upserts converge; HNSW index rebuild not needed at this scale).
- `EMBEDDING_MODEL=text-embedding-3-small` → `vector(1536)` pinned in schema.sql.
  Changing model → new dim → re-embed (data-management §6).

## 2. LG manual URL

The task-specified `https://gscs-b2c.lge.com/downloadFile?fileId=uzd9l7qf1exIqd8TH2Yvbg`
returned an **HTML error page, not a PDF** (verified 2026-08-11). Used the dustin.eu
retail mirror from docs/data-research.md §4 / data-management §3.2
(`lg_oled_55b9pla.pdf`, 770 KB, parses to 19 pages). Noted in
`scripts/provision-data.sh`.

## 3. searchHybrid merge semantics (kb + sql)

- `"kb"` = document_chunks: FTS (GIN tsvector) + pgvector cosine (HNSW) fused by
  RRF in one SQL statement (`1/(60+rank)`, k=60, 50 candidates each).
- `"sql"` = tickets FTS over `complaint_narrative || ticket_subject || product_purchased`,
  scored RRF-style `1/(60+rank)` for scale consistency with kb.
- When both sources enabled, each contributes `ceil(topK/2)` results; **kb results
  come first**, then sql. `topK` counts per source, not total (documented here, not
  in the contract — the agent/tools layer can re-order by score or per-source caps).
- `filter` currently allowlists `{ docName, section }` only.

## 4. Schema field names (data-management §2 canonical over backend-agent-retrieval §4.5)

Used `docs/design/data-management.md` §2 DDL exactly (per track instructions):
`tickets.status` (NOT `ticket_status`), `documents.file_path` unique,
`document_chunks.section / heading_path / page_start / page_end / chunk_text`.
`HybridResult.source` maps: `section`/`heading_path` → `sectionPath`, `page_start`
→ `page`, `chunk_text` → `text`. `url` is always `null` (documents table has no URL
column yet — add provenance URLs later if needed).

## 5. ingestTickets scope

Only `"suraj520"` implemented (parquet → CSV → map → upsert). `"cfpb"`/`"comcast"`
throw a clear NotImplemented error until their raw files are provisioned + mappers
written — the contract signature is preserved.

## 6. Extra exports beyond the contract

- `embedTexts` (contract) + `embeddingsEnabled()`, `embeddingDim()`, `EMBEDDING_MODEL`
  (helpers for agent tools / UI to display which backend is active).
- `closePool()` for CLI exit hygiene; `parsePdf`/`chunkDocument` exported for the
  BullMQ ingest worker (api-streaming phase).

## 7. Dependencies

No new npm packages added — `pg`, `openai`, `pdf-parse`, `dotenv` were already in
package.json. suraj520 parquet → CSV conversion uses `python3` + `pyarrow` (on PATH;
see DEPS.md).

## 8. Verification commands (all run in this worktree)

```
npm run db:migrate            # schema applied; vector@0.5.1 + pg_trgm@1.6
npm run ingest -- --all       # 8,469 tickets (idempotent upsert) + 3 manuals → 282 chunks
npm run query -- "lg tv wifi reset"   # hybrid results (see FINISH report)
```
