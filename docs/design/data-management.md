# Data Management Design — v2 (pi SDK rebuild)

Status: design (Phase 3.4 / plan item 4.2). Owner defers data provisioning; this doc makes it turnkey.
Companion doc: `docs/data-research.md` (all URLs below verified there on 2026-08-11).
Baselines extended here: legacy `config/schema.yml`, `config/ingest.py`, `config/sql/*.sql` (kept untouched in git history).

---

## 1. Final dataset selection

### Primary bundle

| Role | Dataset | License | One-line rationale |
|---|---|---|---|
| SQL `tickets` (schema-compatible) | **Customer Support Ticket Dataset** — Kaggle `suraj520/customer-support-ticket-dataset` (HF parquet mirror `gorkemsevinc/customer_support_tickets`) | CC0 | 8,469 rows whose products, `Ticket Type` and `Ticket Priority` values match legacy `schema.yml` almost exactly (LG Smart TV, iPhone, Sony Xperia, HP Pavilion, Dell XPS, LG OLED; `technical issue`/`refund request`/`billing inquiry`/…; `critical/high/medium/low`) — near drop-in for the `tickets` table. |
| SQL + narrative text | **CFPB Consumer Complaint Database** — `https://files.consumerfinance.gov/ccdb/complaints.csv.zip` | CC0 (public domain) | The only large, real dataset pairing 18 structured fields with free-text consumer narratives; ideal second source and the honest "real data" counterweight to synthetic tickets. |
| Vector RAG manuals | **Manufacturer PDFs** — LG OLED, HP Pavilion, Dell XPS, Sony Xperia + archive.org appliance manuals (§3 below) | Vendor-free redistribution / public archive | Real `.pdf` files for the exact products named in `schema.yml`; required by `RAGIngestion`-style `pdf → chunks → embeddings` pipeline. |

### Fallbacks (in order, from `docs/data-research.md`)
1. **FCC CGB Consumer Complaints** (`https://opendata.fcc.gov/api/views/3xyp-aqkj/rows.csv?accessType=DOWNLOAD`, public domain) — big telecom structured corpus; **no narrative text** (categorical `Issue` only).
2. **Comcast telecom complaints CSV** (`raw.githubusercontent.com/kuhimans/.../Comcast_telecom_complaints_data.csv`, ~2,224 rows) — small, real complaint text + `Received Via` (maps to `ticket_channel`); no license file (FCC-derived) → internal use.
3. **Thinknook "Customer Support on Twitter"** (Kaggle `thoughtvector/customer-support-on-twitter`, ~3.98M rows) — real consumer↔brand conversation text; license unclear (Twitter ToS) → research use, subsample.
4. **Twitter US Airline Sentiment** (Kaggle `crowdflower/twitter-airline-sentiment`, 14,640 rows) — social-channel + sentiment labels; **CC BY-NC-SA 4.0**.
5. **Tobi-Bueck customer-support-tickets** (Kaggle `tobiasbueck/multilingual-customer-support-tickets`, 61.8k labeled ticket emails, **CC BY 4.0**; HF mirror is CC BY-NC — avoid) — labeled subject/body/answer pairs.
6. **Console-AI/IT-helpdesk-synthetic-tickets** (HF, MIT) — small MIT-licensed fallback for load testing.

**Scope note:** suraj520 also contains products not in `schema.yml` (canon eos, gopro hero, nest thermostat, amazon echo, roomba, microsoft surface, dyson vacuum, …). We keep all products in the table (data integrity); manuals cover the schema's primary families, and web fallback covers the rest. No join is required between tickets and documents at the schema level.

---

## 2. Data model

### 2.1 Principles
- **Natural keys for idempotency**: `tickets (source, source_ticket_id)`; `documents (file_path)`; `document_chunks (doc_id, chunk_index)`.
- **Provenance always visible**: every row carries `source` (+ `source_ticket_id`); synthesized identity fields flagged with `is_synthetic`.
- **SQL agent is SELECT-only** (AGENTS.md): DDL/migrations run via a migration tool (`node-pg-migrate` or plain `.sql` applied by a script), never via agent tools.
- Vector dimension **1536** (OpenAI `text-embedding-3-small`, the model used by legacy `ingest.py`); changing the model ⇒ new column/dimension + re-embed (see §6).
- Legacy equivalent tables (`t_docs`, `t_docs_chunks`, `config/sql/pgvector.sql`) are superseded; v2 names below are canonical.

### 2.2 DDL — `tickets` (extends legacy `schema.yml` fields)

```sql
CREATE EXTENSION IF NOT EXISTS pg_trgm;          -- optional, for LIKE/narrative search

CREATE TABLE tickets (
    ticket_id           BIGSERIAL PRIMARY KEY,
    -- provenance (idempotency key)
    source              TEXT NOT NULL,            -- 'suraj520' | 'cfpb' | 'comcast' | ...
    source_ticket_id    TEXT NOT NULL,            -- natural key from source; suraj520: md5(email||product||narrative)
    -- legacy schema.yml fields
    customer_name       TEXT,                     -- suraj520: derived from email; cfpb/comcast: NULL (no PII published)
    customer_email      TEXT,                     -- cfpb/comcast: NULL
    customer_age        INT  CHECK (customer_age BETWEEN 0 AND 120),   -- suraj520: synthesized
    customer_gender     TEXT CHECK (customer_gender IN ('Male','Female','Other')),  -- synthesized
    product_purchased   TEXT NOT NULL,            -- e.g. 'LG Smart TV'
    date_of_purchase    DATE,                     -- suraj520: synthesized; cfpb: Date received; comcast: Date
    ticket_type         TEXT NOT NULL,            -- 'Refund request' | 'Billing inquiry' | 'Product inquiry' | 'Cancellation request' | 'Technical issue'
    ticket_priority     TEXT CHECK (ticket_priority IN ('Critical','High','Medium','Low')),
    ticket_channel      TEXT CHECK (ticket_channel IN ('Social Media','Email','Phone','Chat')),
    -- v2 extensions
    ticket_subject      TEXT,                     -- suraj520 Ticket Subject / Tobi-Bueck subject
    complaint_narrative TEXT,                     -- suraj520 Combined Text / CFPB narrative / Comcast Customer Complaint
    company             TEXT,                     -- cfpb Company / comcast Comcast
    state               TEXT,                     -- cfpb/comcast geography
    zip_code            TEXT,
    status              TEXT,                     -- open/closed/resolved (comcast, cfpb response state)
    is_synthetic        BOOLEAN NOT NULL DEFAULT FALSE,  -- TRUE if any identity/purchase field was synthesized
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source, source_ticket_id)
);

-- btree indexes for the SQL agent's typical filter/group patterns
CREATE INDEX tickets_priority_idx     ON tickets (ticket_priority);
CREATE INDEX tickets_type_idx         ON tickets (ticket_type);
CREATE INDEX tickets_product_idx      ON tickets (product_purchased);
CREATE INDEX tickets_channel_idx      ON tickets (ticket_channel);
CREATE INDEX tickets_purchase_date_idx ON tickets (date_of_purchase);
CREATE INDEX tickets_source_idx       ON tickets (source);
-- optional: fast substring search on narrative (used by LIKE '%foo%' queries)
CREATE INDEX tickets_narrative_trgm_idx ON tickets USING gin (complaint_narrative gin_trgm_ops);
```

### 2.3 DDL — vector store (`documents`, `document_chunks`)

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    doc_id      BIGSERIAL PRIMARY KEY,
    doc_name    TEXT NOT NULL,           -- file name, e.g. lg_oled_55b9pla.pdf
    file_path   TEXT NOT NULL,           -- path under config/data/manuals/
    doc_type    TEXT,                    -- 'pdf' | 'html' | 'md'
    page_count  INT,
    total_chars INT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (file_path)
);

CREATE TABLE document_chunks (
    chunk_id     BIGSERIAL PRIMARY KEY,
    doc_id       BIGINT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index  INT  NOT NULL,          -- order within the document (idempotency key)
    chunk_text   TEXT NOT NULL,
    -- structural metadata
    page_start   INT,                    -- first page containing this chunk (PDFs)
    page_end     INT,
    section      TEXT,                   -- e.g. 'Troubleshooting', 'Safety Instructions'
    heading_path TEXT,                   -- breadcrumb: 'Home > Settings > Picture > ...'
    embedding    vector(1536) NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (doc_id, chunk_index)
);

-- 1) HNSW vector index (cosine; equivalent to legacy ip_ops when vectors are normalized)
CREATE INDEX document_chunks_embedding_hnsw_idx
    ON document_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);
-- 2) GIN full-text index (hybrid retrieval: FTS + vector + RRF)
CREATE INDEX document_chunks_fts_idx
    ON document_chunks USING gin (to_tsvector('english', chunk_text));
-- 3) btree for navigation / section filters
CREATE INDEX document_chunks_doc_idx    ON document_chunks (doc_id, chunk_index);
CREATE INDEX document_chunks_section_idx ON document_chunks (section);
```

Legacy note: `config/sql/retrieval.sql` used `embedding <#> :emb LIMIT :top_k` (negative inner product). With `text-embedding-3-small`, vectors are normalized so inner-product rank ≈ cosine rank; v2 uses `<=>` (cosine) for clarity. HNSW `m/ef_construction` tuned at ingest-time query-time via `SET hnsw.ef_search`.

### 2.4 Schema mismatch: suraj520 CSV → legacy `schema.yml` (+ mapping/enrichment)

| suraj520 column | → tickets field | Transform |
|---|---|---|
| `Customer Email` | `customer_email` | as-is (already synthetic-looking placeholders) |
| `Product Purchased` | `product_purchased` | title-case canonicalization (`lg smart tv` → `LG Smart TV`; keep unknown products as-is) |
| `Ticket Type` | `ticket_type` | title-case (`technical issue` → `Technical issue`) |
| `Ticket Priority` | `ticket_priority` | capitalize (`critical` → `Critical`) |
| `Ticket Subject` | `ticket_subject` | as-is |
| `Combined Text` | `complaint_narrative` | as-is |
| *(missing)* `customer_name` | synthesize deterministically | email local-part split on `[._-]` + digits, title-case each token, join; fallback `customer_{n}` |
| *(missing)* `customer_age` | synthesize deterministically | `18 + (md5(email)::bigint % 63)` → 18–80 |
| *(missing)* `customer_gender` | synthesize deterministically | `md5(email)::bigint % 100`: <46 Male, <93 Female, else Other (≈45/47/8) |
| *(missing)* `date_of_purchase` | synthesize deterministically | `2023-01-01 + (md5(email||product)::bigint % 730) days` (fixed 2-year window) |
| *(missing)* `ticket_channel` | synthesize deterministically | `md5(email||ticket_type)::bigint % 4` → Social Media/Email/Phone/Chat (weighted Email 40%, rest 20%) |

**Deterministic synthesis rules (non-negotiable):**
- All synthesis derives from **hashes of existing values** (`md5(customer_email || …)`), never `random()`; identical input ⇒ identical output across runs and machines → idempotent ingest, reproducible demos.
- Synthesized rows set `is_synthetic = TRUE` and `source_ticket_id = md5(customer_email || product_purchased || complaint_narrative)` (stable natural key).
- CFPB/Comcast rows: `customer_name/email/age/gender = NULL` (no PII published), `is_synthetic = FALSE`, `source_ticket_id =` CFPB `Complaint ID` / Comcast `Ticket #`.
- The SQL agent must tolerate NULLs in identity columns (document in the agent's system prompt).

---

## 3. Provisioning scripts

### 3.1 File layout under `config/data/`

```
config/data/
├── .gitignore                  # committed: "*" + "!.gitkeep" (dir exists in git, data never committed)
├── .gitkeep
├── raw/                        # untouched upstream downloads
│   ├── suraj520/tickets.parquet                 # 1.1 MB (HF mirror)
│   ├── cfpb/complaints.csv.zip                  # 1.41 GB full dump (snapshot date pinned in script)
│   └── comcast/Comcast_telecom_complaints_data.csv   # ~1 MB
├── tickets/                    # processed, ingest-ready CSVs (gitignored)
│   └── tickets.csv             # suraj520 mapped+enriched (8,469 rows)
│   └── cfpb_narratives.csv     # filtered CFPB subset (see 3.2)
│   └── comcast.csv             # as-is + mapped columns
└── manuals/                    # PDFs for RAG (gitignored)
    ├── lg_oled_55b9pla.pdf                     # 770 KB
    ├── hp_pavilion_user_guide.pdf              # 758 KB
    ├── dell_xps13_9310_service_manual.pdf      # 44.7 MB
    ├── sony_xperia_1v_manual.pdf               # 2.2 MB
    ├── google_pixel_7_manual.pdf               # ManualsLib (manual step, see note)
    └── kenmore_fridge_25331115308.pdf          # archive.org appliance manual
```

### 3.2 `scripts/provision-data.sh` (idempotent; skip-if-exists)

```bash
#!/usr/bin/env bash
# Idempotent: re-running skips files that already exist with expected size (curl -C - resumes).
set -euo pipefail
DATA=config/data
mkdir -p "$DATA/raw/suraj520" "$DATA/raw/cfpb" "$DATA/raw/comcast" "$DATA/tickets" "$DATA/manuals"

fetch() { # $1=url $2=out $3=min_bytes
  if [[ -f "$2" ]] && [[ $(stat -f%z "$2" 2>/dev/null || stat -c%s "$2") -ge "$3" ]]; then
    echo "skip  $2"; return; fi
  echo "fetch $2"; curl -fL --retry 3 --retry-delay 2 -C - -o "$2" "$1"
}

# 1) suraj520 tickets (CC0) — HF parquet mirror, no Kaggle auth needed
fetch "https://huggingface.co/datasets/gorkemsevinc/customer_support_tickets/resolve/main/data/train-00000-of-00001.parquet" \
      "$DATA/raw/suraj520/tickets.parquet" 1000000
# (Kaggle CLI alternative: kaggle datasets download -d suraj520/customer-support-ticket-dataset -p config/data/raw/suraj520)

# 2) CFPB full dump (CC0) — static file, curl-friendly (search UI/API are bot-protected)
fetch "https://files.consumerfinance.gov/ccdb/complaints.csv.zip" "$DATA/raw/cfpb/complaints.csv.zip" 1400000000

# 3) Comcast telecom complaints
fetch "https://raw.githubusercontent.com/kuhimans/Comcast-Telecom-Consumer-Complaints-Analysis/master/Comcast_telecom_complaints_data.csv" \
      "$DATA/raw/comcast/comcast.csv" 500000

# 4) Manuals (verified URLs, see docs/data-research.md §4)
fetch "https://media.dustin.eu/media/d200001003283774/oled55b9pla-55-4k-smart-oled-user-manual.pdf" "$DATA/manuals/lg_oled_55b9pla.pdf" 500000
fetch "http://www.hp.com/ctg/Manual/bpi04347.pdf" "$DATA/manuals/hp_pavilion_user_guide.pdf" 500000
curl -fL -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0" --retry 3 \
     -C - -o "$DATA/manuals/dell_xps13_9310_service_manual.pdf" \
     "https://dl.dell.com/topicspdf/xps-13-9310-laptop_Service-Manual_en-us.pdf"   # 44.7 MB; Dell blocks default UA
fetch "https://theinformr.com/downloads/cell-phones/manuals/2797/sony-xperia-1-v-manual.pdf" "$DATA/manuals/sony_xperia_1v_manual.pdf" 1000000
fetch "https://archive.org/download/Kenmore_25331115308_Refrigerator_User_Manual/Kenmore_25331115308_Refrigerator_User_Manual.pdf" \
      "$DATA/manuals/kenmore_fridge_25331115308.pdf" 500000

# 5) Manual step (no clean direct URL): Google Pixel manual
echo "Google Pixel: download PDF from https://www.manualslib.com/manual/2876995/Google-Pixel-7.html (Download button) -> $DATA/manuals/google_pixel_7_manual.pdf"
```

### 3.3 CFPB subset (narrative-only, keeps demo sizes sane)

The full 1.41 GB dump is downloaded once; a filter step (`scripts/filter-cfpb.mjs`, zero deps or `pg-query`-free csv parser) produces `config/data/tickets/cfpb_narratives.csv`:
- keep rows where `Consumer complaint narrative` is non-empty (only a minority of complaints publish a narrative),
- optional narrower window: `Date received >= 2023-01-01` and `Product` in (`Credit card or prepaid card`, `Checking or savings account`, `Bank account or service`),
- expected output: **~5k–50k rows** depending on filters; snapshot date pinned for reproducibility.

**Verified constraint:** the CFPB search UI and `search/api/v1` endpoint are behind Akamai bot protection (scripted access returns 403 "Access Denied" from datacenter IPs). The static `files.consumerfinance.gov/ccdb/complaints.csv.zip` works with plain `curl` (HEAD verified). The HF loader `CFPB/consumer-finance-complaints` pulls the same static file (but runs arbitrary loader code — prefer the direct zip + local filter).

### 3.4 Verification & `.gitignore`

- **Expected sizes (verify after provisioning):** `du -sh config/data/raw config/data/manuals` → raw ≈ 1.41 GB (mostly CFPB zip), manuals ≈ 50 MB; row counts: suraj520 8,469 (verify with `python3 -c "import pandas as pd; print(len(pd.read_parquet('config/data/raw/suraj520/tickets.parquet')))"` or `node` + `parquetjs`), Comcast 2,224 lines, CFPB subset 5k–50k.
- **`.gitignore` strategy:**
  - Root `.gitignore`: add `config/data/` (covers `raw/`, `tickets/`, `manuals/`).
  - Commit `config/data/.gitignore` containing `*` and `!.gitkeep` so the directory skeleton (and this doc's layout) survives clones while data never enters git.
  - Also ignore `*.parquet` / `*.zip` outside `config/data/` as belt-and-braces; `.env` already ignored.

---

## 4. Ingest pipeline design (v2, TypeScript)

### 4.1 CSV → `tickets` (map → coerce → enrich → upsert)

1. **Read** raw CSV/Parquet (`config/data/raw/suraj520/tickets.parquet`, CFPB subset, Comcast).
2. **Map/rename** per §2.4 table; drop unknown columns.
3. **Coerce types** (`ticket_priority` → enum validation against check constraint; `date_of_purchase` → `YYYY-MM-DD`; ages → int; fail-fast on rows violating enums, log a sample).
4. **Enrich** deterministically (§2.4): `customer_name`, `customer_age`, `customer_gender`, `date_of_purchase`, `ticket_channel`, `source_ticket_id`, `is_synthetic`.
5. **Dedupe** on `(source, source_ticket_id)` (suraj520 has duplicate emails+texts); keep first.
6. **Upsert** into Postgres: `INSERT ... ON CONFLICT (source, source_ticket_id) DO UPDATE SET <all cols>, updated_at = now()`.
   - Batched multi-row inserts (1,000/batch) with `pg` pool; optionally `pg-copy-streams` for bulk first load.
   - Idempotent by natural key: re-running a file converges; deleting rows not present in the source is out of scope (append-only + explicit `--replace-source=suraj520` option that `DELETE FROM tickets WHERE source='suraj520'` first).
7. **Dry-run** (`--dry-run`): parse/coerce/enrich in memory, print first 5 mapped rows + column stats + row count + enum-violation summary; **no DB writes, no embedding calls**; exit code 1 on schema violations.

### 4.2 PDF → text → structural chunks → embeddings → pgvector

1. **Text extraction** — `pdfjs-dist` (or `pdf-parse`) per page; keep page numbers; skip scanned/image-only PDFs (report as failed, `--resume` later).
2. **Section detection** — build a heading tree from font-size heuristics (pdfjs text items expose font size: heading = size ≥ 1.4× body, or bold) plus regex anchors (`Troubleshooting|Safety Instructions|Specifications|FAQ|^[0-9]+(\.[0-9]+)*\s+[A-Z]`); fall back to page-boundary chunks when no headings are detected.
3. **Chunking (section-based with overlap + metadata)**:
   - target ~512 tokens (~1,500–2,000 chars) per chunk, cap 2,500 chars;
   - overlap ~10–15% (~100–200 chars) between consecutive chunks within a section;
   - a section larger than the cap is split with overlap; tiny sections merge with the next;
   - each chunk carries `chunk_index`, `page_start`, `page_end`, `section`, `heading_path`.
   - (This replaces legacy `split_text`'s blind fixed-window slicing in `ingest.py` — that was a known weakness; see `docs/project-analysis.md`.)
4. **Embeddings** — OpenAI `text-embedding-3-small`, batch size 100 (legacy `MAX_BATCH_SIZE`), async with 60s timeout per batch; retry with exponential backoff (base 1s, ×2, max 5 tries) on `429`/`5xx`; on persistent failure: log batch IDs, mark chunk as pending, continue — re-run converges.
5. **Upsert** — `documents` by `file_path` (insert if new, else reuse `doc_id`); `document_chunks` by `(doc_id, chunk_index)` with `ON CONFLICT DO UPDATE` (embedding + text + metadata).
6. **Idempotency** — full re-run recomputes and upserts; add `--only=file.pdf` and `--resume` (skips docs whose chunks already exist) options.
7. **Logging/progress** — structured logs (pino) per file: pages processed, sections found, chunks created, embedding batch `x/y`, ETA; final summary `{docs, chunks, tickets, rows_skipped, failures[]}`; non-zero exit on total failure, partial success reported as warnings.
8. **Dry-run** — extract + chunk + (optionally) print 3 sample chunks with metadata and estimated token counts; **no embedding API calls, no DB writes**.

### 4.3 Config surface (env/CLI)

```bash
DATABASE_URL=postgres://...   OPENAI_API_KEY=...   EMBEDDING_MODEL=text-embedding-3-small
npm run ingest:tickets  -- --source suraj520 --dry-run        # or --commit
npm run ingest:tickets  -- --source cfpb --file config/data/tickets/cfpb_narratives.csv
npm run ingest:manuals  -- --dir config/data/manuals --dry-run # or --commit
```

---

## 5. Dev workflow

Order of operations (each step verifiable before the next):

1. **Start DB** — `cp .env.example .env` (fill `DATABASE_URL`, `OPENAI_API_KEY`); `docker compose up -d` (ankane/pgvector + redis:7); wait for `pg_isready`.
2. **Provision** — `bash scripts/provision-data.sh` (skips existing; verify sizes per §3.4); do the one manual step (Pixel PDF).
3. **Migrate** — apply `migrations/001_tickets.sql`, `migrations/002_vector.sql` (§2 DDL); `\dt` shows `tickets`, `documents`, `document_chunks`; confirm `SELECT extversion FROM pg_extension WHERE extname IN ('vector','pg_trgm')`.
4. **Ingest tickets (dry-run first)** — `npm run ingest:tickets -- --source suraj520 --dry-run` → inspect mapping output; then `--commit`; repeat for `cfpb` and `comcast` subsets. Verify: `SELECT source, count(*) FROM tickets GROUP BY 1;`.
5. **Ingest manuals (dry-run first)** — `npm run ingest:manuals -- --dry-run` → inspect 3 sample chunks; then `--commit`. Verify: `SELECT count(*), count(DISTINCT doc_id) FROM document_chunks;`.
6. **Smoke tests** — `npm run smoke-test` runs the query set below and asserts shapes/counts.

### Smoke-test query set (3 SQL + 3 vector + 1 hybrid)

SQL:
```sql
-- S1: priority distribution (expect 4 rows summing to ingested ticket count)
SELECT ticket_priority, count(*) FROM tickets GROUP BY 1 ORDER BY 2 DESC;

-- S2: technical issues by product (expect lg smart tv / lg oled / dell xps / hp pavilion / sony xperia in top rows)
SELECT product_purchased, count(*) FROM tickets
WHERE ticket_type = 'Technical issue' GROUP BY 1 ORDER BY 2 DESC LIMIT 5;

-- S3: channel breakdown + narrative coverage (expect 4 channels; with_narrative > 0 from CFPB subset)
SELECT ticket_channel, count(*) FROM tickets GROUP BY 1 ORDER BY 2 DESC;
SELECT count(*) FILTER (WHERE complaint_narrative IS NOT NULL AND length(complaint_narrative) > 50) AS with_narrative,
       count(DISTINCT source) AS sources FROM tickets;
```

Vector (embedding computed at runtime by the app; smoke script embeds the query then runs):
```sql
-- V1: LG manual troubleshooting (expect chunks from lg_oled_55b9pla.pdf, section 'Troubleshooting')
SELECT d.doc_name, c.section, left(c.chunk_text, 120)
FROM document_chunks c JOIN documents d USING (doc_id)
ORDER BY c.embedding <=> :emb('black screen on my LG OLED TV')::vector LIMIT 3;

-- V2: Dell manual (expect dell_xps13_9310_service_manual.pdf)
SELECT d.doc_name, c.page_start, left(c.chunk_text, 120)
FROM document_chunks c JOIN documents d USING (doc_id)
ORDER BY c.embedding <=> :emb('Dell XPS laptop will not charge with the original charger')::vector LIMIT 3;

-- V3: Sony manual (expect sony_xperia_1v_manual.pdf)
SELECT d.doc_name, c.section, left(c.chunk_text, 120)
FROM document_chunks c JOIN documents d USING (doc_id)
ORDER BY c.embedding <=> :emb('how to factory reset a Sony Xperia phone')::vector LIMIT 3;
```

Hybrid (FTS + vector + RRF — shape used by the retrieval agent):
```sql
-- H1: 'hdmi no signal on tv' → union of FTS and vector candidates, Reciprocal-Rank-Fusion score
WITH fts AS (
  SELECT c.chunk_id, ts_rank_cd(to_tsvector('english', c.chunk_text), plainto_tsquery('english', 'hdmi no signal')) AS score
  FROM document_chunks c WHERE to_tsvector('english', c.chunk_text) @@ plainto_tsquery('english', 'hdmi no signal')
  ORDER BY score DESC LIMIT 10
), vec AS (
  SELECT c.chunk_id, 1 - (c.embedding <=> :emb('hdmi no signal on tv')::vector) AS score
  FROM document_chunks c ORDER BY c.embedding <=> :emb('hdmi no signal on tv')::vector LIMIT 10
), rrf AS (
  SELECT chunk_id, sum(1.0 / (60 + rn)) AS rrf_score FROM (
    SELECT chunk_id, row_number() OVER (ORDER BY score DESC) rn FROM fts
    UNION ALL
    SELECT chunk_id, row_number() OVER (ORDER BY score DESC) rn FROM vec
  ) t GROUP BY chunk_id
)
SELECT d.doc_name, c.section, rrf.rrf_score
FROM rrf JOIN document_chunks c USING (chunk_id) JOIN documents d USING (doc_id)
ORDER BY rrf_score DESC LIMIT 3;
```
Pass criteria: S1/S2/S3 return expected shapes (no exceptions, non-zero counts); V1–V3 each return chunks whose `doc_name` matches the expected manual; H1 returns ≥1 chunk and mixes FTS+vector sources (RRF fusion working).

---

## 6. Risks & license notes

- **Licenses (from `docs/data-research.md`, all verified):**
  - ✅ **CC0 / public domain**: CFPB (federal work), suraj520 tickets, FCC. Safest; no attribution or NC restrictions.
  - ⚠️ **CC BY-NC**: Tobi-Bueck **HF mirror** (avoid; use Kaggle **CC BY 4.0** copy instead). **CC BY-NC-SA**: airline sentiment (demo-only). Thinknook Twitter: **no explicit license** + Twitter/X ToS → research use only, don't ship in a commercial product.
  - ⚠️ **Comcast CSV**: repo has no license file; data is FCC-derived (public-domain provenance) but the compiled file's terms are unclear → internal use only.
  - ⚠️ **Manual PDFs**: vendor copyright (LG/HP/Dell/Sony/Google). Fine for internal RAG ingestion (that's their purpose), **do not redistribute**; archive.org appliance manuals are public-archive items (open access).
- **CFPB size**: full dump 1.41 GB CSV / 1.57 GB JSON — pin a snapshot date, download once, filter locally (§3.3). Search UI/API are bot-protected (403 on scripted access — verified); the static file endpoint is the reliable path. Dataset updates daily ⇒ snapshot for reproducibility.
- **Synthetic-data caveats**: suraj520 is template-generated (repetitive phrasing, "resolution pending" endings) — fine for pipeline/agent demos, **not** a training corpus; identity fields (name/age/gender/date/channel) are deterministically synthesized placeholders — always filter/flag via `is_synthetic` and document in the SQL-agent prompt. CFPB is real but self-selected complainants (not a statistical sample) and only a minority of rows carry narratives.
- **Vector dimension lock-in**: DDL pins `vector(1536)` (text-embedding-3-small). Switching embedding models requires a new column or re-embed + rebuild of the HNSW index.
- **Embedding cost/rate limits**: corpus is small (8,469 ticket narratives + ≤50k CFPB narratives + ~5–10k chunks) ≈ tens of thousands of calls ≈ low cost; batch (100), retry/backoff, `--resume` — a mid-run crash must not force a full re-embed (upsert + `--resume` handles it).
- **Operational**: `config/data/` never committed (§3.4); SQL agent is SELECT-only so DDL/DML stay under the migration tool; `docker compose up -d` brings up both Postgres (pgvector) and Redis (BullMQ) — data services are prerequisites for any ingest run.
