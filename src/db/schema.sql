-- ============================================================================
-- agentic-customer-support v2 — schema.sql
-- Canonical DDL per docs/design/data-management.md §2 (owner-approved fresh design).
-- Applied idempotently by `runSchema()` (src/db/migrate.ts); DDL never runs from
-- agent tools — migrations only via this file / the migration script.
--
-- Vector dimension is pinned to vector(1536) == OpenAI text-embedding-3-small
-- (EMBEDDING_MODEL default). Switching models requires a new column + re-embed
-- (see data-management §6 "Vector dimension lock-in").
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector: vector type + HNSW index
CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- trigram GIN for LIKE / narrative search

-- ---------------------------------------------------------------------------
-- 1. tickets — SQL retrieval target (suraj520 / cfpb / comcast)
--    Natural key: (source, source_ticket_id) → idempotent upsert.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id           BIGSERIAL PRIMARY KEY,
    -- provenance (idempotency key)
    source              TEXT NOT NULL,            -- 'suraj520' | 'cfpb' | 'comcast' | ...
    source_ticket_id    TEXT NOT NULL,            -- natural key; suraj520: md5(email||product||narrative)
    -- core fields
    customer_name       TEXT,
    customer_email      TEXT,
    customer_age        INT  CHECK (customer_age BETWEEN 0 AND 120),
    customer_gender     TEXT CHECK (customer_gender IN ('Male','Female','Other')),
    product_purchased   TEXT NOT NULL,
    date_of_purchase    DATE,
    ticket_type         TEXT NOT NULL,
    ticket_priority     TEXT CHECK (ticket_priority IN ('Critical','High','Medium','Low')),
    ticket_channel      TEXT,  -- suraj520 enum (Social Media/Email/Phone/Chat) or CFPB (Web/Postal mail/Referral/Fax)
    -- extensions
    ticket_subject      TEXT,
    complaint_narrative TEXT,
    company             TEXT,
    state               TEXT,
    zip_code            TEXT,
    status              TEXT,
    is_synthetic        BOOLEAN NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source, source_ticket_id),
    -- generated FTS vector for indexed search at scale (Phase 5b.3: inline
    -- to_tsvector over 3.76M rows was a 46s seq scan; GIN makes it ms).
    -- ticket_type included so "refund tickets"-style queries match by type
    -- (found by the eval harness, 5b.6).
    search_tsv tsvector GENERATED ALWAYS AS (
      to_tsvector('english',
        COALESCE(complaint_narrative, '') || ' ' ||
        COALESCE(ticket_subject, '')   || ' ' ||
        COALESCE(product_purchased, '') || ' ' ||
        COALESCE(ticket_type, ''))
    ) STORED
);

-- GIN full-text index (tickets search + hybrid retrieval sql source)
CREATE INDEX IF NOT EXISTS tickets_search_tsv_gin_idx ON tickets USING gin (search_tsv);

-- btree indexes for the SQL agent's typical filter/group patterns
CREATE INDEX IF NOT EXISTS tickets_priority_idx      ON tickets (ticket_priority);
CREATE INDEX IF NOT EXISTS tickets_type_idx          ON tickets (ticket_type);
CREATE INDEX IF NOT EXISTS tickets_product_idx       ON tickets (product_purchased);
CREATE INDEX IF NOT EXISTS tickets_channel_idx       ON tickets (ticket_channel);
CREATE INDEX IF NOT EXISTS tickets_purchase_date_idx ON tickets (date_of_purchase);
CREATE INDEX IF NOT EXISTS tickets_source_idx        ON tickets (source);
-- fast substring search on narrative (LIKE '%foo%')
CREATE INDEX IF NOT EXISTS tickets_narrative_trgm_idx ON tickets USING gin (complaint_narrative gin_trgm_ops);

-- ---------------------------------------------------------------------------
-- 2. documents — one row per ingested source file (PDF/CSV); natural key: file_path
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    doc_id      BIGSERIAL PRIMARY KEY,
    doc_name    TEXT NOT NULL,              -- display name, e.g. lg_oled_55b9pla.pdf
    file_path   TEXT NOT NULL,              -- path under config/data/manuals/
    doc_type    TEXT,                       -- 'pdf' | 'html' | 'md'
    page_count  INT,
    total_chars INT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (file_path)
);

-- ---------------------------------------------------------------------------
-- 3. document_chunks — vector/lexical retrieval target
--    Natural key: (doc_id, chunk_index) → idempotent upsert.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id     BIGSERIAL PRIMARY KEY,
    doc_id       BIGINT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index  INT  NOT NULL,             -- order within the document
    chunk_text   TEXT NOT NULL,
    -- structural metadata (powers source citations in the SSE `done` payload)
    page_start   INT,
    page_end     INT,
    section      TEXT,                      -- e.g. 'Troubleshooting'
    heading_path TEXT,                      -- breadcrumb: 'Network > Wi-Fi Connection'
    embedding    vector(1536),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (doc_id, chunk_index)
);

-- 1) HNSW vector index (cosine distance <=>, m/ef_construction tuned per design)
CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx
    ON document_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);
-- 2) GIN full-text index (hybrid retrieval: FTS + vector + RRF)
CREATE INDEX IF NOT EXISTS document_chunks_fts_idx
    ON document_chunks USING gin (to_tsvector('english', chunk_text));
-- 3) btree for navigation / section filters
CREATE INDEX IF NOT EXISTS document_chunks_doc_idx     ON document_chunks (doc_id, chunk_index);
CREATE INDEX IF NOT EXISTS document_chunks_section_idx ON document_chunks (section);

-- ---------------------------------------------------------------------------
-- 4. chats / chat_messages — session persistence (Phase 5b.7)
--    The in-memory registry is hydrated from these on boot; every finished turn
--    is written back (best-effort) so the UI sidebar + history survive restarts.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chats (
    chat_id         TEXT PRIMARY KEY,            -- 'chat_<hex>'
    conversation_id TEXT NOT NULL,               -- groups follow-ups into one thread
    status          TEXT NOT NULL,               -- running | done | error | canceled
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ,
    message_count   INT  NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id         BIGSERIAL PRIMARY KEY,
    chat_id    TEXT NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
    role       TEXT NOT NULL,                    -- user | assistant | tool
    content    JSONB NOT NULL,                   -- raw AgentMessage content parts
    turn_index INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chats_created_idx          ON chats (created_at DESC);
CREATE INDEX IF NOT EXISTS chats_conversation_idx     ON chats (conversation_id);
CREATE INDEX IF NOT EXISTS chat_messages_chat_id_idx  ON chat_messages (chat_id);
