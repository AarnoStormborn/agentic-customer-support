SET TIMEZONE = 'Asia/Kolkata';

CREATE EXTENSION if not EXISTS vector;
CREATE TABLE t_docs (
    id BIGSERIAL PRIMARY KEY,
    doc_name VARCHAR(256) NOT NULL,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE t_docs_chunks (
    id BIGSERIAL PRIMARY KEY,
    doc_id BIGSERIAL NOT NULL REFERENCES t_docs(id),
    chunk jsonb NOT NULL,
    embedding vector(1536) NOT NULL,
    created_at TIMESTAMP DEFAULT now()
);

CREATE INDEX ON t_docs_chunks USING 
hnsw (embedding vector_ip_ops)
WITH (m=16, ef_construction=128);