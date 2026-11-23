CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_search;

CREATE TABLE IF NOT EXISTS documents (
  id              UUID PRIMARY KEY,
  org_id          UUID NOT NULL,
  space_id        UUID NOT NULL,
  clearance_scope TEXT NOT NULL,
  source_uri      TEXT NOT NULL,
  content_blake3  CHAR(64) NOT NULL,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS knowledge_blocks (
  id              UUID PRIMARY KEY,
  document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  org_id          UUID NOT NULL,
  space_id        UUID NOT NULL,
  clearance_scope TEXT NOT NULL,
  block_type      TEXT NOT NULL,
  canonical_text  TEXT NOT NULL,
  content_blake3  CHAR(64) NOT NULL,
  citations       JSONB NOT NULL DEFAULT '[]'::jsonb,
  embedding       VECTOR(64),
  valid_from      TIMESTAMPTZ NOT NULL DEFAULT now(),
  valid_to        TIMESTAMPTZ,
  superseded_by   UUID REFERENCES knowledge_blocks(id),
  heat_score      DOUBLE PRECISION NOT NULL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_kb_active_scope
  ON knowledge_blocks (org_id, space_id, document_id)
  WHERE valid_to IS NULL;

CREATE INDEX IF NOT EXISTS idx_kb_doc_blake3
  ON knowledge_blocks (document_id, content_blake3);

CREATE INDEX IF NOT EXISTS idx_kb_embedding_hnsw
  ON knowledge_blocks USING hnsw (embedding vector_cosine_ops);

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_indexes WHERE indexname = 'idx_kb_bm25'
  ) THEN
    CALL paradedb.create_bm25(
      index_name => 'idx_kb_bm25',
      table_name => 'knowledge_blocks',
      key_field => 'id',
      text_fields => paradedb.field('canonical_text')
    );
  END IF;
EXCEPTION
  WHEN undefined_function THEN
    CREATE INDEX IF NOT EXISTS idx_kb_bm25
      ON knowledge_blocks USING bm25 (id, canonical_text)
      WITH (key_field='id');
END $$;

CREATE TABLE IF NOT EXISTS outbox_events (
  id           BIGSERIAL PRIMARY KEY,
  msg_id       TEXT NOT NULL UNIQUE,
  subject      TEXT NOT NULL,
  payload      JSONB NOT NULL,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  published_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_outbox_unpublished
  ON outbox_events (id)
  WHERE published_at IS NULL;

CREATE TABLE IF NOT EXISTS model_registry (
  model_id     TEXT PRIMARY KEY,
  endpoint_url TEXT NOT NULL,
  is_active    BOOLEAN NOT NULL DEFAULT false,
  warmed_at    TIMESTAMPTZ,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
