-- Project Convergent: Layered Overlays & Server-Authoritative Changesets (MVCC)

DROP TABLE IF EXISTS memory_branches CASCADE;
DROP TABLE IF EXISTS memory_commits CASCADE;

CREATE TABLE IF NOT EXISTS space_revisions (
  id BIGSERIAL PRIMARY KEY,
  org_id UUID NOT NULL,
  space_id UUID NOT NULL,
  rev BIGINT NOT NULL,
  author TEXT NOT NULL,
  message TEXT NOT NULL,
  action TEXT NOT NULL DEFAULT 'land', -- 'land' | 'undo'
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (org_id, space_id, rev)
);

CREATE TABLE IF NOT EXISTS overlays (
  id UUID PRIMARY KEY,
  org_id UUID NOT NULL,
  space_id UUID NOT NULL,
  base_rev BIGINT NOT NULL DEFAULT 0,
  owner TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'open', -- 'open' | 'landed' | 'discarded'
  landed_rev BIGINT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS overlay_deltas (
  id UUID PRIMARY KEY,
  overlay_id UUID NOT NULL REFERENCES overlays(id) ON DELETE CASCADE,
  op TEXT NOT NULL DEFAULT 'upsert', -- 'upsert' | 'retract'
  source_uri TEXT NOT NULL,
  block_type TEXT NOT NULL DEFAULT 'fact',
  canonical_text TEXT NOT NULL DEFAULT '',
  content_blake3 TEXT NOT NULL DEFAULT '',
  citations JSONB NOT NULL DEFAULT '[]'::jsonb,
  embedding vector(64),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (overlay_id, source_uri)
);

ALTER TABLE knowledge_blocks
  ADD COLUMN IF NOT EXISTS landed_rev BIGINT NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_kb_active_space_rev
  ON knowledge_blocks (org_id, space_id, landed_rev)
  WHERE valid_to IS NULL;
