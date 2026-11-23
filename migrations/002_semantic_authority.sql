-- Project Convergent: Source Authority & Conflict Review Queue (idempotent)

ALTER TABLE documents
  ADD COLUMN IF NOT EXISTS source_authority SMALLINT NOT NULL DEFAULT 2;

CREATE TABLE IF NOT EXISTS conflict_reviews (
  id UUID PRIMARY KEY,
  org_id UUID NOT NULL,
  space_id UUID NOT NULL,
  existing_block_id UUID NOT NULL REFERENCES knowledge_blocks(id),
  candidate_block_id UUID NOT NULL REFERENCES knowledge_blocks(id),
  existing_authority SMALLINT NOT NULL,
  candidate_authority SMALLINT NOT NULL,
  confidence REAL NOT NULL,
  reason TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'pending', -- 'pending' | 'approved' | 'rejected'
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  resolved_at TIMESTAMPTZ,
  resolved_by TEXT
);

CREATE INDEX IF NOT EXISTS idx_conflict_reviews_pending
  ON conflict_reviews (org_id, space_id, status)
  WHERE status = 'pending';
