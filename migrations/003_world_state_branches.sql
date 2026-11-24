-- Project Convergent: Git-for-World-State (Branches, Commits, Merge & Instant Revert)

CREATE TABLE IF NOT EXISTS memory_commits (
  id UUID PRIMARY KEY,
  org_id UUID NOT NULL,
  space_id UUID NOT NULL,
  branch_name TEXT NOT NULL,
  parent_commit_id UUID REFERENCES memory_commits(id),
  author TEXT NOT NULL,
  message TEXT NOT NULL,
  action TEXT NOT NULL DEFAULT 'commit', -- 'init' | 'branch' | 'commit' | 'merge' | 'revert'
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_memory_commits_branch
  ON memory_commits (org_id, space_id, branch_name, created_at DESC);

CREATE TABLE IF NOT EXISTS memory_branches (
  id UUID PRIMARY KEY,
  org_id UUID NOT NULL,
  space_id UUID NOT NULL,
  name TEXT NOT NULL,
  head_commit_id UUID REFERENCES memory_commits(id),
  created_by TEXT NOT NULL DEFAULT 'system',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (org_id, space_id, name)
);

ALTER TABLE knowledge_blocks
  ADD COLUMN IF NOT EXISTS branch_name TEXT NOT NULL DEFAULT 'main';

ALTER TABLE knowledge_blocks
  ADD COLUMN IF NOT EXISTS commit_id UUID REFERENCES memory_commits(id);

CREATE INDEX IF NOT EXISTS idx_kb_active_branch
  ON knowledge_blocks (org_id, space_id, branch_name)
  WHERE valid_to IS NULL;
