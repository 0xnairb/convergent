"""Integration tests for Git-for-World-State: branch, commit, diff, merge, and instant revert."""

from __future__ import annotations

import os
import uuid

import psycopg

from convergent_worker.branches import (
    commit_to_branch,
    create_branch,
    diff_branches,
    merge_branch,
    revert_commit,
)

PG_DSN = os.environ.get(
    "CONVERGENT_PG_DSN",
    "postgresql://convergent:convergent_dev_password@127.0.0.1:55432/convergent",
)


def _active_texts(conn: psycopg.Connection, org_id: uuid.UUID, space_id: uuid.UUID, branch: str) -> list[str]:
    rows = conn.execute(
        """
        SELECT canonical_text
        FROM knowledge_blocks
        WHERE org_id = %s AND space_id = %s AND branch_name = %s AND valid_to IS NULL
        """,
        (org_id, space_id, branch),
    ).fetchall()
    return [str(r[0]) for r in rows]


def test_git_world_state_branch_diff_merge_and_instant_revert() -> None:
    org_id = uuid.uuid4()
    space_id = uuid.uuid4()

    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        # 1. Commit baseline architecture decision on `main`
        c_main = commit_to_branch(
            conn,
            org_id=org_id,
            space_id=space_id,
            branch_name="main",
            source_uri="adr://001-auth.md",
            text="Auth service uses opaque session cookies with 24h TTL.",
            author="alice",
            message="Initial auth architecture",
        )
        assert len(c_main.block_ids) == 1

        # 2. Alice branches `alice/oauth-pkce` off `main` and commits an update + new runbook
        create_branch(conn, org_id=org_id, space_id=space_id, new_branch="alice/oauth-pkce", from_branch="main", author="alice")
        commit_to_branch(
            conn,
            org_id=org_id,
            space_id=space_id,
            branch_name="alice/oauth-pkce",
            source_uri="adr://001-auth.md",
            text="Auth service uses OAuth 2.1 PKCE (S256) with 1h short-lived JWTs.",
            author="alice",
            message="Upgrade auth to OAuth 2.1 PKCE",
        )
        commit_to_branch(
            conn,
            org_id=org_id,
            space_id=space_id,
            branch_name="alice/oauth-pkce",
            source_uri="runbook://jwt-rotation.md",
            text="Rotate JWT signing keys every 30 days via Valkey dual-key overlap.",
            author="alice",
            message="Add JWT rotation runbook",
        )

        # Verify strict branch isolation: `main` MUST still have ONLY opaque session cookies!
        main_before = _active_texts(conn, org_id, space_id, "main")
        assert any("opaque session cookies" in t for t in main_before)
        assert not any("OAuth 2.1 PKCE" in t for t in main_before)
        assert not any("JWT signing keys" in t for t in main_before)

        # 3. Semantic diff between `alice/oauth-pkce` and `main`
        diff = diff_branches(conn, org_id=org_id, space_id=space_id, source_branch="alice/oauth-pkce", target_branch="main")
        assert len(diff.added) == 1 and diff.added[0]["source_uri"] == "runbook://jwt-rotation.md"
        assert len(diff.updated) == 1 and diff.updated[0]["source_uri"] == "adr://001-auth.md"

        # 4. Merge `alice/oauth-pkce` -> `main` (multiplayer broadcast + atomic supersession + KV eviction outbox)
        merged = merge_branch(
            conn,
            org_id=org_id,
            space_id=space_id,
            source_branch="alice/oauth-pkce",
            target_branch="main",
            author="alice",
        )
        assert len(merged.applied_block_ids) == 2
        assert len(merged.superseded_block_ids) == 1

        main_after = _active_texts(conn, org_id, space_id, "main")
        assert any("OAuth 2.1 PKCE" in t for t in main_after)
        assert any("JWT signing keys" in t for t in main_after)
        assert not any("opaque session cookies" in t for t in main_after)

        # 5. Instant revert of the merge commit restores `opaque session cookies` and removes OAuth/runbook
        rev = revert_commit(
            conn,
            org_id=org_id,
            space_id=space_id,
            branch_name="main",
            commit_id=merged.merge_commit_id,
            author="bob",
        )
        assert len(rev.deactivated_block_ids) == 2
        assert len(rev.restored_block_ids) == 1

        main_reverted = _active_texts(conn, org_id, space_id, "main")
        assert any("opaque session cookies" in t for t in main_reverted)
        assert not any("OAuth 2.1 PKCE" in t for t in main_reverted)
        assert not any("JWT signing keys" in t for t in main_reverted)
