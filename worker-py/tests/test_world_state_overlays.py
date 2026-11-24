"""Integration tests for Layered Overlays & Server-Authoritative Changesets (Zero-Copy MVCC)."""

from __future__ import annotations

import os
import uuid

import psycopg

from convergent_worker.overlays import (
    land_overlay,
    open_overlay,
    read_mount_view,
    stage_delta,
    undo_revision,
)

PG_DSN = os.environ.get(
    "CONVERGENT_PG_DSN",
    "postgresql://convergent:convergent_dev_password@127.0.0.1:55432/convergent",
)


def test_zero_copy_overlay_auto_rebase_conflict_detection_and_undo() -> None:
    org_id = uuid.uuid4()
    space_id = uuid.uuid4()

    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        # 1. Seed initial world-state at r1
        ov_init = open_overlay(conn, org_id=org_id, space_id=space_id, owner="system")
        stage_delta(conn, overlay_id=ov_init.overlay_id, source_uri="adr://001-auth.md", text="Auth uses session cookies.")
        stage_delta(conn, overlay_id=ov_init.overlay_id, source_uri="adr://002-db.md", text="Primary DB is PostgreSQL 17.")
        r1 = land_overlay(conn, overlay_id=ov_init.overlay_id, author="system", message="Initial world-state")
        assert r1.status == "landed" and r1.rev == 1

        # Verify 0-byte overlay spawn: opening two parallel agent overlays copies ZERO rows in knowledge_blocks
        before_count = conn.execute("SELECT COUNT(*) FROM knowledge_blocks WHERE org_id = %s", (org_id,)).fetchone()[0]
        ov_alice = open_overlay(conn, org_id=org_id, space_id=space_id, owner="agent-alice")
        ov_bob = open_overlay(conn, org_id=org_id, space_id=space_id, owner="agent-bob")
        after_count = conn.execute("SELECT COUNT(*) FROM knowledge_blocks WHERE org_id = %s", (org_id,)).fetchone()[0]
        assert before_count == after_count == 2
        assert ov_alice.base_rev == ov_bob.base_rev == 1

        # 2. Alice stages update to adr://001-auth.md; Bob stages NON-OVERLAPPING runbook://deploy.md
        stage_delta(conn, overlay_id=ov_alice.overlay_id, source_uri="adr://001-auth.md", text="Auth uses OAuth 2.1 PKCE.")
        stage_delta(conn, overlay_id=ov_bob.overlay_id, source_uri="runbook://deploy.md", text="Continuous daily deploys enabled.")

        # Zero-copy mount view for Alice: sees her overlay delta + untouched base DB block (sharing base clearance_scope!)
        alice_view = read_mount_view(conn, org_id=org_id, space_id=space_id, overlay_id=ov_alice.overlay_id)
        by_uri = {b["source_uri"]: b for b in alice_view}
        assert "OAuth 2.1 PKCE" in by_uri["adr://001-auth.md"]["canonical_text"]
        assert by_uri["adr://001-auth.md"]["layer"] == "overlay"
        assert by_uri["adr://002-db.md"]["layer"] == "base"
        assert by_uri["adr://002-db.md"]["clearance_scope"] == f"space:{space_id}"

        # Shared base space is still untouched until Land
        base_view = read_mount_view(conn, org_id=org_id, space_id=space_id)
        assert any("session cookies" in b["canonical_text"] for b in base_view)
        assert not any("OAuth 2.1 PKCE" in b["canonical_text"] for b in base_view)

        # 3. Alice lands at r2; Bob (still pinned to base_rev=1!) auto-rebases & lands cleanly at r3!
        r2 = land_overlay(conn, overlay_id=ov_alice.overlay_id, author="agent-alice", message="Upgrade auth")
        assert r2.status == "landed" and r2.rev == 2

        r3 = land_overlay(conn, overlay_id=ov_bob.overlay_id, author="agent-bob", message="Add deploy runbook")
        assert r3.status == "landed" and r3.rev == 3

        # 4. Stale agent Carol (pinned to base_rev=1) tries to overwrite adr://001-auth.md -> CONFLICT!
        ov_carol = open_overlay(conn, org_id=org_id, space_id=space_id, owner="agent-carol", base_rev=1)
        stage_delta(conn, overlay_id=ov_carol.overlay_id, source_uri="adr://001-auth.md", text="Auth uses mTLS only.")
        r_conflict = land_overlay(conn, overlay_id=ov_carol.overlay_id, author="agent-carol", message="Conflicting auth change")
        assert r_conflict.status == "conflict"
        assert r_conflict.conflicts[0]["source_uri"] == "adr://001-auth.md"
        assert r_conflict.conflicts[0]["current_rev"] == 2

        # 5. Instant Undo of r2 restores `session cookies` while preserving Bob's r3 deploy runbook!
        undone = undo_revision(conn, org_id=org_id, space_id=space_id, rev=2, author="ops")
        assert undone.undo_rev == 4
        assert len(undone.restored_block_ids) == 1

        final_view = read_mount_view(conn, org_id=org_id, space_id=space_id)
        final_texts = [b["canonical_text"] for b in final_view]
        assert any("session cookies" in t for t in final_texts)
        assert any("Continuous daily deploys" in t for t in final_texts)
        assert not any("OAuth 2.1 PKCE" in t for t in final_texts)
