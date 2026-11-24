"""Git-for-World-State: branch, commit, diff, merge, and instant revert over primary-source memory."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
import uuid

import psycopg

from .embedding import embed_text, pgvector_literal
from .memory import blake3_hex, canonical_text, split_blocks


@dataclass(frozen=True)
class BranchCommitResult:
    commit_id: uuid.UUID
    branch_name: str
    block_ids: list[uuid.UUID]
    superseded_block_ids: list[uuid.UUID]


@dataclass(frozen=True)
class BranchDiff:
    source_branch: str
    target_branch: str
    added: list[dict[str, Any]]
    updated: list[dict[str, Any]]
    unchanged_count: int


@dataclass(frozen=True)
class MergeResult:
    merge_commit_id: uuid.UUID
    source_branch: str
    target_branch: str
    applied_block_ids: list[uuid.UUID]
    superseded_block_ids: list[uuid.UUID]


@dataclass(frozen=True)
class RevertResult:
    revert_commit_id: uuid.UUID
    reverted_commit_id: uuid.UUID
    deactivated_block_ids: list[uuid.UUID]
    restored_block_ids: list[uuid.UUID]


def branch_clearance_scope(space_id: uuid.UUID, branch_name: str) -> str:
    """Scope KV cache keys to (space, branch) so branch experiments never collide with main."""
    if branch_name == "main":
        return f"space:{space_id}"
    return f"space:{space_id}:branch:{branch_name}"


def ensure_branch(
    conn: psycopg.Connection,
    *,
    org_id: uuid.UUID,
    space_id: uuid.UUID,
    branch_name: str = "main",
    author: str = "system",
) -> uuid.UUID:
    """Ensure a branch exists and return its current head_commit_id (or newly created init commit)."""
    row = conn.execute(
        "SELECT head_commit_id FROM memory_branches WHERE org_id = %s AND space_id = %s AND name = %s",
        (org_id, space_id, branch_name),
    ).fetchone()
    if row is not None and row[0] is not None:
        return row[0]

    init_commit_id = uuid.uuid4()
    branch_id = uuid.uuid4()
    with conn.transaction():
        conn.execute(
            """
            INSERT INTO memory_commits (id, org_id, space_id, branch_name, parent_commit_id, author, message, action)
            VALUES (%s, %s, %s, %s, NULL, %s, %s, 'init')
            """,
            (init_commit_id, org_id, space_id, branch_name, author, f"Initialize branch {branch_name}"),
        )
        conn.execute(
            """
            INSERT INTO memory_branches (id, org_id, space_id, name, head_commit_id, created_by)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (org_id, space_id, name) DO UPDATE SET head_commit_id = COALESCE(memory_branches.head_commit_id, EXCLUDED.head_commit_id)
            """,
            (branch_id, org_id, space_id, branch_name, init_commit_id, author),
        )
    return init_commit_id


def create_branch(
    conn: psycopg.Connection,
    *,
    org_id: uuid.UUID,
    space_id: uuid.UUID,
    new_branch: str,
    from_branch: str = "main",
    author: str = "agent",
) -> uuid.UUID:
    """Create an isolated memory branch snapshotting active blocks from `from_branch`."""
    parent_head = ensure_branch(conn, org_id=org_id, space_id=space_id, branch_name=from_branch, author=author)
    branch_commit_id = uuid.uuid4()
    branch_id = uuid.uuid4()
    new_scope = branch_clearance_scope(space_id, new_branch)

    with conn.transaction():
        conn.execute(
            """
            INSERT INTO memory_commits (id, org_id, space_id, branch_name, parent_commit_id, author, message, action)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'branch')
            """,
            (branch_commit_id, org_id, space_id, new_branch, parent_head, author, f"Branch {new_branch} from {from_branch}"),
        )
        conn.execute(
            """
            INSERT INTO memory_branches (id, org_id, space_id, name, head_commit_id, created_by)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (branch_id, org_id, space_id, new_branch, branch_commit_id, author),
        )

        # Copy-on-write snapshot of active blocks from `from_branch` onto `new_branch`
        active_rows = conn.execute(
            """
            SELECT document_id, block_type, canonical_text, content_blake3, citations, embedding
            FROM knowledge_blocks
            WHERE org_id = %s AND space_id = %s AND branch_name = %s AND valid_to IS NULL
            """,
            (org_id, space_id, from_branch),
        ).fetchall()
        for doc_id, block_type, text, bhash, citations, emb in active_rows:
            conn.execute(
                """
                INSERT INTO knowledge_blocks (
                  id, document_id, org_id, space_id, clearance_scope, branch_name, commit_id,
                  block_type, canonical_text, content_blake3, citations, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                """,
                (
                    uuid.uuid4(),
                    doc_id,
                    org_id,
                    space_id,
                    new_scope,
                    new_branch,
                    branch_commit_id,
                    block_type,
                    text,
                    bhash,
                    json.dumps(citations),
                    emb,
                ),
            )
    return branch_commit_id


def commit_to_branch(
    conn: psycopg.Connection,
    *,
    org_id: uuid.UUID,
    space_id: uuid.UUID,
    branch_name: str,
    source_uri: str,
    text: str,
    author: str,
    message: str,
    block_type: str = "decision",
) -> BranchCommitResult:
    """Commit primary-source content to a specific branch, superseding prior blocks for `source_uri` on that branch only."""
    parent_head = ensure_branch(conn, org_id=org_id, space_id=space_id, branch_name=branch_name, author=author)
    norm = canonical_text(text)
    doc_hash = blake3_hex(norm)
    doc_id = uuid.uuid4()
    commit_id = uuid.uuid4()
    scope = branch_clearance_scope(space_id, branch_name)
    parsed = split_blocks(norm, source_uri)

    new_block_ids: list[uuid.UUID] = []
    superseded_ids: list[uuid.UUID] = []

    with conn.transaction():
        conn.execute(
            """
            INSERT INTO memory_commits (id, org_id, space_id, branch_name, parent_commit_id, author, message, action)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'commit')
            """,
            (commit_id, org_id, space_id, branch_name, parent_head, author, message),
        )
        conn.execute(
            "UPDATE memory_branches SET head_commit_id = %s WHERE org_id = %s AND space_id = %s AND name = %s",
            (commit_id, org_id, space_id, branch_name),
        )
        conn.execute(
            """
            INSERT INTO documents (id, org_id, space_id, clearance_scope, source_uri, content_blake3)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (doc_id, org_id, space_id, scope, source_uri, doc_hash),
        )

        prior_rows = conn.execute(
            """
            SELECT kb.id, kb.content_blake3, kb.clearance_scope
            FROM knowledge_blocks kb
            JOIN documents d ON d.id = kb.document_id
            WHERE kb.org_id = %s AND kb.space_id = %s AND kb.branch_name = %s
              AND d.source_uri = %s AND kb.valid_to IS NULL
            FOR UPDATE OF kb
            """,
            (org_id, space_id, branch_name, source_uri),
        ).fetchall()

        new_hashes: list[str] = []
        for part, citations in parsed:
            bid = uuid.uuid4()
            bhash = blake3_hex(part)
            emb = pgvector_literal(embed_text(part))
            conn.execute(
                """
                INSERT INTO knowledge_blocks (
                  id, document_id, org_id, space_id, clearance_scope, branch_name, commit_id,
                  block_type, canonical_text, content_blake3, citations, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                """,
                (bid, doc_id, org_id, space_id, scope, branch_name, commit_id, block_type, part, bhash, json.dumps(citations), emb),
            )
            new_block_ids.append(bid)
            new_hashes.append(bhash)

        primary_new_id = new_block_ids[0]
        primary_new_hash = new_hashes[0]
        for old_id, old_hash, old_scope in prior_rows:
            conn.execute(
                "UPDATE knowledge_blocks SET valid_to = now(), superseded_by = %s WHERE id = %s",
                (primary_new_id, old_id),
            )
            superseded_ids.append(old_id)
            _emit_superseded_outbox(conn, org_id, space_id, old_scope, old_id, old_hash, primary_new_id, primary_new_hash, f"branch_commit:{branch_name}")

    return BranchCommitResult(
        commit_id=commit_id,
        branch_name=branch_name,
        block_ids=new_block_ids,
        superseded_block_ids=superseded_ids,
    )


def diff_branches(
    conn: psycopg.Connection,
    *,
    org_id: uuid.UUID,
    space_id: uuid.UUID,
    source_branch: str,
    target_branch: str = "main",
) -> BranchDiff:
    """Compute semantic diff of active world-state blocks between `source_branch` and `target_branch`."""
    def _load(bname: str) -> dict[str, dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT d.source_uri, kb.id, kb.document_id, kb.content_blake3, kb.canonical_text, kb.citations, kb.block_type, kb.embedding
            FROM knowledge_blocks kb
            JOIN documents d ON d.id = kb.document_id
            WHERE kb.org_id = %s AND kb.space_id = %s AND kb.branch_name = %s AND kb.valid_to IS NULL
            ORDER BY kb.valid_from DESC
            """,
            (org_id, space_id, bname),
        ).fetchall()
        out: dict[str, dict[str, Any]] = {}
        for uri, bid, doc_id, bhash, text, cits, btype, emb in rows:
            if uri not in out:
                out[uri] = {
                    "source_uri": uri,
                    "block_id": str(bid),
                    "document_id": doc_id,
                    "content_blake3": bhash,
                    "canonical_text": text,
                    "citations": cits,
                    "block_type": btype,
                    "embedding": emb,
                }
        return out

    src = _load(source_branch)
    tgt = _load(target_branch)
    added: list[dict[str, Any]] = []
    updated: list[dict[str, Any]] = []
    unchanged = 0

    for uri, item in src.items():
        if uri not in tgt:
            added.append({"source_uri": uri, "content_blake3": item["content_blake3"], "canonical_text": item["canonical_text"]})
        elif tgt[uri]["content_blake3"] != item["content_blake3"]:
            updated.append({
                "source_uri": uri,
                "old_text": tgt[uri]["canonical_text"],
                "new_text": item["canonical_text"],
                "old_block_id": tgt[uri]["block_id"],
                "new_hash": item["content_blake3"],
            })
        else:
            unchanged += 1

    return BranchDiff(
        source_branch=source_branch,
        target_branch=target_branch,
        added=added,
        updated=updated,
        unchanged_count=unchanged,
    )


def merge_branch(
    conn: psycopg.Connection,
    *,
    org_id: uuid.UUID,
    space_id: uuid.UUID,
    source_branch: str,
    target_branch: str = "main",
    author: str = "agent",
    message: str | None = None,
) -> MergeResult:
    """Atomically merge `source_branch` into `target_branch`, evicting stale target KV keys and broadcasting merge event."""
    target_head = ensure_branch(conn, org_id=org_id, space_id=space_id, branch_name=target_branch, author=author)
    target_scope = branch_clearance_scope(space_id, target_branch)
    merge_commit_id = uuid.uuid4()
    merge_msg = message or f"Merge branch '{source_branch}' into '{target_branch}'"

    applied_ids: list[uuid.UUID] = []
    superseded_ids: list[uuid.UUID] = []

    with conn.transaction():
        conn.execute(
            """
            INSERT INTO memory_commits (id, org_id, space_id, branch_name, parent_commit_id, author, message, action)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'merge')
            """,
            (merge_commit_id, org_id, space_id, target_branch, target_head, author, merge_msg),
        )
        conn.execute(
            "UPDATE memory_branches SET head_commit_id = %s WHERE org_id = %s AND space_id = %s AND name = %s",
            (merge_commit_id, org_id, space_id, target_branch),
        )

        src_rows = conn.execute(
            """
            SELECT d.source_uri, kb.document_id, kb.block_type, kb.canonical_text, kb.content_blake3, kb.citations, kb.embedding
            FROM knowledge_blocks kb
            JOIN documents d ON d.id = kb.document_id
            WHERE kb.org_id = %s AND kb.space_id = %s AND kb.branch_name = %s AND kb.valid_to IS NULL
            """,
            (org_id, space_id, source_branch),
        ).fetchall()

        for uri, doc_id, btype, text, bhash, cits, emb in src_rows:
            existing = conn.execute(
                """
                SELECT kb.id, kb.content_blake3, kb.clearance_scope
                FROM knowledge_blocks kb
                JOIN documents d ON d.id = kb.document_id
                WHERE kb.org_id = %s AND kb.space_id = %s AND kb.branch_name = %s
                  AND d.source_uri = %s AND kb.valid_to IS NULL
                FOR UPDATE OF kb
                """,
                (org_id, space_id, target_branch, uri),
            ).fetchall()

            # Skip if target already has identical active content
            if existing and all(row[1] == bhash for row in existing):
                continue

            new_bid = uuid.uuid4()
            conn.execute(
                """
                INSERT INTO knowledge_blocks (
                  id, document_id, org_id, space_id, clearance_scope, branch_name, commit_id,
                  block_type, canonical_text, content_blake3, citations, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                """,
                (new_bid, doc_id, org_id, space_id, target_scope, target_branch, merge_commit_id, btype, text, bhash, json.dumps(cits), emb),
            )
            applied_ids.append(new_bid)

            for old_id, old_hash, old_scope in existing:
                conn.execute(
                    "UPDATE knowledge_blocks SET valid_to = now(), superseded_by = %s WHERE id = %s",
                    (new_bid, old_id),
                )
                superseded_ids.append(old_id)
                _emit_superseded_outbox(conn, org_id, space_id, old_scope, old_id, old_hash, new_bid, bhash, f"merge:{source_branch}->{target_branch}")

        # Emit multiplayer broadcast event so connected teammates/agents refresh immediately
        conn.execute(
            """
            INSERT INTO outbox_events (msg_id, subject, payload)
            VALUES (%s, 'convergent.acl.changed', %s::jsonb)
            ON CONFLICT (msg_id) DO NOTHING
            """,
            (
                f"merge:{merge_commit_id}",
                json.dumps({
                    "event": "branch.merged",
                    "org_id": str(org_id),
                    "space_id": str(space_id),
                    "source_branch": source_branch,
                    "target_branch": target_branch,
                    "merge_commit_id": str(merge_commit_id),
                    "applied_blocks": len(applied_ids),
                    "superseded_blocks": len(superseded_ids),
                }),
            ),
        )

    return MergeResult(
        merge_commit_id=merge_commit_id,
        source_branch=source_branch,
        target_branch=target_branch,
        applied_block_ids=applied_ids,
        superseded_block_ids=superseded_ids,
    )


def revert_commit(
    conn: psycopg.Connection,
    *,
    org_id: uuid.UUID,
    space_id: uuid.UUID,
    branch_name: str,
    commit_id: uuid.UUID,
    author: str = "agent",
) -> RevertResult:
    """Instant rollback: deactivate blocks introduced by `commit_id`, restore blocks it superseded, and evict KV."""
    head = ensure_branch(conn, org_id=org_id, space_id=space_id, branch_name=branch_name, author=author)
    revert_commit_id = uuid.uuid4()
    deactivated: list[uuid.UUID] = []
    restored: list[uuid.UUID] = []

    with conn.transaction():
        conn.execute(
            """
            INSERT INTO memory_commits (id, org_id, space_id, branch_name, parent_commit_id, author, message, action)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'revert')
            """,
            (revert_commit_id, org_id, space_id, branch_name, head, author, f"Revert commit {commit_id}"),
        )
        conn.execute(
            "UPDATE memory_branches SET head_commit_id = %s WHERE org_id = %s AND space_id = %s AND name = %s",
            (revert_commit_id, org_id, space_id, branch_name),
        )

        bad_blocks = conn.execute(
            """
            SELECT id, content_blake3, clearance_scope
            FROM knowledge_blocks
            WHERE org_id = %s AND space_id = %s AND branch_name = %s AND commit_id = %s AND valid_to IS NULL
            FOR UPDATE
            """,
            (org_id, space_id, branch_name, commit_id),
        ).fetchall()

        for bad_id, bad_hash, bad_scope in bad_blocks:
            conn.execute("UPDATE knowledge_blocks SET valid_to = now() WHERE id = %s", (bad_id,))
            deactivated.append(bad_id)

            prev_rows = conn.execute(
                """
                SELECT id, content_blake3
                FROM knowledge_blocks
                WHERE org_id = %s AND space_id = %s AND branch_name = %s AND superseded_by = %s
                FOR UPDATE
                """,
                (org_id, space_id, branch_name, bad_id),
            ).fetchall()
            for prev_id, prev_hash in prev_rows:
                conn.execute(
                    "UPDATE knowledge_blocks SET valid_to = NULL, superseded_by = NULL WHERE id = %s",
                    (prev_id,),
                )
                restored.append(prev_id)
                _emit_superseded_outbox(conn, org_id, space_id, bad_scope, bad_id, bad_hash, prev_id, prev_hash, f"revert:{commit_id}")

    return RevertResult(
        revert_commit_id=revert_commit_id,
        reverted_commit_id=commit_id,
        deactivated_block_ids=deactivated,
        restored_block_ids=restored,
    )


def _emit_superseded_outbox(
    conn: psycopg.Connection,
    org_id: uuid.UUID,
    space_id: uuid.UUID,
    clearance_scope: str,
    old_id: uuid.UUID,
    old_hash: str,
    new_id: uuid.UUID,
    new_hash: str,
    reason: str,
) -> None:
    msg_id = f"superseded:{old_id}:{new_id}:{uuid.uuid4().hex[:8]}"
    payload = {
        "tenant_id": str(org_id),
        "space_id": str(space_id),
        "clearance_scope": clearance_scope,
        "old_block_id": str(old_id),
        "old_block_hash": old_hash,
        "new_block_id": str(new_id),
        "new_block_hash": new_hash,
        "reason": reason,
    }
    conn.execute(
        """
        INSERT INTO outbox_events (msg_id, subject, payload)
        VALUES (%s, 'convergent.memory.superseded', %s::jsonb)
        ON CONFLICT (msg_id) DO NOTHING
        """,
        (msg_id, json.dumps(payload)),
    )
