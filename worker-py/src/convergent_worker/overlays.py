"""Layered Overlays & Server-Authoritative Changesets (Zero-Copy MVCC World-State)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
import uuid

import psycopg

from .embedding import embed_text, pgvector_literal
from .memory import blake3_hex, canonical_text


@dataclass(frozen=True)
class OverlayInfo:
    overlay_id: uuid.UUID
    base_rev: int
    owner: str


@dataclass(frozen=True)
class LandResult:
    status: str  # "landed" | "conflict"
    rev: int | None
    applied_block_ids: list[uuid.UUID]
    superseded_block_ids: list[uuid.UUID]
    conflicts: list[dict[str, Any]]


@dataclass(frozen=True)
class UndoResult:
    undo_rev: int
    undone_rev: int
    deactivated_block_ids: list[uuid.UUID]
    restored_block_ids: list[uuid.UUID]


def current_head_rev(conn: psycopg.Connection, *, org_id: uuid.UUID, space_id: uuid.UUID) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(rev), 0) FROM space_revisions WHERE org_id = %s AND space_id = %s",
        (org_id, space_id),
    ).fetchone()
    return int(row[0]) if row else 0


def open_overlay(
    conn: psycopg.Connection,
    *,
    org_id: uuid.UUID,
    space_id: uuid.UUID,
    owner: str,
    base_rev: int | None = None,
) -> OverlayInfo:
    """Create a 0-byte virtual overlay pinned to `base_rev` (copies ZERO knowledge blocks)."""
    rev = current_head_rev(conn, org_id=org_id, space_id=space_id) if base_rev is None else base_rev
    oid = uuid.uuid4()
    with conn.transaction():
        conn.execute(
            """
            INSERT INTO overlays (id, org_id, space_id, base_rev, owner, status)
            VALUES (%s, %s, %s, %s, %s, 'open')
            """,
            (oid, org_id, space_id, rev, owner),
        )
    return OverlayInfo(overlay_id=oid, base_rev=rev, owner=owner)


def stage_delta(
    conn: psycopg.Connection,
    *,
    overlay_id: uuid.UUID,
    source_uri: str,
    text: str = "",
    op: str = "upsert",
    block_type: str = "fact",
) -> uuid.UUID:
    """Stage an upsert or retract delta inside an open overlay without touching shared space state."""
    if op not in ("upsert", "retract"):
        raise ValueError("op must be 'upsert' or 'retract'")
    norm = canonical_text(text) if op == "upsert" else ""
    bhash = blake3_hex(norm) if norm else ""
    emb = pgvector_literal(embed_text(norm)) if norm else None
    citations = (
        [{"source_uri": source_uri, "offset": 0, "length": len(norm), "quote_hash": bhash[:16]}]
        if norm
        else []
    )
    delta_id = uuid.uuid4()
    with conn.transaction():
        conn.execute(
            """
            INSERT INTO overlay_deltas (id, overlay_id, op, source_uri, block_type, canonical_text, content_blake3, citations, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
            ON CONFLICT (overlay_id, source_uri) DO UPDATE
              SET op = EXCLUDED.op,
                  block_type = EXCLUDED.block_type,
                  canonical_text = EXCLUDED.canonical_text,
                  content_blake3 = EXCLUDED.content_blake3,
                  citations = EXCLUDED.citations,
                  embedding = EXCLUDED.embedding,
                  updated_at = now()
            """,
            (delta_id, overlay_id, op, source_uri, block_type, norm, bhash, json.dumps(citations), emb),
        )
    return delta_id


def read_mount_view(
    conn: psycopg.Connection,
    *,
    org_id: uuid.UUID,
    space_id: uuid.UUID,
    overlay_id: uuid.UUID | None = None,
) -> list[dict[str, Any]]:
    """Zero-copy Mount Stack read: `Base(space) ⊕ Overlay(delta)`.
    Base blocks keep shared `space:{id}` clearance_scope for 100% KV reuse across agents;
    only mutated overlay deltas use `overlay:{id}` scope."""
    base_rows = conn.execute(
        """
        SELECT kb.id, d.source_uri, kb.clearance_scope, kb.canonical_text, kb.content_blake3, kb.citations, kb.landed_rev
        FROM knowledge_blocks kb
        JOIN documents d ON d.id = kb.document_id
        WHERE kb.org_id = %s AND kb.space_id = %s AND kb.valid_to IS NULL
        ORDER BY kb.landed_rev DESC, kb.valid_from DESC
        """,
        (org_id, space_id),
    ).fetchall()

    deltas: dict[str, dict[str, Any]] = {}
    if overlay_id is not None:
        d_rows = conn.execute(
            """
            SELECT id, op, source_uri, canonical_text, content_blake3, citations
            FROM overlay_deltas
            WHERE overlay_id = %s
            """,
            (overlay_id,),
        ).fetchall()
        for did, op, uri, text, bhash, cits in d_rows:
            deltas[uri] = {
                "id": str(did),
                "op": op,
                "source_uri": uri,
                "clearance_scope": f"overlay:{overlay_id}",
                "canonical_text": text,
                "content_blake3": bhash,
                "citations": cits,
                "layer": "overlay",
            }

    view: list[dict[str, Any]] = []
    for bid, uri, scope, text, bhash, cits, rev in base_rows:
        if uri in deltas:
            continue
        view.append({
            "id": str(bid),
            "source_uri": uri,
            "clearance_scope": scope,
            "canonical_text": text,
            "content_blake3": bhash,
            "citations": cits,
            "landed_rev": int(rev),
            "layer": "base",
        })

    for item in deltas.values():
        if item["op"] == "upsert":
            view.append(item)
    return view


def land_overlay(
    conn: psycopg.Connection,
    *,
    overlay_id: uuid.UUID,
    author: str,
    message: str,
    force: bool = False,
) -> LandResult:
    """Server-authoritative atomic Land with optimistic concurrency control & auto-rebase for non-overlapping deltas."""
    applied_ids: list[uuid.UUID] = []
    superseded_ids: list[uuid.UUID] = []

    with conn.transaction():
        ov = conn.execute(
            "SELECT org_id, space_id, base_rev, status FROM overlays WHERE id = %s FOR UPDATE",
            (overlay_id,),
        ).fetchone()
        if ov is None or ov[3] != "open":
            raise ValueError(f"open overlay {overlay_id} not found")
        org_id, space_id, base_rev, _status = ov

        # Serialize lands per space via advisory transaction lock so rev is strictly monotonic
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"{org_id}:{space_id}",))

        deltas = conn.execute(
            """
            SELECT op, source_uri, block_type, canonical_text, content_blake3, citations, embedding::text
            FROM overlay_deltas
            WHERE overlay_id = %s
            ORDER BY source_uri ASC
            """,
            (overlay_id,),
        ).fetchall()

        # Check optimistic concurrency: did ANY touched source_uri change in a rev > base_rev?
        if not force and deltas:
            touched_uris = [d[1] for d in deltas]
            conflicts_rows = conn.execute(
                """
                SELECT d.source_uri, kb.landed_rev, kb.canonical_text
                FROM knowledge_blocks kb
                JOIN documents d ON d.id = kb.document_id
                WHERE kb.org_id = %s AND kb.space_id = %s
                  AND d.source_uri = ANY(%s)
                  AND kb.landed_rev > %s
                  AND kb.valid_to IS NULL
                """,
                (org_id, space_id, touched_uris, base_rev),
            ).fetchall()
            if conflicts_rows:
                return LandResult(
                    status="conflict",
                    rev=None,
                    applied_block_ids=[],
                    superseded_block_ids=[],
                    conflicts=[
                        {"source_uri": uri, "current_rev": int(rev), "current_text": txt}
                        for uri, rev, txt in conflicts_rows
                    ],
                )

        new_rev = current_head_rev(conn, org_id=org_id, space_id=space_id) + 1
        conn.execute(
            """
            INSERT INTO space_revisions (org_id, space_id, rev, author, message, action)
            VALUES (%s, %s, %s, %s, %s, 'land')
            """,
            (org_id, space_id, new_rev, author, message),
        )

        space_scope = f"space:{space_id}"
        for op, uri, btype, text, bhash, cits, emb_text in deltas:
            prior = conn.execute(
                """
                SELECT kb.id, kb.content_blake3, kb.clearance_scope
                FROM knowledge_blocks kb
                JOIN documents d ON d.id = kb.document_id
                WHERE kb.org_id = %s AND kb.space_id = %s AND d.source_uri = %s AND kb.valid_to IS NULL
                FOR UPDATE OF kb
                """,
                (org_id, space_id, uri),
            ).fetchall()

            new_bid: uuid.UUID | None = None
            if op == "upsert":
                doc_id = uuid.uuid4()
                new_bid = uuid.uuid4()
                conn.execute(
                    """
                    INSERT INTO documents (id, org_id, space_id, clearance_scope, source_uri, content_blake3)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (doc_id, org_id, space_id, space_scope, uri, bhash),
                )
                conn.execute(
                    """
                    INSERT INTO knowledge_blocks (
                      id, document_id, org_id, space_id, clearance_scope, landed_rev,
                      block_type, canonical_text, content_blake3, citations, embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                    """,
                    (new_bid, doc_id, org_id, space_id, space_scope, new_rev, btype, text, bhash, json.dumps(cits), emb_text),
                )
                applied_ids.append(new_bid)

            for old_id, old_hash, old_scope in prior:
                conn.execute(
                    "UPDATE knowledge_blocks SET valid_to = now(), superseded_by = %s WHERE id = %s",
                    (new_bid, old_id),
                )
                superseded_ids.append(old_id)
                _emit_eviction_outbox(conn, org_id, space_id, old_scope, old_id, old_hash, new_bid, bhash, f"land:r{new_rev}")

        conn.execute(
            "UPDATE overlays SET status = 'landed', landed_rev = %s, updated_at = now() WHERE id = %s",
            (new_rev, overlay_id),
        )

    return LandResult(
        status="landed",
        rev=new_rev,
        applied_block_ids=applied_ids,
        superseded_block_ids=superseded_ids,
        conflicts=[],
    )


def undo_revision(
    conn: psycopg.Connection,
    *,
    org_id: uuid.UUID,
    space_id: uuid.UUID,
    rev: int,
    author: str,
) -> UndoResult:
    """Atomically reverse a landed revision: deactivate blocks landed at `rev`, restore superseded blocks, evict KV."""
    deactivated: list[uuid.UUID] = []
    restored: list[uuid.UUID] = []

    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"{org_id}:{space_id}",))
        undo_rev = current_head_rev(conn, org_id=org_id, space_id=space_id) + 1
        conn.execute(
            """
            INSERT INTO space_revisions (org_id, space_id, rev, author, message, action)
            VALUES (%s, %s, %s, %s, %s, 'undo')
            """,
            (org_id, space_id, undo_rev, author, f"Undo revision r{rev}"),
        )

        bad_blocks = conn.execute(
            """
            SELECT id, content_blake3, clearance_scope
            FROM knowledge_blocks
            WHERE org_id = %s AND space_id = %s AND landed_rev = %s AND valid_to IS NULL
            FOR UPDATE
            """,
            (org_id, space_id, rev),
        ).fetchall()

        for bad_id, bad_hash, bad_scope in bad_blocks:
            conn.execute("UPDATE knowledge_blocks SET valid_to = now() WHERE id = %s", (bad_id,))
            deactivated.append(bad_id)

            prev_rows = conn.execute(
                """
                SELECT id, content_blake3
                FROM knowledge_blocks
                WHERE org_id = %s AND space_id = %s AND superseded_by = %s
                FOR UPDATE
                """,
                (org_id, space_id, bad_id),
            ).fetchall()
            for prev_id, prev_hash in prev_rows:
                conn.execute(
                    "UPDATE knowledge_blocks SET valid_to = NULL, superseded_by = NULL WHERE id = %s",
                    (prev_id,),
                )
                restored.append(prev_id)
                _emit_eviction_outbox(conn, org_id, space_id, bad_scope, bad_id, bad_hash, prev_id, prev_hash, f"undo:r{rev}")

    return UndoResult(
        undo_rev=undo_rev,
        undone_rev=rev,
        deactivated_block_ids=deactivated,
        restored_block_ids=restored,
    )


def _emit_eviction_outbox(
    conn: psycopg.Connection,
    org_id: uuid.UUID,
    space_id: uuid.UUID,
    clearance_scope: str,
    old_id: uuid.UUID,
    old_hash: str,
    new_id: uuid.UUID | None,
    new_hash: str,
    reason: str,
) -> None:
    msg_id = f"superseded:{old_id}:{new_id or 'retracted'}:{uuid.uuid4().hex[:8]}"
    payload = {
        "tenant_id": str(org_id),
        "space_id": str(space_id),
        "clearance_scope": clearance_scope,
        "old_block_id": str(old_id),
        "old_block_hash": old_hash,
        "new_block_id": str(new_id) if new_id else None,
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
