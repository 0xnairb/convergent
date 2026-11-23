"""Layer 3 Smart Memory: bi-temporal knowledge blocks, hybrid search, and transactional outbox."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import unicodedata
import uuid

import blake3
import psycopg

from .embedding import embed_text, pgvector_literal


@dataclass(frozen=True)
class IngestResult:
    document_id: uuid.UUID
    block_ids: list[uuid.UUID]
    superseded_block_ids: list[uuid.UUID]
    content_blake3: str


def canonical_text(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def blake3_hex(text: str) -> str:
    return blake3.blake3(text.encode("utf-8")).hexdigest()


def split_blocks(text: str, source_uri: str) -> list[tuple[str, list[dict[str, object]]]]:
    """Split text into non-empty paragraph blocks with exact character-offset citations."""
    norm = canonical_text(text)
    blocks: list[tuple[str, list[dict[str, object]]]] = []
    offset = 0
    for raw in norm.split("\n\n"):
        part = raw.strip()
        if not part:
            offset += len(raw) + 2
            continue
        start = norm.find(part, offset)
        citation = {
            "source_uri": source_uri,
            "offset": start,
            "length": len(part),
            "quote_hash": blake3_hex(part)[:16],
        }
        blocks.append((part, [citation]))
        offset = start + len(part)
    return blocks or [(norm, [{"source_uri": source_uri, "offset": 0, "length": len(norm), "quote_hash": blake3_hex(norm)[:16]}])]


def ingest_document(
    conn: psycopg.Connection,
    *,
    org_id: uuid.UUID,
    space_id: uuid.UUID,
    clearance_scope: str,
    source_uri: str,
    text: str,
    block_type: str = "fact",
) -> IngestResult:
    """Atomically ingest a document, supersede prior active blocks for the same source_uri, and emit outbox events."""
    norm = canonical_text(text)
    doc_hash = blake3_hex(norm)
    doc_id = uuid.uuid4()
    parsed = split_blocks(norm, source_uri)

    with conn.transaction():
        # Find prior active blocks for this source_uri in the same org+space
        prior_rows = conn.execute(
            """
            SELECT kb.id, kb.content_blake3, kb.clearance_scope
            FROM knowledge_blocks kb
            JOIN documents d ON d.id = kb.document_id
            WHERE kb.org_id = %s
              AND kb.space_id = %s
              AND d.source_uri = %s
              AND kb.valid_to IS NULL
            FOR UPDATE OF kb
            """,
            (org_id, space_id, source_uri),
        ).fetchall()

        conn.execute(
            """
            INSERT INTO documents (id, org_id, space_id, clearance_scope, source_uri, content_blake3)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (doc_id, org_id, space_id, clearance_scope, source_uri, doc_hash),
        )

        new_block_ids: list[uuid.UUID] = []
        new_hashes: list[str] = []
        for part, citations in parsed:
            bid = uuid.uuid4()
            bhash = blake3_hex(part)
            emb = pgvector_literal(embed_text(part))
            conn.execute(
                """
                INSERT INTO knowledge_blocks (
                  id, document_id, org_id, space_id, clearance_scope,
                  block_type, canonical_text, content_blake3, citations, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                """,
                (
                    bid,
                    doc_id,
                    org_id,
                    space_id,
                    clearance_scope,
                    block_type,
                    part,
                    bhash,
                    json.dumps(citations),
                    emb,
                ),
            )
            new_block_ids.append(bid)
            new_hashes.append(bhash)

        superseded_ids: list[uuid.UUID] = []
        primary_new_id = new_block_ids[0]
        primary_new_hash = new_hashes[0]

        for old_id, old_hash, old_scope in prior_rows:
            conn.execute(
                """
                UPDATE knowledge_blocks
                SET valid_to = now(), superseded_by = %s
                WHERE id = %s
                """,
                (primary_new_id, old_id),
            )
            superseded_ids.append(old_id)
            msg_id = f"superseded:{old_id}:{primary_new_id}"
            payload = {
                "tenant_id": str(org_id),
                "space_id": str(space_id),
                "clearance_scope": old_scope,
                "old_block_id": str(old_id),
                "old_block_hash": old_hash,
                "new_block_id": str(primary_new_id),
                "new_block_hash": primary_new_hash,
            }
            conn.execute(
                """
                INSERT INTO outbox_events (msg_id, subject, payload)
                VALUES (%s, 'convergent.memory.superseded', %s::jsonb)
                ON CONFLICT (msg_id) DO NOTHING
                """,
                (msg_id, json.dumps(payload)),
            )

    return IngestResult(
        document_id=doc_id,
        block_ids=new_block_ids,
        superseded_block_ids=superseded_ids,
        content_blake3=doc_hash,
    )


def hybrid_search(
    conn: psycopg.Connection,
    *,
    org_id: uuid.UUID,
    space_id: uuid.UUID,
    query: str,
    allowed_doc_ids: list[uuid.UUID] | None = None,
    as_of: datetime | None = None,
    top_k: int = 5,
) -> list[dict[str, object]]:
    """Hybrid pgvector cosine + lexical rank search with strict active or bi-temporal as_of filtering."""
    q_vec = pgvector_literal(embed_text(query))
    params: list[object] = [q_vec, query, org_id, space_id]
    filters = ["org_id = %s", "space_id = %s"]

    if allowed_doc_ids is not None:
        if not allowed_doc_ids:
            return []
        filters.append("document_id = ANY(%s)")
        params.append(allowed_doc_ids)

    if as_of is None:
        filters.append("valid_to IS NULL")
    else:
        filters.append("valid_from <= %s AND (valid_to IS NULL OR valid_to > %s)")
        params.extend([as_of, as_of])

    params.append(top_k)
    where_sql = " AND ".join(filters)

    rows = conn.execute(
        f"""
        SELECT
          id,
          document_id,
          clearance_scope,
          canonical_text,
          content_blake3,
          citations,
          valid_from,
          valid_to,
          superseded_by,
          (
            0.65 * (1.0 - (embedding <=> %s::vector))
            + 0.35 * ts_rank_cd(to_tsvector('english', canonical_text), plainto_tsquery('english', %s))
          ) AS score
        FROM knowledge_blocks
        WHERE {where_sql}
        ORDER BY score DESC, valid_from DESC
        LIMIT %s
        """,
        params,
    ).fetchall()

    return [
        {
            "id": str(r[0]),
            "document_id": str(r[1]),
            "clearance_scope": r[2],
            "canonical_text": r[3],
            "content_blake3": r[4],
            "citations": r[5],
            "valid_from": r[6],
            "valid_to": r[7],
            "superseded_by": str(r[8]) if r[8] else None,
            "score": float(r[9]),
        }
        for r in rows
    ]
