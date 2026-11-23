"""Tests for Semantic Smart Memory: cross-document contradiction supersession & false-positive safety."""

from __future__ import annotations

import os
import uuid

import psycopg

from convergent_worker.memory import hybrid_search
from convergent_worker.semantic import ExtractedClaim, RelationVerdict, ingest_document_semantic

PG_DSN = os.environ.get(
    "CONVERGENT_PG_DSN",
    "postgresql://convergent:convergent_dev_password@127.0.0.1:55432/convergent",
)


def _deterministic_extractor(text: str) -> list[ExtractedClaim]:
    return [ExtractedClaim(entity="policy", claim=text.strip(), quote=text.strip()[:80])]


def _deterministic_classifier(existing: str, new: str) -> RelationVerdict:
    if "$500" in existing and "$1,200" in new:
        return RelationVerdict("SUPERSEDES", 0.96, "reimbursement cap updated from $500 to $1,200")
    if "flight" in new.lower() or "economy" in new.lower():
        return RelationVerdict("COMPATIBLE", 0.94, "flight class rule is additive to hotel/meal cap")
    return RelationVerdict("UNRELATED", 0.90, "different topic")


def test_cross_document_contradiction_supersedes_and_preserves_additive_and_space_isolation() -> None:
    org_id = uuid.uuid4()
    space_finance = uuid.uuid4()
    space_other = uuid.uuid4()

    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        # 1. Ingest old PDF policy in space_finance AND identical text in space_other
        r_pdf = ingest_document_semantic(
            conn,
            org_id=org_id,
            space_id=space_finance,
            clearance_scope=f"space:{space_finance}",
            source_uri="policy://travel-2025.pdf",
            text="Corporate travel reimbursement cap is $500 per trip.",
            extractor=_deterministic_extractor,
            classifier=_deterministic_classifier,
        )
        r_other = ingest_document_semantic(
            conn,
            org_id=org_id,
            space_id=space_other,
            clearance_scope=f"space:{space_other}",
            source_uri="policy://other-space-travel.pdf",
            text="Corporate travel reimbursement cap is $500 per trip.",
            extractor=_deterministic_extractor,
            classifier=_deterministic_classifier,
        )

        # 2. Ingest additive flight rule from a different URI -> MUST NOT supersede travel cap!
        r_flight = ingest_document_semantic(
            conn,
            org_id=org_id,
            space_id=space_finance,
            clearance_scope=f"space:{space_finance}",
            source_uri="policy://flight-booking.md",
            text="Corporate travel flights under 6 hours must be booked in economy class.",
            extractor=_deterministic_extractor,
            classifier=_deterministic_classifier,
        )
        assert r_flight.superseded_block_ids == []

        # 3. Ingest CFO memo from a COMPLETELY DIFFERENT source_uri contradicting the $500 cap
        r_memo = ingest_document_semantic(
            conn,
            org_id=org_id,
            space_id=space_finance,
            clearance_scope=f"space:{space_finance}",
            source_uri="memo://cfo-q4-update.md",
            text="Effective Q4, corporate travel reimbursement cap is $1,200 per trip.",
            extractor=_deterministic_extractor,
            classifier=_deterministic_classifier,
        )
        assert r_pdf.block_ids[0] in r_memo.cross_doc_superseded_ids
        assert r_flight.block_ids[0] not in r_memo.superseded_block_ids
        assert r_other.block_ids[0] not in r_memo.superseded_block_ids

        # 4. Active search in space_finance returns $1,200 + flight rule, NEVER $500
        active_finance = hybrid_search(
            conn, org_id=org_id, space_id=space_finance, query="corporate travel reimbursement", top_k=10
        )
        texts = [str(b["canonical_text"]) for b in active_finance]
        assert any("$1,200" in t for t in texts)
        assert any("economy class" in t for t in texts)
        assert not any("$500" in t for t in texts)

        # 5. space_other MUST remain untouched (cross-space isolation)
        active_other = hybrid_search(
            conn, org_id=org_id, space_id=space_other, query="corporate travel reimbursement", top_k=5
        )
        assert any("$500" in str(b["canonical_text"]) for b in active_other)

        # 6. Outbox event recorded cross_document=True for KV eviction
        outbox = conn.execute(
            "SELECT payload FROM outbox_events WHERE msg_id = %s",
            (f"superseded:{r_pdf.block_ids[0]}:{r_memo.block_ids[0]}",),
        ).fetchone()
        assert outbox is not None
        assert outbox[0]["cross_document"] is True
