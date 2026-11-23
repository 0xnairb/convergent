"""Semantic Smart Memory: atomic claim distillation & cross-document contradiction supersession."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable
import uuid

import httpx
import psycopg

from .embedding import embed_text, pgvector_literal
from .memory import blake3_hex, canonical_text


@dataclass(frozen=True)
class ExtractedClaim:
    entity: str
    claim: str
    quote: str


@dataclass(frozen=True)
class RelationVerdict:
    relation: str  # "SUPERSEDES" | "COMPATIBLE" | "UNRELATED"
    confidence: float
    reason: str


@dataclass(frozen=True)
class SemanticIngestResult:
    document_id: uuid.UUID
    block_ids: list[uuid.UUID]
    superseded_block_ids: list[uuid.UUID]
    cross_doc_superseded_ids: list[uuid.UUID]
    quarantined_conflict_ids: list[uuid.UUID]
    claims: list[ExtractedClaim]


_AUTHORITY_PREFIXES: list[tuple[tuple[str, ...], int]] = [
    (("legal://", "policy://", "compliance://", "signed://"), 4),
    (("adr://", "memo://", "infosec://", "incident://", "postmortem://", "finance://", "eng://"), 3),
    (("handbook://", "rfc://", "sales://", "ops://", "infra://", "email://", "docs://", "wiki://"), 2),
    (("slack://", "chat://", "note://", "dm://"), 1),
]


def infer_source_authority(source_uri: str) -> int:
    """Infer governance authority tier (1=informal chat .. 4=signed legal/policy) from source URI."""
    lower = source_uri.strip().lower()
    for prefixes, level in _AUTHORITY_PREFIXES:
        if lower.startswith(prefixes):
            return level
    return 2


def ollama_json_chat(
    http: httpx.Client,
    *,
    ollama_url: str,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    """Call local Ollama /api/chat with deterministic JSON output (thinking disabled for speed)."""
    resp = http.post(
        f"{ollama_url.rstrip('/')}/api/chat",
        json={
            "model": model_id,
            "stream": False,
            "think": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 512},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
    )
    resp.raise_for_status()
    raw = resp.json()["message"]["content"]
    return json.loads(raw)


def extract_claims_llm(
    http: httpx.Client,
    *,
    ollama_url: str,
    model_id: str,
    text: str,
) -> list[ExtractedClaim]:
    """Distill a raw document into atomic, self-contained factual claims with verbatim quotes."""
    sys_prompt = (
        "You are a precise enterprise knowledge distiller. Extract atomic, self-contained factual claims "
        "from the user's document. Return ONLY JSON matching: "
        '{"claims": [{"entity": "short subject topic", "claim": "single self-contained sentence", "quote": "exact substring from input"}]}. '
        "Keep numbers, limits, dates, regions, and policy rules exact. Do not invent facts."
    )
    data = ollama_json_chat(
        http,
        ollama_url=ollama_url,
        model_id=model_id,
        system_prompt=sys_prompt,
        user_prompt=text,
    )
    claims: list[ExtractedClaim] = []
    for item in data.get("claims") or []:
        claim_text = canonical_text(str(item.get("claim") or ""))
        if not claim_text:
            continue
        claims.append(
            ExtractedClaim(
                entity=canonical_text(str(item.get("entity") or "general")),
                claim=claim_text,
                quote=canonical_text(str(item.get("quote") or claim_text)),
            )
        )
    if not claims:
        norm = canonical_text(text)
        claims.append(ExtractedClaim(entity="document", claim=norm, quote=norm[:120]))
    return claims


def classify_relationship_llm(
    http: httpx.Client,
    *,
    ollama_url: str,
    model_id: str,
    existing_claim: str,
    new_claim: str,
) -> RelationVerdict:
    """Decide whether `new_claim` SUPERSEDES (updates/contradicts) `existing_claim`, is COMPATIBLE, or UNRELATED."""
    sys_prompt = (
        "You are a strict bi-temporal knowledge auditor. Compare EXISTING_CLAIM (older) and NEW_CLAIM (newer). "
        "Return ONLY JSON: "
        '{"relation": "SUPERSEDES" | "COMPATIBLE" | "UNRELATED", "confidence": 0.0-1.0, "reason": "short explanation"}.\n'
        "Rules:\n"
        "- SUPERSEDES: NEW_CLAIM changes, updates, replaces, abolishes, or contradicts the exact rule, number, technology, region, or SLA stated in EXISTING_CLAIM.\n"
        "- COMPATIBLE: Both claims can be true at the same time (e.g. different rules, complementary details, primary vs DR region, SLA vs support channel, hotel cap vs flight class).\n"
        "- UNRELATED: Completely different subjects."
    )
    user_prompt = f"EXISTING_CLAIM (older): {existing_claim}\nNEW_CLAIM (newer): {new_claim}"
    data = ollama_json_chat(
        http,
        ollama_url=ollama_url,
        model_id=model_id,
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
    )
    rel = str(data.get("relation") or "UNRELATED").strip().upper()
    if rel not in ("SUPERSEDES", "COMPATIBLE", "UNRELATED"):
        rel = "UNRELATED"
    try:
        conf = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return RelationVerdict(relation=rel, confidence=conf, reason=str(data.get("reason") or ""))


def _find_citation(norm_doc: str, source_uri: str, quote: str) -> list[dict[str, object]]:
    idx = norm_doc.find(quote) if quote else -1
    if idx < 0:
        idx = 0
        span = norm_doc[: min(len(norm_doc), 160)]
    else:
        span = quote
    return [
        {
            "source_uri": source_uri,
            "offset": idx,
            "length": len(span),
            "quote_hash": blake3_hex(span)[:16],
        }
    ]


def ingest_document_semantic(
    conn: psycopg.Connection,
    *,
    org_id: uuid.UUID,
    space_id: uuid.UUID,
    clearance_scope: str,
    source_uri: str,
    text: str,
    extractor: Callable[[str], list[ExtractedClaim]],
    classifier: Callable[[str, str], RelationVerdict],
    source_authority: int | None = None,
    confidence_threshold: float = 0.75,
    candidate_limit: int = 8,
    block_type: str = "claim",
) -> SemanticIngestResult:
    """Distill text into atomic claims, detect cross-document contradictions within the same clearance scope,
    enforce source-authority guardrails (quarantining lower-authority overrides into `conflict_reviews`),
    and atomically supersede authorized conflicting blocks + emit durable KV eviction outbox events."""
    norm = canonical_text(text)
    doc_hash = blake3_hex(norm)
    doc_id = uuid.uuid4()
    new_auth = source_authority if source_authority is not None else infer_source_authority(source_uri)
    claims = extractor(norm)

    # Fetch candidate active blocks in the SAME org + space + clearance_scope (cross-document!)
    candidate_rows = conn.execute(
        """
        SELECT kb.id, kb.content_blake3, kb.clearance_scope, kb.canonical_text, d.source_uri, d.source_authority
        FROM knowledge_blocks kb
        JOIN documents d ON d.id = kb.document_id
        WHERE kb.org_id = %s
          AND kb.space_id = %s
          AND kb.clearance_scope = %s
          AND kb.valid_to IS NULL
        ORDER BY kb.valid_from DESC
        LIMIT 50
        """,
        (org_id, space_id, clearance_scope),
    ).fetchall()

    # Map each new claim index -> list of auto-supersessions and quarantined conflicts
    supersede_plan: dict[int, list[tuple[uuid.UUID, str, str, str, str]]] = {}
    quarantine_plan: dict[int, list[tuple[uuid.UUID, int, float, str]]] = {}
    already_claimed: set[uuid.UUID] = set()

    for idx, c in enumerate(claims):
        c_tokens = {w for w in c.claim.lower().replace(",", "").split() if len(w) > 2}
        e_tokens = {w for w in c.entity.lower().replace(",", "").split() if len(w) > 2}
        q_tokens = c_tokens | e_tokens

        scored: list[tuple[int, tuple[Any, ...]]] = []
        for row in candidate_rows:
            old_id, _old_hash, _old_scope, old_text, old_uri, _old_auth = row
            if old_id in already_claimed:
                continue
            if old_uri == source_uri:
                scored.append((1000, row))
                continue
            o_tokens = {w for w in str(old_text).lower().replace(",", "").split() if len(w) > 2}
            overlap = len(q_tokens & o_tokens)
            if overlap >= 1:
                scored.append((overlap, row))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        for _score, row in scored[:candidate_limit]:
            old_id, old_hash, old_scope, old_text, old_uri, old_auth = row
            if old_id in already_claimed:
                continue
            if old_uri == source_uri:
                supersede_plan.setdefault(idx, []).append((old_id, old_hash, old_scope, old_uri, "same_source_uri_update"))
                already_claimed.add(old_id)
                continue
            verdict = classifier(str(old_text), c.claim)
            if verdict.relation == "SUPERSEDES" and verdict.confidence >= confidence_threshold:
                if new_auth >= int(old_auth):
                    supersede_plan.setdefault(idx, []).append((old_id, old_hash, old_scope, old_uri, verdict.reason))
                else:
                    quarantine_plan.setdefault(idx, []).append((old_id, int(old_auth), verdict.confidence, verdict.reason))
                already_claimed.add(old_id)

    new_block_ids: list[uuid.UUID] = []
    superseded_ids: list[uuid.UUID] = []
    cross_doc_ids: list[uuid.UUID] = []
    quarantined_ids: list[uuid.UUID] = []

    with conn.transaction():
        conn.execute(
            """
            INSERT INTO documents (id, org_id, space_id, clearance_scope, source_uri, content_blake3, source_authority)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (doc_id, org_id, space_id, clearance_scope, source_uri, doc_hash, new_auth),
        )

        for idx, c in enumerate(claims):
            bid = uuid.uuid4()
            bhash = blake3_hex(c.claim)
            citations = _find_citation(norm, source_uri, c.quote)
            emb = pgvector_literal(embed_text(c.claim))
            is_quarantined = idx in quarantine_plan
            conn.execute(
                """
                INSERT INTO knowledge_blocks (
                  id, document_id, org_id, space_id, clearance_scope,
                  block_type, canonical_text, content_blake3, citations, embedding, valid_to
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector, CASE WHEN %s THEN now() ELSE NULL END)
                """,
                (
                    bid,
                    doc_id,
                    org_id,
                    space_id,
                    clearance_scope,
                    "quarantined_claim" if is_quarantined else block_type,
                    c.claim,
                    bhash,
                    json.dumps(citations),
                    emb,
                    is_quarantined,
                ),
            )
            new_block_ids.append(bid)

            for old_id, old_auth, conf, reason in quarantine_plan.get(idx, []):
                cid = uuid.uuid4()
                conn.execute(
                    """
                    INSERT INTO conflict_reviews (
                      id, org_id, space_id, existing_block_id, candidate_block_id,
                      existing_authority, candidate_authority, confidence, reason, status
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'pending')
                    """,
                    (cid, org_id, space_id, old_id, bid, old_auth, new_auth, conf, reason),
                )
                quarantined_ids.append(cid)

            for old_id, old_hash, old_scope, old_uri, reason in supersede_plan.get(idx, []):
                updated = conn.execute(
                    """
                    UPDATE knowledge_blocks
                    SET valid_to = now(), superseded_by = %s
                    WHERE id = %s AND valid_to IS NULL
                    """,
                    (bid, old_id),
                )
                if updated.rowcount == 0:
                    continue
                superseded_ids.append(old_id)
                if old_uri != source_uri:
                    cross_doc_ids.append(old_id)
                msg_id = f"superseded:{old_id}:{bid}"
                payload = {
                    "tenant_id": str(org_id),
                    "space_id": str(space_id),
                    "clearance_scope": old_scope,
                    "old_block_id": str(old_id),
                    "old_block_hash": old_hash,
                    "new_block_id": str(bid),
                    "new_block_hash": bhash,
                    "reason": reason,
                    "cross_document": old_uri != source_uri,
                }
                conn.execute(
                    """
                    INSERT INTO outbox_events (msg_id, subject, payload)
                    VALUES (%s, 'convergent.memory.superseded', %s::jsonb)
                    ON CONFLICT (msg_id) DO NOTHING
                    """,
                    (msg_id, json.dumps(payload)),
                )

    return SemanticIngestResult(
        document_id=doc_id,
        block_ids=new_block_ids,
        superseded_block_ids=superseded_ids,
        cross_doc_superseded_ids=cross_doc_ids,
        quarantined_conflict_ids=quarantined_ids,
        claims=claims,
    )


def resolve_conflict(
    conn: psycopg.Connection,
    *,
    conflict_id: uuid.UUID,
    decision: str,
    resolved_by: str = "reviewer",
) -> dict[str, Any]:
    """Approve or reject a quarantined lower-authority conflict review atomically."""
    if decision not in ("approve", "reject"):
        raise ValueError("decision must be 'approve' or 'reject'")

    with conn.transaction():
        row = conn.execute(
            """
            SELECT cr.org_id, cr.space_id, cr.existing_block_id, cr.candidate_block_id, cr.reason,
                   eb.content_blake3, eb.clearance_scope, cb.content_blake3
            FROM conflict_reviews cr
            JOIN knowledge_blocks eb ON eb.id = cr.existing_block_id
            JOIN knowledge_blocks cb ON cb.id = cr.candidate_block_id
            WHERE cr.id = %s AND cr.status = 'pending'
            FOR UPDATE OF cr
            """,
            (conflict_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"pending conflict {conflict_id} not found")

        org_id, space_id, old_id, new_id, reason, old_hash, old_scope, new_hash = row
        if decision == "approve":
            conn.execute(
                "UPDATE knowledge_blocks SET valid_to = NULL, block_type = 'claim' WHERE id = %s",
                (new_id,),
            )
            conn.execute(
                "UPDATE knowledge_blocks SET valid_to = now(), superseded_by = %s WHERE id = %s AND valid_to IS NULL",
                (new_id, old_id),
            )
            msg_id = f"superseded:{old_id}:{new_id}"
            payload = {
                "tenant_id": str(org_id),
                "space_id": str(space_id),
                "clearance_scope": old_scope,
                "old_block_id": str(old_id),
                "old_block_hash": old_hash,
                "new_block_id": str(new_id),
                "new_block_hash": new_hash,
                "reason": f"approved_conflict:{reason}",
                "cross_document": True,
            }
            conn.execute(
                """
                INSERT INTO outbox_events (msg_id, subject, payload)
                VALUES (%s, 'convergent.memory.superseded', %s::jsonb)
                ON CONFLICT (msg_id) DO NOTHING
                """,
                (msg_id, json.dumps(payload)),
            )
            new_status = "approved"
        else:
            new_status = "rejected"

        conn.execute(
            "UPDATE conflict_reviews SET status = %s, resolved_at = now(), resolved_by = %s WHERE id = %s",
            (new_status, resolved_by, conflict_id),
        )
    return {"conflict_id": str(conflict_id), "status": new_status, "superseded_block_id": str(old_id) if decision == "approve" else None}
