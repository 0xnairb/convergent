"""Layer 3 ingestion consumer: processes `convergent.ingest.requested` and registers OpenFGA tuples."""

from __future__ import annotations

import json
from typing import Any
import uuid

import httpx
import psycopg
import redis

from .memory import IngestResult, ingest_document


def resolve_openfga_store_id(http: httpx.Client, openfga_url: str, store_name: str = "convergent") -> str:
    resp = http.get(f"{openfga_url.rstrip('/')}/stores")
    resp.raise_for_status()
    for store in resp.json().get("stores", []):
        if store.get("name") == store_name:
            return str(store["id"])
    raise RuntimeError(f"OpenFGA store '{store_name}' not found")


def register_document_openfga(
    http: httpx.Client,
    openfga_url: str,
    store_id: str,
    *,
    document_id: uuid.UUID,
    space_id: uuid.UUID,
    org_id: uuid.UUID,
) -> None:
    """Link document -> space (parent) and space -> org so space readers inherit document access."""
    tuples = [
        {"user": f"space:{space_id}", "relation": "space", "object": f"document:{document_id}"},
        {"user": f"org:{org_id}", "relation": "org", "object": f"space:{space_id}"},
    ]
    for t in tuples:
        resp = http.post(
            f"{openfga_url.rstrip('/')}/stores/{store_id}/write",
            json={"writes": {"tuple_keys": [t]}},
        )
        # OpenFGA returns 400 if an identical tuple already exists (e.g. space->org on 2nd doc).
        if resp.status_code == 400 and "already exists" in resp.text.lower():
            continue
        resp.raise_for_status()


def invalidate_acl_caches(valkey: redis.Redis | None, store_id: str) -> None:
    """Clear short-TTL Valkey ACL caches after new document tuples are written."""
    if valkey is None:
        return
    keys = list(valkey.scan_iter(match=f"cvg:acl:{store_id}:*"))
    if keys:
        valkey.delete(*keys)


def process_ingest_payload(
    *,
    conn: psycopg.Connection,
    http: httpx.Client,
    openfga_url: str,
    openfga_store_id: str,
    payload: dict[str, Any],
    valkey: redis.Redis | None = None,
) -> IngestResult:
    org_id = uuid.UUID(str(payload["org_id"]))
    space_id = uuid.UUID(str(payload["space_id"]))
    source_uri = str(payload["source_uri"])
    content = str(payload["content"])
    clearance_scope = str(payload.get("clearance_scope") or f"space:{space_id}")
    block_type = str(payload.get("block_type") or "fact")

    result = ingest_document(
        conn,
        org_id=org_id,
        space_id=space_id,
        clearance_scope=clearance_scope,
        source_uri=source_uri,
        text=content,
        block_type=block_type,
    )
    register_document_openfga(
        http,
        openfga_url,
        openfga_store_id,
        document_id=result.document_id,
        space_id=space_id,
        org_id=org_id,
    )
    invalidate_acl_caches(valkey, openfga_store_id)
    return result


async def drain_ingest_once(
    *,
    conn: psycopg.Connection,
    js: Any,
    http: httpx.Client,
    openfga_url: str,
    openfga_store_id: str,
    valkey: redis.Redis | None = None,
    durable_name: str = "convergent-ingest-worker",
    batch_size: int = 10,
    fetch_timeout: float = 0.25,
) -> list[IngestResult]:
    """Consume pending `convergent.ingest.requested` messages, persist blocks, and register OpenFGA tuples."""
    sub = await js.pull_subscribe("convergent.ingest.requested", durable=durable_name)
    results: list[IngestResult] = []
    try:
        msgs = await sub.fetch(batch_size, timeout=fetch_timeout)
    except TimeoutError:
        return results
    for msg in msgs:
        payload = json.loads(msg.data.decode("utf-8"))
        res = process_ingest_payload(
            conn=conn,
            http=http,
            openfga_url=openfga_url,
            openfga_store_id=openfga_store_id,
            payload=payload,
            valkey=valkey,
        )
        await msg.ack()
        results.append(res)
    return results
