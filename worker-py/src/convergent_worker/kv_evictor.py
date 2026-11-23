"""Layer 1 Smart Cache eviction consumer and zero-downtime open-weight model-swap re-warmer."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

import httpx
import psycopg
import redis


def derive_kv_key(master_seed: bytes, clearance_scope: str, model_id: str, block_blake3: str) -> str:
    """Byte-identical to gateway-rs/src/kv_keys.rs::derive_kv_key."""
    scope_secret = hmac.new(master_seed, b"cvg:scope:v1:" + clearance_scope.encode("utf-8"), hashlib.sha256).digest()
    digest = hmac.new(scope_secret, f"{model_id}:{block_blake3}".encode("utf-8"), hashlib.sha256).hexdigest()
    return f"cvg:kv:v1:{digest}"


def register_kv_key(valkey: redis.Redis, block_blake3: str, kv_key: str, payload: str = "warm", ttl_secs: int = 3600) -> None:
    """Track an active KV entry under its block index so supersession eviction is O(k)."""
    pipe = valkey.pipeline()
    pipe.set(kv_key, payload, ex=ttl_secs)
    pipe.sadd(f"cvg:kv_index:{block_blake3}", kv_key)
    pipe.execute()


def evict_block_kv(valkey: redis.Redis, old_block_hash: str) -> int:
    """Delete all tracked KV entries for a superseded block hash in Valkey."""
    idx_key = f"cvg:kv_index:{old_block_hash}"
    kv_keys = list(valkey.smembers(idx_key))
    deleted = 0
    if kv_keys:
        deleted = int(valkey.delete(*kv_keys))
    valkey.delete(idx_key)
    return deleted


def prewarm_block(
    http: httpx.Client,
    vllm_url: str,
    model_id: str,
    block_text: str,
    kv_key: str,
    valkey: redis.Redis,
    block_blake3: str,
) -> None:
    """Issue a non-generating max_tokens=1 prefill call to warm vLLM + LMCache."""
    resp = http.post(
        f"{vllm_url.rstrip('/')}/v1/completions",
        json={
            "model": model_id,
            "prompt": block_text,
            "max_tokens": 1,
            "temperature": 0.0,
            "extra_body": {"kv_cache_key": kv_key},
        },
    )
    resp.raise_for_status()
    register_kv_key(valkey, block_blake3, kv_key)


def handle_supersession_event(
    *,
    conn: psycopg.Connection,
    valkey: redis.Redis,
    http: httpx.Client,
    vllm_url: str,
    model_id: str,
    master_seed: bytes,
    event: dict[str, Any],
) -> dict[str, Any]:
    """Process one `convergent.memory.superseded` event: evict stale KV and pre-warm replacement."""
    old_hash = event["old_block_hash"]
    new_hash = event["new_block_hash"]
    new_id = event["new_block_id"]
    scope = event["clearance_scope"]

    evicted = evict_block_kv(valkey, old_hash)

    row = conn.execute(
        "SELECT canonical_text FROM knowledge_blocks WHERE id = %s",
        (new_id,),
    ).fetchone()
    warmed_key: str | None = None
    if row is not None:
        warmed_key = derive_kv_key(master_seed, scope, model_id, new_hash)
        prewarm_block(http, vllm_url, model_id, row[0], warmed_key, valkey, new_hash)

    return {"evicted_count": evicted, "warmed_key": warmed_key}


def rewarm_model_and_cutover(
    *,
    conn: psycopg.Connection,
    valkey: redis.Redis,
    http: httpx.Client,
    new_model_id: str,
    new_endpoint_url: str,
    master_seed: bytes,
    top_n: int = 50,
) -> int:
    """Zero-downtime model swap: pre-warm top-heat active blocks on new model, then atomically cut over."""
    hot_blocks = conn.execute(
        """
        SELECT content_blake3, clearance_scope, canonical_text
        FROM knowledge_blocks
        WHERE valid_to IS NULL
        ORDER BY heat_score DESC, valid_from DESC
        LIMIT %s
        """,
        (top_n,),
    ).fetchall()

    warmed = 0
    for bhash, scope, text in hot_blocks:
        kv_key = derive_kv_key(master_seed, scope, new_model_id, bhash)
        prewarm_block(http, new_endpoint_url, new_model_id, text, kv_key, valkey, bhash)
        warmed += 1

    with conn.transaction():
        conn.execute(
            """
            INSERT INTO model_registry (model_id, endpoint_url, is_active, warmed_at)
            VALUES (%s, %s, false, now())
            ON CONFLICT (model_id) DO UPDATE
              SET endpoint_url = EXCLUDED.endpoint_url, warmed_at = now()
            """,
            (new_model_id, new_endpoint_url),
        )
        conn.execute("UPDATE model_registry SET is_active = false WHERE model_id <> %s", (new_model_id,))
        conn.execute("UPDATE model_registry SET is_active = true, warmed_at = now() WHERE model_id = %s", (new_model_id,))

    return warmed


async def drain_supersession_once(
    *,
    conn: psycopg.Connection,
    valkey: redis.Redis,
    js: Any,
    http: httpx.Client,
    vllm_url: str,
    model_id: str,
    master_seed: bytes,
    durable_name: str = "convergent-kv-evictor",
    batch_size: int = 10,
) -> list[dict[str, Any]]:
    """Consume pending supersession events from NATS JetStream and ack only after Valkey DEL."""
    sub = await js.pull_subscribe("convergent.memory.superseded", durable=durable_name)
    results: list[dict[str, Any]] = []
    try:
        msgs = await sub.fetch(batch_size, timeout=1.0)
    except TimeoutError:
        return results
    for msg in msgs:
        event = json.loads(msg.data.decode("utf-8"))
        res = handle_supersession_event(
            conn=conn,
            valkey=valkey,
            http=http,
            vllm_url=vllm_url,
            model_id=model_id,
            master_seed=master_seed,
            event=event,
        )
        await msg.ack()
        results.append(res)
    return results
