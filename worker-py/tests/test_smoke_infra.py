"""Phase 0.1 smoke test: verify Postgres 17 (pgvector+pg_search), Valkey, OpenFGA, and NATS JetStream."""

from __future__ import annotations

import os
import uuid

import httpx
import nats
import psycopg
import redis


PG_DSN = os.environ.get(
    "CONVERGENT_PG_DSN",
    "postgresql://convergent:convergent_dev_password@127.0.0.1:55432/convergent",
)
VALKEY_URL = os.environ.get("CONVERGENT_VALKEY_URL", "redis://127.0.0.1:56379/0")
OPENFGA_URL = os.environ.get("CONVERGENT_OPENFGA_URL", "http://127.0.0.1:58080")
NATS_URL = os.environ.get("CONVERGENT_NATS_URL", "nats://127.0.0.1:54222")


def test_postgres_extensions_and_tables() -> None:
    with psycopg.connect(PG_DSN) as conn:
        exts = {
            row[0]
            for row in conn.execute("SELECT extname FROM pg_extension WHERE extname IN ('vector', 'pg_search')")
        }
        assert exts == {"vector", "pg_search"}
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            )
        }
        assert {"documents", "knowledge_blocks", "outbox_events", "model_registry"}.issubset(tables)


def test_valkey_roundtrip() -> None:
    r = redis.from_url(VALKEY_URL, decode_responses=True)
    key = f"cvg:smoke:{uuid.uuid4()}"
    r.set(key, "ok", ex=30)
    assert r.get(key) == "ok"
    r.delete(key)


def test_openfga_store_ready() -> None:
    resp = httpx.get(f"{OPENFGA_URL}/stores", timeout=5.0)
    resp.raise_for_status()
    names = [s["name"] for s in resp.json().get("stores", [])]
    assert "convergent" in names


async def test_nats_jetstream_roundtrip() -> None:
    nc = await nats.connect(NATS_URL)
    try:
        js = nc.jetstream()
        info = await js.stream_info("CONVERGENT_MEMORY")
        assert info.config.name == "CONVERGENT_MEMORY"
    finally:
        await nc.close()
