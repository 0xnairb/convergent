"""Idempotent bootstrap for PostgreSQL 17 (ParadeDB), OpenFGA, and NATS JetStream."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import httpx
import nats
from nats.js.api import RetentionPolicy, StreamConfig
import psycopg

ROOT = Path(__file__).resolve().parents[1]
PG_DSN = os.environ.get(
    "CONVERGENT_PG_DSN",
    "postgresql://convergent:convergent_dev_password@127.0.0.1:55432/convergent",
)
OPENFGA_URL = os.environ.get("CONVERGENT_OPENFGA_URL", "http://127.0.0.1:58080")
NATS_URL = os.environ.get("CONVERGENT_NATS_URL", "nats://127.0.0.1:54222")


def migrate_postgres() -> None:
    sql_path = ROOT / "migrations" / "001_init.sql"
    sql = sql_path.read_text(encoding="utf-8")
    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        conn.execute(sql)


def bootstrap_openfga(store_name: str = "convergent") -> tuple[str, str]:
    model = json.loads((ROOT / "migrations" / "openfga_model.json").read_text(encoding="utf-8"))
    with httpx.Client(base_url=OPENFGA_URL, timeout=10.0) as client:
        stores = client.get("/stores").raise_for_status().json().get("stores", [])
        existing = next((s for s in stores if s.get("name") == store_name), None)
        if existing:
            store_id = existing["id"]
        else:
            store_id = client.post("/stores", json={"name": store_name}).raise_for_status().json()["id"]

        models = (
            client.get(f"/stores/{store_id}/authorization-models")
            .raise_for_status()
            .json()
            .get("authorization_models", [])
        )
        if models:
            model_id = models[0]["id"]
        else:
            model_id = (
                client.post(f"/stores/{store_id}/authorization-models", json=model)
                .raise_for_status()
                .json()["authorization_model_id"]
            )
    return store_id, model_id


async def bootstrap_nats() -> None:
    nc = await nats.connect(NATS_URL)
    try:
        js = nc.jetstream()
        cfg = StreamConfig(
            name="CONVERGENT_MEMORY",
            subjects=[
                "convergent.ingest.requested",
                "convergent.memory.superseded",
                "convergent.model.warmup",
                "convergent.acl.changed",
            ],
            retention=RetentionPolicy.WORK_QUEUE,
        )
        try:
            await js.stream_info("CONVERGENT_MEMORY")
        except Exception:
            await js.add_stream(cfg)
    finally:
        await nc.close()


def main() -> None:
    migrate_postgres()
    store_id, model_id = bootstrap_openfga()
    asyncio.run(bootstrap_nats())
    print(json.dumps({"postgres": "ok", "openfga_store_id": store_id, "openfga_model_id": model_id, "nats": "ok"}))


if __name__ == "__main__":
    main()
