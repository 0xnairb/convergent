"""Phase 1 tests: bi-temporal ingestion, supersession, hybrid search, as_of recovery, and NATS outbox."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import time
import uuid

import nats
import psycopg

from convergent_worker.memory import hybrid_search, ingest_document
from convergent_worker.outbox import relay_outbox_once

PG_DSN = os.environ.get(
    "CONVERGENT_PG_DSN",
    "postgresql://convergent:convergent_dev_password@127.0.0.1:55432/convergent",
)
NATS_URL = os.environ.get("CONVERGENT_NATS_URL", "nats://127.0.0.1:54222")


def test_bitemporal_supersession_and_as_of_recovery() -> None:
    org_id = uuid.uuid4()
    space_id = uuid.uuid4()
    source_uri = "policy://finance/travel-cap"

    with psycopg.connect(PG_DSN) as conn:
        v1 = ingest_document(
            conn,
            org_id=org_id,
            space_id=space_id,
            clearance_scope=f"org:{org_id}:public",
            source_uri=source_uri,
            text="Travel meal budget limit is $100 per day for all engineers.",
        )
        assert len(v1.block_ids) == 1
        assert v1.superseded_block_ids == []

        # Capture timestamp between v1 and v2
        time.sleep(0.05)
        t_between = datetime.now(timezone.utc)
        time.sleep(0.05)

        v2 = ingest_document(
            conn,
            org_id=org_id,
            space_id=space_id,
            clearance_scope=f"org:{org_id}:public",
            source_uri=source_uri,
            text="Travel meal budget limit is $250 per day for all engineers.",
        )
        assert len(v2.block_ids) == 1
        assert v2.superseded_block_ids == v1.block_ids

        # Active hybrid search returns ONLY v2 with exact source citations
        active = hybrid_search(
            conn,
            org_id=org_id,
            space_id=space_id,
            query="travel meal budget limit",
            top_k=5,
        )
        assert len(active) == 1
        assert "$250" in active[0]["canonical_text"]
        assert active[0]["citations"][0]["source_uri"] == source_uri

        # Historical as_of query recovers v1 cleanly
        historical = hybrid_search(
            conn,
            org_id=org_id,
            space_id=space_id,
            query="travel meal budget limit",
            as_of=t_between,
            top_k=5,
        )
        assert len(historical) == 1
        assert "$100" in historical[0]["canonical_text"]
        assert historical[0]["superseded_by"] == str(v2.block_ids[0])


async def test_supersession_outbox_relays_durable_nats_event() -> None:
    org_id = uuid.uuid4()
    space_id = uuid.uuid4()
    scope = f"space:{space_id}:matter:alpha"
    source_uri = "contract://alpha/sla"

    with psycopg.connect(PG_DSN) as conn:
        v1 = ingest_document(
            conn,
            org_id=org_id,
            space_id=space_id,
            clearance_scope=scope,
            source_uri=source_uri,
            text="SLA response time is 24 hours.",
        )
        v2 = ingest_document(
            conn,
            org_id=org_id,
            space_id=space_id,
            clearance_scope=scope,
            source_uri=source_uri,
            text="SLA response time is 4 hours.",
        )

        nc = await nats.connect(NATS_URL)
        try:
            js = nc.jetstream()
            count = await relay_outbox_once(conn, js)
            assert count >= 1

            # Pull from NATS JetStream and verify payload matches our supersession
            sub = await js.pull_subscribe(
                "convergent.memory.superseded",
                durable="convergent-kv-evictor",
            )
            msgs = await sub.fetch(10, timeout=2.0)
            matched = None
            for msg in msgs:
                body = json.loads(msg.data.decode("utf-8"))
                await msg.ack()
                if body.get("old_block_id") == str(v1.block_ids[0]):
                    matched = body
            assert matched is not None
            assert matched["tenant_id"] == str(org_id)
            assert matched["clearance_scope"] == scope
            assert matched["old_block_hash"] == v1.content_blake3
            assert matched["new_block_hash"] == v2.content_blake3
        finally:
            await nc.close()
