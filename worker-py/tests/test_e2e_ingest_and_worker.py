"""End-to-end test: NATS ingest -> Postgres + OpenFGA -> supersession outbox -> KV evictor -> dev_vllm_server."""

from __future__ import annotations

from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import uuid

import httpx
import nats
import psycopg
import pytest
import redis

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from dev_vllm_server import make_handler  # noqa: E402

from convergent_worker.ingest import drain_ingest_once, resolve_openfga_store_id  # noqa: E402
from convergent_worker.kv_evictor import derive_kv_key, drain_supersession_once, register_kv_key  # noqa: E402
from convergent_worker.outbox import relay_outbox_once  # noqa: E402

PG_DSN = os.environ.get(
    "CONVERGENT_PG_DSN",
    "postgresql://convergent:convergent_dev_password@127.0.0.1:55432/convergent",
)
VALKEY_URL = os.environ.get("CONVERGENT_VALKEY_URL", "redis://127.0.0.1:56379/0")
OPENFGA_URL = os.environ.get("CONVERGENT_OPENFGA_URL", "http://127.0.0.1:58080")
NATS_URL = os.environ.get("CONVERGENT_NATS_URL", "nats://127.0.0.1:54222")


@pytest.mark.asyncio
async def test_e2e_ingest_openfga_supersession_and_dev_vllm() -> None:
    model_id = "Qwen/Qwen3-32B-FP8"
    master_seed = b"e2e-apple-silicon-seed"

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(VALKEY_URL, model_id))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    vllm_url = f"http://127.0.0.1:{port}"

    org_id = uuid.uuid4()
    space_id = uuid.uuid4()
    scope = f"space:{space_id}"
    source_uri = f"docs://e2e/policy-{uuid.uuid4().hex[:8]}.md"
    user_id = f"alice-{uuid.uuid4().hex[:6]}"

    nc = await nats.connect(NATS_URL)
    js = nc.jetstream()
    valkey = redis.Redis.from_url(VALKEY_URL, decode_responses=True)

    try:
        with psycopg.connect(PG_DSN) as conn, httpx.Client(timeout=10.0) as http:
            store_id = resolve_openfga_store_id(http, OPENFGA_URL, "convergent")

            # Grant user viewer on space
            http.post(
                f"{OPENFGA_URL}/stores/{store_id}/write",
                json={"writes": {"tuple_keys": [{"user": f"user:{user_id}", "relation": "viewer", "object": f"space:{space_id}"}]}},
            ).raise_for_status()

            # 1. Publish v1 ingest request to NATS
            await js.publish(
                "convergent.ingest.requested",
                json.dumps(
                    {
                        "org_id": str(org_id),
                        "space_id": str(space_id),
                        "clearance_scope": scope,
                        "source_uri": source_uri,
                        "content": "Policy v1: Legacy reimbursement cap is $500.",
                    }
                ).encode("utf-8"),
            )
            v1_results = await drain_ingest_once(
                conn=conn,
                js=js,
                http=http,
                openfga_url=OPENFGA_URL,
                openfga_store_id=store_id,
            )
            assert len(v1_results) >= 1
            v1_res = v1_results[-1]

            # Verify OpenFGA automatically granted user:alice viewer access to v1 document via space inheritance
            allowed = http.post(
                f"{OPENFGA_URL}/stores/{store_id}/list-objects",
                json={"user": f"user:{user_id}", "relation": "viewer", "type": "document"},
            ).raise_for_status().json()["objects"]
            assert f"document:{v1_res.document_id}" in allowed

            # Simulate warm KV entry for v1 block in Valkey
            v1_hash = conn.execute(
                "SELECT content_blake3 FROM knowledge_blocks WHERE id = %s", (v1_res.block_ids[0],)
            ).fetchone()[0]
            v1_kv_key = derive_kv_key(master_seed, scope, model_id, v1_hash)
            register_kv_key(valkey, v1_hash, v1_kv_key)
            assert valkey.exists(v1_kv_key) == 1

            # 2. Publish v2 ingest request superseding v1
            await js.publish(
                "convergent.ingest.requested",
                json.dumps(
                    {
                        "org_id": str(org_id),
                        "space_id": str(space_id),
                        "clearance_scope": scope,
                        "source_uri": source_uri,
                        "content": "Policy v2: Updated reimbursement cap is $1200.",
                    }
                ).encode("utf-8"),
            )
            v2_results = await drain_ingest_once(
                conn=conn,
                js=js,
                http=http,
                openfga_url=OPENFGA_URL,
                openfga_store_id=store_id,
            )
            v2_res = v2_results[-1]
            assert v1_res.block_ids[0] in v2_res.superseded_block_ids

            # 3. Relay outbox -> drain supersession -> evict v1 KV & warm v2 on dev_vllm_server
            relayed = await relay_outbox_once(conn, js)
            assert relayed >= 1
            evicted = await drain_supersession_once(
                conn=conn,
                valkey=valkey,
                js=js,
                http=http,
                vllm_url=vllm_url,
                model_id=model_id,
                master_seed=master_seed,
            )
            assert evicted and evicted[-1]["evicted_count"] == 1
            assert valkey.exists(v1_kv_key) == 0

            v2_kv_key = evicted[-1]["warmed_key"]
            assert v2_kv_key is not None
            assert valkey.exists(v2_kv_key) == 1

            # 4. Query dev_vllm_server with v2_kv_key and confirm cache hit=1
            chat_resp = http.post(
                f"{vllm_url}/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": "What is the reimbursement cap?"}],
                    "extra_body": {"kv_cache_keys": [v2_kv_key]},
                },
            )
            chat_resp.raise_for_status()
            assert chat_resp.headers["x-convergent-dev-cache-hits"] == "1"
    finally:
        server.shutdown()
        server.server_close()
        valkey.close()
        await nc.close()
