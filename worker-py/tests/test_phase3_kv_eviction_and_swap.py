"""Phase 3 tests: atomic Valkey DEL supersession eviction, max_tokens=1 pre-warm, and zero-downtime model swap."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import threading
import uuid

import httpx
import nats
import psycopg
import redis

from convergent_worker.kv_evictor import (
    derive_kv_key,
    drain_supersession_once,
    register_kv_key,
    rewarm_model_and_cutover,
)
from convergent_worker.memory import ingest_document
from convergent_worker.outbox import relay_outbox_once

PG_DSN = os.environ.get(
    "CONVERGENT_PG_DSN",
    "postgresql://convergent:convergent_dev_password@127.0.0.1:55432/convergent",
)
VALKEY_URL = os.environ.get("CONVERGENT_VALKEY_URL", "redis://127.0.0.1:56379/0")
NATS_URL = os.environ.get("CONVERGENT_NATS_URL", "nats://127.0.0.1:54222")


def _start_mock_vllm() -> tuple[HTTPServer, list[dict]]:
    recorded: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            recorded.append(body)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"id": "cmpl-warm", "choices": [{"text": ""}]}).encode("utf-8"))

        def log_message(self, format, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, recorded


async def test_supersession_evicts_stale_valkey_kv_and_prewarms_new() -> None:
    server, recorded = _start_mock_vllm()
    vllm_url = f"http://127.0.0.1:{server.server_port}"
    master_seed = b"phase3-secret-seed"
    model_id = "qwen-2.5-72b"
    org_id = uuid.uuid4()
    space_id = uuid.uuid4()
    scope = f"org:{org_id}:public"

    valkey = redis.from_url(VALKEY_URL, decode_responses=True)
    try:
        with psycopg.connect(PG_DSN) as conn:
            v1 = ingest_document(
                conn,
                org_id=org_id,
                space_id=space_id,
                clearance_scope=scope,
                source_uri="policy://budget/limit",
                text="Budget approval threshold is $10,000.",
            )
            old_kv_key = derive_kv_key(master_seed, scope, model_id, v1.content_blake3)
            register_kv_key(valkey, v1.content_blake3, old_kv_key)
            assert valkey.get(old_kv_key) == "warm"

            v2 = ingest_document(
                conn,
                org_id=org_id,
                space_id=space_id,
                clearance_scope=scope,
                source_uri="policy://budget/limit",
                text="Budget approval threshold is $25,000.",
            )

            nc = await nats.connect(NATS_URL)
            try:
                js = nc.jetstream()
                await relay_outbox_once(conn, js)
                with httpx.Client(timeout=5.0) as http:
                    results = await drain_supersession_once(
                        conn=conn,
                        valkey=valkey,
                        js=js,
                        http=http,
                        vllm_url=vllm_url,
                        model_id=model_id,
                        master_seed=master_seed,
                        durable_name="convergent-kv-evictor",
                    )
                assert len(results) >= 1
            finally:
                await nc.close()

            # Stale v1 KV key MUST be deleted from Valkey; v2 MUST be pre-warmed via max_tokens=1
            assert valkey.get(old_kv_key) is None
            new_kv_key = derive_kv_key(master_seed, scope, model_id, v2.content_blake3)
            assert valkey.get(new_kv_key) == "warm"
            assert any(r["max_tokens"] == 1 and "$25,000" in r["prompt"] for r in recorded)
    finally:
        server.shutdown()


def test_zero_downtime_model_swap_rewarms_hot_blocks_and_cuts_over() -> None:
    server, recorded = _start_mock_vllm()
    new_endpoint = f"http://127.0.0.1:{server.server_port}"
    master_seed = b"phase3-secret-seed"
    new_model = f"deepseek-v3-{uuid.uuid4().hex[:6]}"
    valkey = redis.from_url(VALKEY_URL, decode_responses=True)

    try:
        with psycopg.connect(PG_DSN) as conn:
            with httpx.Client(timeout=5.0) as http:
                warmed = rewarm_model_and_cutover(
                    conn=conn,
                    valkey=valkey,
                    http=http,
                    new_model_id=new_model,
                    new_endpoint_url=new_endpoint,
                    master_seed=master_seed,
                    top_n=5,
                )
            assert warmed >= 1
            assert len(recorded) == warmed
            assert all(r["model"] == new_model and r["max_tokens"] == 1 for r in recorded)

            active = conn.execute(
                "SELECT model_id, endpoint_url FROM model_registry WHERE is_active = true"
            ).fetchall()
            assert active == [(new_model, new_endpoint)]
    finally:
        server.shutdown()
