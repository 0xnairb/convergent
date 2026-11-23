"""Runnable async daemon for Project Convergent background workers."""

from __future__ import annotations

import asyncio
import logging
import os
import signal

import httpx
import nats
import psycopg
import redis

from .ingest import drain_ingest_once, resolve_openfga_store_id
from .kv_evictor import drain_supersession_once
from .outbox import relay_outbox_once

log = logging.getLogger("convergent_worker")


async def run_worker_loop(stop_event: asyncio.Event) -> None:
    pg_dsn = os.environ.get(
        "CONVERGENT_PG_DSN",
        "postgresql://convergent:convergent_dev_password@127.0.0.1:55432/convergent",
    )
    valkey_url = os.environ.get("CONVERGENT_VALKEY_URL", "redis://127.0.0.1:56379/0")
    openfga_url = os.environ.get("CONVERGENT_OPENFGA_URL", "http://127.0.0.1:58080")
    openfga_store_name = os.environ.get("CONVERGENT_OPENFGA_STORE", "convergent")
    nats_url = os.environ.get("CONVERGENT_NATS_URL", "nats://127.0.0.1:54222")
    vllm_url = os.environ.get("CONVERGENT_VLLM_URL", "http://127.0.0.1:8000")
    model_id = os.environ.get("CONVERGENT_MODEL_ID", "Qwen/Qwen3-32B-FP8")
    master_seed = os.environ.get(
        "CONVERGENT_MASTER_SEED",
        "convergent-dev-kv-master-seed-change-in-prod",
    ).encode("utf-8")
    poll_interval = float(os.environ.get("CONVERGENT_WORKER_POLL_SECS", "0.25"))

    nc = await nats.connect(nats_url)
    js = nc.jetstream()
    valkey = redis.Redis.from_url(valkey_url, decode_responses=True)

    with psycopg.connect(pg_dsn, autocommit=True) as conn, httpx.Client(timeout=15.0) as http:
        store_id = resolve_openfga_store_id(http, openfga_url, openfga_store_name)
        log.info(
            "convergent-worker started (openfga_store=%s, vllm=%s, model=%s)",
            store_id,
            vllm_url,
            model_id,
        )
        try:
            while not stop_event.is_set():
                ingested = await drain_ingest_once(
                    conn=conn,
                    js=js,
                    http=http,
                    openfga_url=openfga_url,
                    openfga_store_id=store_id,
                    valkey=valkey,
                )
                if ingested:
                    log.info("ingested %d document(s)", len(ingested))

                relayed = await relay_outbox_once(conn, js)
                if relayed:
                    log.info("relayed %d outbox event(s)", relayed)

                evicted = await drain_supersession_once(
                    conn=conn,
                    valkey=valkey,
                    js=js,
                    http=http,
                    vllm_url=vllm_url,
                    model_id=model_id,
                    master_seed=master_seed,
                )
                if evicted:
                    log.info("processed %d KV eviction/warmup event(s)", len(evicted))

                if not (ingested or relayed or evicted):
                    try:
                        await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
                    except TimeoutError:
                        pass
        finally:
            valkey.close()
            await nc.close()
            log.info("convergent-worker stopped cleanly")


def cli() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    stop_event = asyncio.Event()

    async def _main() -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except NotImplementedError:
                pass
        await run_worker_loop(stop_event)

    asyncio.run(_main())


if __name__ == "__main__":
    cli()
