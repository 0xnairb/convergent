"""Standalone live sample & proof runner for Project Convergent's 3 Layers:
  1. Smart Memory (Layer 3): Bi-temporal supersession, citations, & as_of recovery
  2. Smart Private Context (Layer 2): OpenFGA ReBAC isolation, HMAC-SHA256 KV disjointness, instant revocation
  3. Smart Cache (Layer 1): KV cache hits, atomic supersession eviction + prefill warmup, zero-downtime model swap

Runs completely locally on macOS Apple Silicon against live Postgres 17, Valkey 8,
OpenFGA 1.8.5, NATS JetStream, the real Rust `convergent-gateway` binary, the real
Python `convergent-worker` loop, and `scripts/dev_vllm_server.py`.

Usage:
    cd convergent/worker-py && uv run python ../examples/demo_three_layers.py
"""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timezone
import hashlib
from http.server import ThreadingHTTPServer
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import quote
import uuid

import httpx
import psycopg
import redis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "worker-py" / "src"))

from dev_vllm_server import make_handler  # noqa: E402
from convergent_worker.kv_evictor import derive_kv_key, rewarm_model_and_cutover  # noqa: E402
from convergent_worker.main import run_worker_loop  # noqa: E402

PG_DSN = os.environ.get(
    "CONVERGENT_PG_DSN",
    "postgresql://convergent:convergent_dev_password@127.0.0.1:55432/convergent",
)
VALKEY_URL = os.environ.get("CONVERGENT_VALKEY_URL", "redis://127.0.0.1:56379/0")
OPENFGA_URL = os.environ.get("CONVERGENT_OPENFGA_URL", "http://127.0.0.1:58080")
NATS_URL = os.environ.get("CONVERGENT_NATS_URL", "nats://127.0.0.1:54222")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _pkce_pair() -> tuple[str, str]:
    verifier = f"cvg_verifier_{uuid.uuid4().hex}{uuid.uuid4().hex}"
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _oauth_login(http: httpx.Client, gateway_url: str, sub: str, org_id: uuid.UUID) -> str:
    verifier, challenge = _pkce_pair()
    auth = http.post(
        f"{gateway_url}/oauth/authorize",
        json={"sub": sub, "org_id": str(org_id), "scope": "mcp:read mcp:write", "code_challenge": challenge},
    )
    auth.raise_for_status()
    code = auth.json()["code"]
    tok = http.post(f"{gateway_url}/oauth/token", json={"code": code, "code_verifier": verifier})
    tok.raise_for_status()
    return str(tok.json()["access_token"])


def _mcp_call(http: httpx.Client, gateway_url: str, token: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    resp = http.post(
        f"{gateway_url}/mcp",
        headers={"Authorization": f"Bearer {token}"},
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool, "arguments": arguments}},
    )
    resp.raise_for_status()
    body = resp.json()
    if "error" in body:
        raise RuntimeError(f"MCP error on {tool}: {body['error']}")
    return body["result"]


def _wait_until(predicate: Any, timeout_secs: float = 8.0, interval: float = 0.15) -> Any:
    deadline = time.monotonic() + timeout_secs
    last = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval)
    raise TimeoutError(f"Condition not met within {timeout_secs}s (last={last!r})")


def _ensure_gateway_binary() -> Path:
    bin_path = ROOT / "gateway-rs" / "target" / "debug" / "convergent-gateway"
    if not bin_path.exists():
        subprocess.run(["cargo", "build", "--bin", "convergent-gateway"], cwd=ROOT / "gateway-rs", check=True)
    return bin_path


def run_three_layer_proof(verbose: bool = True) -> dict[str, Any]:
    """Boot the full local stack (dev vLLM + Rust gateway + Python worker) and prove all 3 layers."""
    model_v1 = "Qwen/Qwen3-32B-FP8"
    model_v2 = "DeepSeek/DeepSeek-V3.2-Local"
    master_seed_str = f"demo-seed-{uuid.uuid4().hex[:8]}"
    master_seed = master_seed_str.encode("utf-8")
    jwt_secret = f"demo-jwt-{uuid.uuid4().hex[:8]}"

    # 1. Start local dev vLLM simulator (tracks KV cache keys in Valkey)
    vllm_server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(VALKEY_URL, model_v1))
    vllm_port = vllm_server.server_address[1]
    vllm_url = f"http://127.0.0.1:{vllm_port}"
    threading.Thread(target=vllm_server.serve_forever, daemon=True).start()

    # Ensure model_registry starts clean pointing to our ephemeral vLLM simulator
    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        conn.execute("UPDATE model_registry SET is_active = false")
        conn.execute(
            """
            INSERT INTO model_registry (model_id, endpoint_url, is_active, warmed_at)
            VALUES (%s, %s, true, now())
            ON CONFLICT (model_id) DO UPDATE
              SET endpoint_url = EXCLUDED.endpoint_url, is_active = true, warmed_at = now()
            """,
            (model_v1, vllm_url),
        )

    # 2. Start real Rust convergent-gateway binary
    gw_port = _free_port()
    gateway_url = f"http://127.0.0.1:{gw_port}"
    gw_bin = _ensure_gateway_binary()
    gw_env = {
        **os.environ,
        "CONVERGENT_PG_DSN": PG_DSN,
        "CONVERGENT_VALKEY_URL": VALKEY_URL,
        "CONVERGENT_OPENFGA_URL": OPENFGA_URL,
        "CONVERGENT_NATS_URL": NATS_URL,
        "CONVERGENT_BIND_ADDR": f"127.0.0.1:{gw_port}",
        "CONVERGENT_JWT_SECRET": jwt_secret,
        "CONVERGENT_MASTER_SEED": master_seed_str,
        "CONVERGENT_VLLM_URL": vllm_url,
        "CONVERGENT_MODEL_ID": model_v1,
        "CONVERGENT_INFERENCE_BACKEND": "dev",
    }
    gw_proc = subprocess.Popen([str(gw_bin)], env=gw_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 3. Start real Python convergent-worker async loop in background thread
    worker_stop = asyncio.Event()
    worker_env_backup = {
        k: os.environ.get(k)
        for k in ("CONVERGENT_VLLM_URL", "CONVERGENT_MODEL_ID", "CONVERGENT_MASTER_SEED", "CONVERGENT_WORKER_POLL_SECS")
    }
    os.environ["CONVERGENT_VLLM_URL"] = vllm_url
    os.environ["CONVERGENT_MODEL_ID"] = model_v1
    os.environ["CONVERGENT_MASTER_SEED"] = master_seed_str
    os.environ["CONVERGENT_WORKER_POLL_SECS"] = "0.1"

    worker_loop = asyncio.new_event_loop()

    def _run_worker() -> None:
        asyncio.set_event_loop(worker_loop)
        worker_loop.run_until_complete(run_worker_loop(worker_stop))

    worker_thread = threading.Thread(target=_run_worker, daemon=True)
    worker_thread.start()

    valkey = redis.Redis.from_url(VALKEY_URL, decode_responses=True)
    report: dict[str, Any] = {}

    try:
        with httpx.Client(timeout=15.0) as http:
            # Wait for gateway /healthz
            _wait_until(lambda: _is_healthy(http, gateway_url))

            org_id = uuid.uuid4()
            space_finance = uuid.uuid4()
            space_eng = uuid.uuid4()
            suffix = uuid.uuid4().hex[:6]
            alice, carol, bob = f"alice-{suffix}", f"carol-{suffix}", f"bob-{suffix}"

            tok_alice = _oauth_login(http, gateway_url, alice, org_id)
            tok_carol = _oauth_login(http, gateway_url, carol, org_id)
            tok_bob = _oauth_login(http, gateway_url, bob, org_id)

            # Grant space memberships via gateway HTTP API
            for sid, user in [(space_finance, alice), (space_finance, carol), (space_eng, bob)]:
                http.post(
                    f"{gateway_url}/v1/spaces/{sid}/grants",
                    headers={"Authorization": f"Bearer {tok_alice}"},
                    json={"user_id": user, "relation": "viewer", "action": "grant"},
                ).raise_for_status()

            # =========================================================================
            # LAYER 3 · SMART MEMORY: Bi-temporal ingest, citations, supersession, as_of
            # =========================================================================
            policy_uri = f"policy://travel-policy-{suffix}.md"
            _mcp_call(
                http,
                gateway_url,
                tok_alice,
                "ingest_note",
                {
                    "space_id": str(space_finance),
                    "source_uri": policy_uri,
                    "content": "Corporate travel reimbursement cap is $500 per trip.",
                },
            )

            v1_blocks = _wait_until(
                lambda: [
                    b
                    for b in _mcp_call(
                        http,
                        gateway_url,
                        tok_alice,
                        "search_memory",
                        {"query": "travel reimbursement cap", "space_id": str(space_finance)},
                    )["blocks"]
                    if "$500" in b["canonical_text"]
                ]
            )
            v1_block = v1_blocks[0]
            t_v1 = datetime.now(timezone.utc)
            time.sleep(0.15)

            # Warm v1 KV cache via /v1/chat/completions (Layer 1 cold -> warm)
            cold_resp = http.post(
                f"{gateway_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {tok_alice}", "X-Convergent-Space": str(space_finance)},
                json={"model": model_v1, "messages": [{"role": "user", "content": "What is the travel reimbursement cap?"}]},
            )
            cold_resp.raise_for_status()
            v1_kv_key = derive_kv_key(master_seed, f"space:{space_finance}", model_v1, v1_block["content_blake3"])
            # Register index entry as LMCache/evictor does so supersession DELs it
            valkey.sadd(f"cvg:kv_index:{v1_block['content_blake3']}", v1_kv_key)

            # Warm hit from Carol (same space_finance clearance!)
            warm_carol_resp = http.post(
                f"{gateway_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {tok_carol}", "X-Convergent-Space": str(space_finance)},
                json={"model": model_v1, "messages": [{"role": "user", "content": "What is the travel reimbursement cap?"}]},
            )
            warm_carol_resp.raise_for_status()

            # Supersede travel policy with v2 ($1,200)
            _mcp_call(
                http,
                gateway_url,
                tok_alice,
                "ingest_note",
                {
                    "space_id": str(space_finance),
                    "source_uri": policy_uri,
                    "content": "Corporate travel reimbursement cap is $1,200 per trip effective Q4.",
                },
            )

            # Wait until active search returns ONLY v2 and worker evicts v1 KV & warms v2 KV
            v2_blocks = _wait_until(
                lambda: [
                    b
                    for b in _mcp_call(
                        http,
                        gateway_url,
                        tok_alice,
                        "search_memory",
                        {"query": "travel reimbursement cap", "space_id": str(space_finance)},
                    )["blocks"]
                    if "$1,200" in b["canonical_text"]
                ]
            )
            v2_block = v2_blocks[0]
            v2_kv_key = derive_kv_key(master_seed, f"space:{space_finance}", model_v1, v2_block["content_blake3"])

            _wait_until(lambda: valkey.exists(v1_kv_key) == 0 and valkey.exists(v2_kv_key) == 1)

            # Verify active search has ZERO v1 blocks
            active_now = _mcp_call(
                http,
                gateway_url,
                tok_alice,
                "search_memory",
                {"query": "travel reimbursement cap", "space_id": str(space_finance)},
            )["blocks"]

            # Verify bi-temporal as_of=t_v1 recovers v1 cleanly
            hist_resp = http.get(
                f"{gateway_url}/v1/memory/history?source_uri={quote(policy_uri, safe='')}&as_of={quote(t_v1.isoformat(), safe='')}",
                headers={"Authorization": f"Bearer {tok_alice}"},
            )
            hist_resp.raise_for_status()
            hist_blocks = hist_resp.json()["blocks"]

            report["smart_memory"] = {
                "v1_block_id": v1_block["id"],
                "v2_block_id": v2_block["id"],
                "v2_citations": v2_block["citations"],
                "active_contains_v2_only": any("$1,200" in b["canonical_text"] for b in active_now)
                and not any("$500" in b["canonical_text"] for b in active_now),
                "as_of_v1_recovered": len(hist_blocks) == 1 and "$500" in hist_blocks[0]["canonical_text"],
                "as_of_v1_superseded_by": hist_blocks[0]["superseded_by"] if hist_blocks else None,
            }

            # =========================================================================
            # LAYER 2 · SMART PRIVATE CONTEXT: OpenFGA isolation, HMAC KV disjointness, revocation
            # =========================================================================
            shared_secret = f"Project Helios launch date is October 15 ({suffix})."
            _mcp_call(
                http,
                gateway_url,
                tok_alice,
                "ingest_note",
                {
                    "space_id": str(space_finance),
                    "source_uri": f"finance://helios-{suffix}.md",
                    "content": shared_secret,
                },
            )
            _mcp_call(
                http,
                gateway_url,
                tok_bob,
                "ingest_note",
                {
                    "space_id": str(space_eng),
                    "source_uri": f"eng://helios-{suffix}.md",
                    "content": shared_secret,
                },
            )

            alice_helios = _wait_until(
                lambda: [
                    b
                    for b in _mcp_call(
                        http, gateway_url, tok_alice, "search_memory", {"query": f"Helios {suffix}"}
                    )["blocks"]
                    if "Helios" in b["canonical_text"]
                ]
            )
            bob_helios = _wait_until(
                lambda: [
                    b
                    for b in _mcp_call(
                        http, gateway_url, tok_bob, "search_memory", {"query": f"Helios {suffix}"}
                    )["blocks"]
                    if "Helios" in b["canonical_text"]
                ]
            )

            # Alice warms her finance Helios block; Bob queries his engineering Helios block -> MUST be 0 cache hits!
            http.post(
                f"{gateway_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {tok_alice}", "X-Convergent-Space": str(space_finance)},
                json={"model": model_v1, "messages": [{"role": "user", "content": f"Project Helios {suffix}"}]},
            ).raise_for_status()
            bob_chat = http.post(
                f"{gateway_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {tok_bob}", "X-Convergent-Space": str(space_eng)},
                json={"model": model_v1, "messages": [{"role": "user", "content": f"Project Helios {suffix}"}]},
            )
            bob_chat.raise_for_status()

            kv_alice_helios = derive_kv_key(master_seed, f"space:{space_finance}", model_v1, alice_helios[0]["content_blake3"])
            kv_bob_helios = derive_kv_key(master_seed, f"space:{space_eng}", model_v1, bob_helios[0]["content_blake3"])

            # Revoke Bob's access to space_eng -> immediate Valkey ACL invalidation -> 0 blocks
            http.post(
                f"{gateway_url}/v1/spaces/{space_eng}/grants",
                headers={"Authorization": f"Bearer {tok_alice}"},
                json={"user_id": bob, "relation": "viewer", "action": "revoke"},
            ).raise_for_status()
            bob_after_revoke = _mcp_call(
                http, gateway_url, tok_bob, "search_memory", {"query": f"Helios {suffix}"}
            )["blocks"]

            report["smart_private_context"] = {
                "alice_sees_only_finance": all(b["clearance_scope"] == f"space:{space_finance}" for b in alice_helios),
                "bob_sees_only_engineering": all(b["clearance_scope"] == f"space:{space_eng}" for b in bob_helios),
                "identical_text_same_blake3": alice_helios[0]["content_blake3"] == bob_helios[0]["content_blake3"],
                "disjoint_hmac_kv_keys": kv_alice_helios != kv_bob_helios,
                "bob_cross_clearance_cache_hits": int(bob_chat.headers.get("x-convergent-cache-hits", "-1")),
                "carol_same_clearance_cache_hits": int(warm_carol_resp.headers.get("x-convergent-cache-hits", "-1")),
                "bob_blocks_after_revoke": len(bob_after_revoke),
            }

            # =========================================================================
            # LAYER 1 · SMART CACHE: Eviction + pre-warm hit + zero-downtime model swap
            # =========================================================================
            post_supersede_chat = http.post(
                f"{gateway_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {tok_alice}", "X-Convergent-Space": str(space_finance)},
                json={"model": model_v1, "messages": [{"role": "user", "content": "What is the travel reimbursement cap?"}]},
            )
            post_supersede_chat.raise_for_status()

            with psycopg.connect(PG_DSN) as conn:
                warmed_count = rewarm_model_and_cutover(
                    conn=conn,
                    valkey=valkey,
                    http=http,
                    new_model_id=model_v2,
                    new_endpoint_url=vllm_url,
                    master_seed=master_seed,
                    top_n=10,
                )
            models_after_swap = http.get(f"{gateway_url}/v1/models").json()["data"]

            report["smart_cache"] = {
                "cold_cache_hits": int(cold_resp.headers.get("x-convergent-cache-hits", "-1")),
                "warm_same_scope_hits": int(warm_carol_resp.headers.get("x-convergent-cache-hits", "-1")),
                "stale_v1_kv_evicted": valkey.exists(v1_kv_key) == 0,
                "v2_kv_prewarmed_hit": int(post_supersede_chat.headers.get("x-convergent-cache-hits", "-1")) >= 1,
                "proxy_overhead_ms": float(post_supersede_chat.headers.get("x-convergent-proxy-overhead-ms", "999")),
                "model_swap_rewarmed_blocks": warmed_count,
                "active_model_after_swap": models_after_swap[0]["id"],
            }
    finally:
        worker_loop.call_soon_threadsafe(worker_stop.set)
        worker_thread.join(timeout=3.0)
        worker_loop.close()
        for k, v in worker_env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        gw_proc.terminate()
        try:
            gw_proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            gw_proc.kill()
        vllm_server.shutdown()
        vllm_server.server_close()
        valkey.close()

    if verbose:
        _print_report(report)
    return report


def _is_healthy(http: httpx.Client, gateway_url: str) -> bool:
    try:
        return http.get(f"{gateway_url}/healthz", timeout=1.0).status_code == 200
    except Exception:
        return False


def _print_report(report: dict[str, Any]) -> None:
    print("\n================ PROJECT CONVERGENT · 3-LAYER LIVE PROOF ================")
    for layer, metrics in report.items():
        print(f"\n[{layer.upper()}]")
        for k, v in metrics.items():
            print(f"  - {k}: {v}")
    print("\n=========================================================================\n")


if __name__ == "__main__":
    run_three_layer_proof(verbose=True)
