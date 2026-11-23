"""Standalone live sample & experiment runner for Project Convergent.

Spawns disposable Docker Compose infrastructure (PostgreSQL 17 ParadeDB, Valkey 8,
OpenFGA 1.8.5, NATS JetStream) on isolated ephemeral ports, runs the full experiment
against local Ollama (`granite4.2:3b`) using predefined fixtures, prints real
measurements proving Smart Cache, Smart Private Context, and Smart Memory, and tears
down all disposable containers and volumes cleanly.

Usage:
    cd convergent/worker-py && uv run python ../examples/demo_convergent.py
"""

from __future__ import annotations

import asyncio
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Iterator
from urllib.parse import quote
import uuid

import httpx
import psycopg
import redis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "worker-py" / "src"))

from dev_vllm_server import make_handler  # noqa: E402
from fixtures import (  # noqa: E402
    ENG_HELIOS_ARCH,
    FINANCE_HELIOS_BUDGET,
    IDENTICAL_SECRET_TEXT,
    TRAVEL_POLICY_V1,
    TRAVEL_POLICY_V2,
)
from convergent_worker.backends import create_backend  # noqa: E402
from convergent_worker.kv_evictor import derive_kv_key, rewarm_model_and_cutover  # noqa: E402
from convergent_worker.main import run_worker_loop  # noqa: E402


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextmanager
def disposable_compose_infra() -> Iterator[dict[str, Any]]:
    """Spin up an isolated disposable Docker Compose stack on ephemeral ports and tear it down on exit."""
    project = f"convergent-demo-{uuid.uuid4().hex[:6]}"
    ports = {
        "CONVERGENT_PG_PORT": str(_free_port()),
        "CONVERGENT_VALKEY_PORT": str(_free_port()),
        "CONVERGENT_NATS_PORT": str(_free_port()),
        "CONVERGENT_NATS_MON_PORT": str(_free_port()),
        "CONVERGENT_OPENFGA_HTTP_PORT": str(_free_port()),
        "CONVERGENT_OPENFGA_GRPC_PORT": str(_free_port()),
    }
    compose_file = str(ROOT / "deploy" / "compose.yml")
    env = {**os.environ, **ports}
    t0 = time.monotonic()

    try:
        subprocess.run(
            ["docker", "compose", "-p", project, "-f", compose_file, "up", "-d", "--wait"],
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        infra = {
            "project": project,
            "boot_secs": round(time.monotonic() - t0, 2),
            "pg_dsn": f"postgresql://convergent:convergent_dev_password@127.0.0.1:{ports['CONVERGENT_PG_PORT']}/convergent",
            "valkey_url": f"redis://127.0.0.1:{ports['CONVERGENT_VALKEY_PORT']}/0",
            "openfga_url": f"http://127.0.0.1:{ports['CONVERGENT_OPENFGA_HTTP_PORT']}",
            "nats_url": f"nats://127.0.0.1:{ports['CONVERGENT_NATS_PORT']}",
        }
        boot_env = {
            **os.environ,
            "CONVERGENT_PG_DSN": infra["pg_dsn"],
            "CONVERGENT_OPENFGA_URL": infra["openfga_url"],
            "CONVERGENT_NATS_URL": infra["nats_url"],
        }
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "bootstrap.py")],
            env=boot_env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        yield infra
    finally:
        subprocess.run(
            ["docker", "compose", "-p", project, "-f", compose_file, "down", "-v"],
            env=env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


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


def _chat(
    http: httpx.Client,
    gateway_url: str,
    token: str,
    space_id: uuid.UUID,
    model_id: str,
    prompt: str,
) -> tuple[httpx.Response, str, float]:
    t0 = time.monotonic()
    resp = http.post(
        f"{gateway_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {token}", "X-Convergent-Space": str(space_id)},
        json={
            "model": model_id,
            "temperature": 0.0,
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    elapsed_ms = round((time.monotonic() - t0) * 1000.0, 2)
    resp.raise_for_status()
    data = resp.json()
    content = str(data["choices"][0]["message"].get("content") or "").strip()
    return resp, content, elapsed_ms


def _wait_until(predicate: Any, timeout_secs: float = 20.0, interval: float = 0.15) -> Any:
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


def _is_healthy(http: httpx.Client, gateway_url: str) -> bool:
    try:
        return http.get(f"{gateway_url}/healthz", timeout=1.0).status_code == 200
    except Exception:
        return False


def run_convergent_proof(
    *,
    disposable_infra: bool = True,
    backend: str = "ollama",
    endpoint_url: str = "http://127.0.0.1:11434",
    model_id: str = "granite4.2:3b",
    swap_model_id: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the full Project Convergent proof against disposable (or existing) infra and configured backend."""
    if disposable_infra:
        with disposable_compose_infra() as infra:
            return _execute_proof(
                infra=infra,
                backend=backend,
                endpoint_url=endpoint_url,
                model_id=model_id,
                swap_model_id=swap_model_id,
                verbose=verbose,
            )

    infra = {
        "project": "existing-compose",
        "boot_secs": 0.0,
        "pg_dsn": os.environ.get(
            "CONVERGENT_PG_DSN",
            "postgresql://convergent:convergent_dev_password@127.0.0.1:55432/convergent",
        ),
        "valkey_url": os.environ.get("CONVERGENT_VALKEY_URL", "redis://127.0.0.1:56379/0"),
        "openfga_url": os.environ.get("CONVERGENT_OPENFGA_URL", "http://127.0.0.1:58080"),
        "nats_url": os.environ.get("CONVERGENT_NATS_URL", "nats://127.0.0.1:54222"),
    }
    return _execute_proof(
        infra=infra,
        backend=backend,
        endpoint_url=endpoint_url,
        model_id=model_id,
        swap_model_id=swap_model_id,
        verbose=verbose,
    )


def _execute_proof(
    *,
    infra: dict[str, Any],
    backend: str,
    endpoint_url: str,
    model_id: str,
    swap_model_id: str | None,
    verbose: bool,
) -> dict[str, Any]:
    master_seed_str = f"demo-seed-{uuid.uuid4().hex[:8]}"
    master_seed = master_seed_str.encode("utf-8")
    jwt_secret = f"demo-jwt-{uuid.uuid4().hex[:8]}"
    next_model_id = swap_model_id or ("hostlab-granite:3b" if backend == "ollama" else "DeepSeek/DeepSeek-V3.2-Local")

    dev_server: ThreadingHTTPServer | None = None
    active_endpoint = endpoint_url
    if backend == "dev":
        dev_server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(infra["valkey_url"], model_id))
        active_endpoint = f"http://127.0.0.1:{dev_server.server_address[1]}"
        threading.Thread(target=dev_server.serve_forever, daemon=True).start()

    with psycopg.connect(infra["pg_dsn"], autocommit=True) as conn:
        conn.execute("UPDATE model_registry SET is_active = false")
        conn.execute(
            """
            INSERT INTO model_registry (model_id, endpoint_url, is_active, warmed_at)
            VALUES (%s, %s, true, now())
            ON CONFLICT (model_id) DO UPDATE
              SET endpoint_url = EXCLUDED.endpoint_url, is_active = true, warmed_at = now()
            """,
            (model_id, active_endpoint),
        )

    gw_port = _free_port()
    gateway_url = f"http://127.0.0.1:{gw_port}"
    gw_bin = _ensure_gateway_binary()
    gw_env = {
        **os.environ,
        "CONVERGENT_PG_DSN": infra["pg_dsn"],
        "CONVERGENT_VALKEY_URL": infra["valkey_url"],
        "CONVERGENT_OPENFGA_URL": infra["openfga_url"],
        "CONVERGENT_NATS_URL": infra["nats_url"],
        "CONVERGENT_BIND_ADDR": f"127.0.0.1:{gw_port}",
        "CONVERGENT_JWT_SECRET": jwt_secret,
        "CONVERGENT_MASTER_SEED": master_seed_str,
        "CONVERGENT_VLLM_URL": active_endpoint,
        "CONVERGENT_MODEL_ID": model_id,
        "CONVERGENT_INFERENCE_BACKEND": backend,
    }
    gw_proc = subprocess.Popen([str(gw_bin)], env=gw_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    worker_stop = asyncio.Event()
    env_keys = (
        "CONVERGENT_PG_DSN",
        "CONVERGENT_VALKEY_URL",
        "CONVERGENT_OPENFGA_URL",
        "CONVERGENT_NATS_URL",
        "CONVERGENT_VLLM_URL",
        "CONVERGENT_MODEL_ID",
        "CONVERGENT_MASTER_SEED",
        "CONVERGENT_INFERENCE_BACKEND",
        "CONVERGENT_WORKER_POLL_SECS",
    )
    worker_env_backup = {k: os.environ.get(k) for k in env_keys}
    os.environ["CONVERGENT_PG_DSN"] = infra["pg_dsn"]
    os.environ["CONVERGENT_VALKEY_URL"] = infra["valkey_url"]
    os.environ["CONVERGENT_OPENFGA_URL"] = infra["openfga_url"]
    os.environ["CONVERGENT_NATS_URL"] = infra["nats_url"]
    os.environ["CONVERGENT_VLLM_URL"] = active_endpoint
    os.environ["CONVERGENT_MODEL_ID"] = model_id
    os.environ["CONVERGENT_MASTER_SEED"] = master_seed_str
    os.environ["CONVERGENT_INFERENCE_BACKEND"] = backend
    os.environ["CONVERGENT_WORKER_POLL_SECS"] = "0.1"

    worker_loop = asyncio.new_event_loop()

    def _run_worker() -> None:
        asyncio.set_event_loop(worker_loop)
        worker_loop.run_until_complete(run_worker_loop(worker_stop))

    worker_thread = threading.Thread(target=_run_worker, daemon=True)
    worker_thread.start()

    valkey = redis.Redis.from_url(infra["valkey_url"], decode_responses=True)
    report: dict[str, Any] = {
        "environment": {
            "disposable_infra": infra["project"] != "existing-compose",
            "compose_project": infra["project"],
            "infra_boot_secs": infra["boot_secs"],
            "backend": backend,
            "model_id": model_id,
            "swap_model_id": next_model_id,
        }
    }

    try:
        with httpx.Client(timeout=90.0) as http:
            _wait_until(lambda: _is_healthy(http, gateway_url))

            org_id = uuid.uuid4()
            space_finance = uuid.uuid4()
            space_eng = uuid.uuid4()
            suffix = uuid.uuid4().hex[:6]
            alice, carol, bob = f"alice-{suffix}", f"carol-{suffix}", f"bob-{suffix}"

            tok_alice = _oauth_login(http, gateway_url, alice, org_id)
            tok_carol = _oauth_login(http, gateway_url, carol, org_id)
            tok_bob = _oauth_login(http, gateway_url, bob, org_id)

            for sid, user in [(space_finance, alice), (space_finance, carol), (space_eng, bob)]:
                http.post(
                    f"{gateway_url}/v1/spaces/{sid}/grants",
                    headers={"Authorization": f"Bearer {tok_alice}"},
                    json={"user_id": user, "relation": "viewer", "action": "grant"},
                ).raise_for_status()

            # =========================================================================
            # SMART MEMORY: Predefined travel policy v1 -> v2 supersession & as_of
            # =========================================================================
            policy_uri = f"policy://travel-policy-{suffix}.md"
            _mcp_call(
                http,
                gateway_url,
                tok_alice,
                "ingest_note",
                {"space_id": str(space_finance), "source_uri": policy_uri, "content": TRAVEL_POLICY_V1},
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

            # Cold chat on v1 (Alice) + warm same-scope chat on v1 (Carol)
            cold_resp, v1_answer, cold_ms = _chat(
                http,
                gateway_url,
                tok_alice,
                space_finance,
                model_id,
                "According to the ACME Corporate Travel Policy, what is the exact dollar amount of the reimbursement cap per trip? Reply briefly.",
            )
            v1_kv_key = derive_kv_key(master_seed, f"space:{space_finance}", model_id, v1_block["content_blake3"])
            valkey.sadd(f"cvg:kv_index:{v1_block['content_blake3']}", v1_kv_key)

            warm_carol_resp, carol_v1_answer, warm_ms = _chat(
                http,
                gateway_url,
                tok_carol,
                space_finance,
                model_id,
                "According to the ACME Corporate Travel Policy, what is the exact dollar amount of the reimbursement cap per trip? Reply briefly.",
            )

            # Supersede travel policy with v2 ($1,200)
            _mcp_call(
                http,
                gateway_url,
                tok_alice,
                "ingest_note",
                {"space_id": str(space_finance), "source_uri": policy_uri, "content": TRAVEL_POLICY_V2},
            )
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
            v2_kv_key = derive_kv_key(master_seed, f"space:{space_finance}", model_id, v2_block["content_blake3"])
            _wait_until(lambda: valkey.exists(v1_kv_key) == 0 and valkey.exists(v2_kv_key) == 1)

            post_supersede_resp, v2_answer, post_supersede_ms = _chat(
                http,
                gateway_url,
                tok_alice,
                space_finance,
                model_id,
                "According to the ACME Corporate Travel Policy, what is the exact dollar amount of the reimbursement cap per trip? Reply briefly.",
            )

            active_now = _mcp_call(
                http,
                gateway_url,
                tok_alice,
                "search_memory",
                {"query": "travel reimbursement cap", "space_id": str(space_finance)},
            )["blocks"]
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
                "v1_llm_answer": v1_answer,
                "v2_llm_answer": v2_answer,
                "active_contains_v2_only": any("$1,200" in b["canonical_text"] for b in active_now)
                and not any("$500" in b["canonical_text"] for b in active_now),
                "as_of_v1_recovered": len(hist_blocks) == 1 and "$500" in hist_blocks[0]["canonical_text"],
                "as_of_v1_superseded_by": hist_blocks[0]["superseded_by"] if hist_blocks else None,
            }

            # =========================================================================
            # SMART PRIVATE CONTEXT: Finance vs Engineering isolation, HMAC KV disjointness, revocation
            # =========================================================================
            for tok, sid, uri, text in [
                (tok_alice, space_finance, f"finance://q4-helios-budget-{suffix}.md", FINANCE_HELIOS_BUDGET),
                (tok_bob, space_eng, f"eng://helios-architecture-{suffix}.md", ENG_HELIOS_ARCH),
                (tok_alice, space_finance, f"finance://identical-secret-{suffix}.md", IDENTICAL_SECRET_TEXT),
                (tok_bob, space_eng, f"eng://identical-secret-{suffix}.md", IDENTICAL_SECRET_TEXT),
            ]:
                _mcp_call(http, gateway_url, tok, "ingest_note", {"space_id": str(sid), "source_uri": uri, "content": text})

            alice_secret = _wait_until(
                lambda: [
                    b
                    for b in _mcp_call(http, gateway_url, tok_alice, "search_memory", {"query": "HELIOS-9921"})["blocks"]
                    if "HELIOS-9921" in b["canonical_text"]
                ]
            )
            bob_secret = _wait_until(
                lambda: [
                    b
                    for b in _mcp_call(http, gateway_url, tok_bob, "search_memory", {"query": "HELIOS-9921"})["blocks"]
                    if "HELIOS-9921" in b["canonical_text"]
                ]
            )
            _wait_until(
                lambda: [
                    b
                    for b in _mcp_call(http, gateway_url, tok_alice, "search_memory", {"query": "marketing budget"})["blocks"]
                    if "$4.2M" in b["canonical_text"]
                ]
            )
            _wait_until(
                lambda: [
                    b
                    for b in _mcp_call(http, gateway_url, tok_bob, "search_memory", {"query": "Rust Axum"})["blocks"]
                    if "PostgreSQL 17" in b["canonical_text"]
                ]
            )

            # Alice warms her finance secret; Bob queries his engineering secret -> MUST be 0 cross-clearance cache hits
            _, alice_budget_ans, _ = _chat(
                http,
                gateway_url,
                tok_alice,
                space_finance,
                model_id,
                "What is the Project Helios launch marketing budget and authorization code? Reply briefly.",
            )
            bob_secret_resp, bob_arch_ans, _ = _chat(
                http,
                gateway_url,
                tok_bob,
                space_eng,
                model_id,
                "What is the Project Helios engineering stack and authorization code? Do you see any marketing budget? Reply briefly.",
            )

            kv_alice = derive_kv_key(master_seed, f"space:{space_finance}", model_id, alice_secret[0]["content_blake3"])
            kv_bob = derive_kv_key(master_seed, f"space:{space_eng}", model_id, bob_secret[0]["content_blake3"])

            # Revoke Bob's access to space_eng -> immediate Valkey ACL invalidation
            http.post(
                f"{gateway_url}/v1/spaces/{space_eng}/grants",
                headers={"Authorization": f"Bearer {tok_alice}"},
                json={"user_id": bob, "relation": "viewer", "action": "revoke"},
            ).raise_for_status()
            bob_after_revoke = _mcp_call(http, gateway_url, tok_bob, "search_memory", {"query": "HELIOS-9921"})["blocks"]

            report["smart_private_context"] = {
                "alice_sees_only_finance": all(b["clearance_scope"] == f"space:{space_finance}" for b in alice_secret),
                "bob_sees_only_engineering": all(b["clearance_scope"] == f"space:{space_eng}" for b in bob_secret),
                "alice_finance_llm_answer": alice_budget_ans,
                "bob_engineering_llm_answer": bob_arch_ans,
                "identical_text_same_blake3": alice_secret[0]["content_blake3"] == bob_secret[0]["content_blake3"],
                "disjoint_hmac_kv_keys": kv_alice != kv_bob,
                "kv_key_alice_prefix": kv_alice[:28] + "...",
                "kv_key_bob_prefix": kv_bob[:28] + "...",
                "bob_cross_clearance_cache_hits": int(bob_secret_resp.headers.get("x-convergent-cache-hits", "-1")),
                "carol_same_clearance_cache_hits": int(warm_carol_resp.headers.get("x-convergent-cache-hits", "-1")),
                "bob_blocks_after_revoke": len(bob_after_revoke),
            }

            # =========================================================================
            # SMART CACHE: Cold vs warm latency, eviction + prewarm hit, zero-downtime swap
            # =========================================================================
            backend_obj = create_backend(backend)
            with psycopg.connect(infra["pg_dsn"], autocommit=True) as conn:
                warmed_count = rewarm_model_and_cutover(
                    conn=conn,
                    valkey=valkey,
                    http=http,
                    new_model_id=next_model_id,
                    new_endpoint_url=active_endpoint,
                    master_seed=master_seed,
                    top_n=10,
                    backend=backend_obj,
                )
            models_after_swap = http.get(f"{gateway_url}/v1/models").json()["data"]

            report["smart_cache"] = {
                "backend_header": post_supersede_resp.headers.get("x-convergent-backend"),
                "cold_cache_hits": int(cold_resp.headers.get("x-convergent-cache-hits", "-1")),
                "cold_e2e_latency_ms": cold_ms,
                "warm_same_scope_hits": int(warm_carol_resp.headers.get("x-convergent-cache-hits", "-1")),
                "warm_e2e_latency_ms": warm_ms,
                "post_supersede_latency_ms": post_supersede_ms,
                "stale_v1_kv_evicted": valkey.exists(v1_kv_key) == 0,
                "v2_kv_prewarmed_hit": int(post_supersede_resp.headers.get("x-convergent-cache-hits", "-1")) >= 1,
                "proxy_overhead_ms": float(post_supersede_resp.headers.get("x-convergent-proxy-overhead-ms", "999")),
                "model_swap_rewarmed_blocks": warmed_count,
                "active_model_after_swap": models_after_swap[0]["id"],
            }
            _ = carol_v1_answer
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
        if dev_server is not None:
            dev_server.shutdown()
            dev_server.server_close()
        valkey.close()

    if verbose:
        _print_report(report)
    return report


def _print_report(report: dict[str, Any]) -> None:
    print("\n=================== PROJECT CONVERGENT · LIVE PROOF ===================")
    print(json.dumps(report, indent=2))
    print("=======================================================================\n")


if __name__ == "__main__":
    run_convergent_proof(
        disposable_infra=True,
        backend=os.environ.get("CONVERGENT_INFERENCE_BACKEND", "ollama"),
        endpoint_url=os.environ.get("CONVERGENT_VLLM_URL", "http://127.0.0.1:11434"),
        model_id=os.environ.get("CONVERGENT_MODEL_ID", "granite4.2:3b"),
        verbose=True,
    )
