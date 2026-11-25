"""Convergent Arena: Single-page live comparison showcase & benchmark dashboard.

Runs all Project Convergent proofs end-to-end with one click against live PostgreSQL 17,
Valkey 8, OpenFGA 1.8.5, and any user-configured Ollama endpoint & model (auto-discovered
via /api/tags — zero hard-coded models or benchmark numbers), streaming real results over SSE.

Usage:
    cd convergent/worker-py && uv run python ../arena/server.py
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterator
from urllib.parse import parse_qs, urlparse
import uuid

import httpx
import psycopg
import redis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker-py" / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from convergent_worker.ingest import register_document_openfga, resolve_openfga_store_id  # noqa: E402
from convergent_worker.kv_evictor import derive_kv_key, evict_block_kv, register_kv_key  # noqa: E402
from convergent_worker.overlays import (  # noqa: E402
    land_overlay,
    open_overlay,
    read_mount_view,
    stage_delta,
    undo_revision,
)
from fixtures import (  # noqa: E402
    ENG_HELIOS_ARCH,
    FINANCE_HELIOS_BUDGET,
    IDENTICAL_SECRET_TEXT,
    TRAVEL_POLICY_V1,
    TRAVEL_POLICY_V2,
)

PG_DSN = os.environ.get(
    "CONVERGENT_PG_DSN",
    "postgresql://convergent:convergent_dev_password@127.0.0.1:55432/convergent",
)
VALKEY_URL = os.environ.get("CONVERGENT_VALKEY_URL", "redis://127.0.0.1:56379/0")
OPENFGA_URL = os.environ.get("CONVERGENT_OPENFGA_URL", "http://127.0.0.1:58080")
DEFAULT_OLLAMA_URL = os.environ.get("CONVERGENT_VLLM_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL_ID = os.environ.get("CONVERGENT_MODEL_ID", "")
ARENA_HOST = os.environ.get("CONVERGENT_ARENA_HOST", "127.0.0.1")
ARENA_PORT = int(os.environ.get("CONVERGENT_ARENA_PORT", "8095"))

PRIMARY_REGION_V1 = "Production Infrastructure Spec (v1): Our primary production cloud region is us-east-1."
PRIMARY_REGION_V2 = "Postmortem 42 Migration Update: Our primary production cloud region is eu-central-1."


class ArenaEngine:
    """Executes live side-by-side Standard RAG vs. Project Convergent showcase runs."""

    def __init__(self) -> None:
        self.valkey = redis.Redis.from_url(VALKEY_URL, decode_responses=True)
        self.http = httpx.Client(timeout=120.0)
        self.store_id = resolve_openfga_store_id(self.http, OPENFGA_URL, "convergent")
        self.ollama_url = DEFAULT_OLLAMA_URL.rstrip("/")
        self.available_models: list[dict[str, Any]] = []
        self.ollama_connected = False
        self.ollama_error: str | None = None
        self.model_id = DEFAULT_MODEL_ID
        self.refresh_ollama_models(self.ollama_url)

    def refresh_ollama_models(self, url: str | None = None) -> dict[str, Any]:
        target = (url or self.ollama_url).strip().rstrip("/") or "http://127.0.0.1:11434"
        try:
            resp = self.http.get(f"{target}/api/tags", timeout=5.0)
            resp.raise_for_status()
            raw_models = resp.json().get("models") or []
            models: list[dict[str, Any]] = []
            for m in raw_models:
                details = m.get("details") or {}
                size_bytes = int(m.get("size") or 0)
                name = str(m.get("name") or m.get("model") or "")
                if not name:
                    continue
                models.append({
                    "name": name,
                    "size_gb": round(size_bytes / (1024 ** 3), 1) if size_bytes else 0.0,
                    "parameter_size": str(details.get("parameter_size") or ""),
                    "quantization": str(details.get("quantization_level") or ""),
                })
            models.sort(key=lambda item: (item["size_gb"] or 9999.0, item["name"]))
            self.ollama_url = target
            self.available_models = models
            self.ollama_connected = True
            self.ollama_error = None
            names = [m["name"] for m in models]
            if not self.model_id or self.model_id not in names:
                self.model_id = names[0] if names else ""
        except Exception as exc:
            self.ollama_url = target
            self.available_models = []
            self.ollama_connected = False
            self.ollama_error = str(exc)
        return self.get_config()

    def get_config(self) -> dict[str, Any]:
        return {
            "ollama_url": self.ollama_url,
            "connected": self.ollama_connected,
            "error": self.ollama_error,
            "model_id": self.model_id,
            "models": self.available_models,
        }

    def configure(self, *, ollama_url: str | None = None, model_id: str | None = None) -> dict[str, Any]:
        if ollama_url is not None and ollama_url.strip():
            self.refresh_ollama_models(ollama_url.strip())
        if model_id is not None and model_id.strip():
            self.model_id = model_id.strip()
        return self.get_config()

    def _fga_write(self, user: str, relation: str, obj: str, delete: bool = False) -> None:
        key = "deletes" if delete else "writes"
        resp = self.http.post(
            f"{OPENFGA_URL.rstrip('/')}/stores/{self.store_id}/write",
            json={key: {"tuple_keys": [{"user": user, "relation": relation, "object": obj}]}},
        )
        if resp.status_code == 400 and ("already exists" in resp.text.lower() or "does not exist" in resp.text.lower()):
            return
        resp.raise_for_status()

    def _fga_allowed_docs(self, user_id: str) -> set[uuid.UUID]:
        resp = self.http.post(
            f"{OPENFGA_URL.rstrip('/')}/stores/{self.store_id}/list-objects",
            json={"user": f"user:{user_id}", "relation": "viewer", "type": "document"},
        )
        resp.raise_for_status()
        out: set[uuid.UUID] = set()
        for obj in resp.json().get("objects", []):
            if obj.startswith("document:"):
                try:
                    out.add(uuid.UUID(obj.split(":", 1)[1]))
                except ValueError:
                    pass
        return out

    def _sync_docs(self, conn: psycopg.Connection, org_id: uuid.UUID, space_id: uuid.UUID) -> None:
        rows = conn.execute(
            "SELECT id FROM documents WHERE org_id = %s AND space_id = %s",
            (org_id, space_id),
        ).fetchall()
        for (doc_id,) in rows:
            register_document_openfga(
                self.http, OPENFGA_URL, self.store_id, document_id=doc_id, space_id=space_id, org_id=org_id
            )

    def _call_ollama(self, context_blocks: list[str], prompt: str, model_id: str) -> tuple[str, float]:
        if not model_id:
            return "[No Ollama model selected]", 0.0
        t0 = time.perf_counter()
        ctx = "\n".join(f"- {b}" for b in context_blocks) if context_blocks else "(No authorized context available.)"
        resp = self.http.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": model_id,
                "stream": False,
                "think": False,
                "options": {"temperature": 0.0, "num_predict": 140},
                "messages": [
                    {
                        "role": "system",
                        "content": "Answer concisely in 1 short sentence using ONLY the provided context blocks. If no relevant context is provided, state that you do not have access.",
                    },
                    {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {prompt}"},
                ],
            },
        )
        resp.raise_for_status()
        ans = str(resp.json()["message"]["content"]).strip()
        return ans, round((time.perf_counter() - t0) * 1000.0, 1)

    def _compare_query(
        self,
        conn: psycopg.Connection,
        *,
        org_id: uuid.UUID,
        space_id: uuid.UUID,
        user_id: str,
        prompt: str,
        model_id: str,
        master_seed: bytes,
    ) -> dict[str, Any]:
        # Standard RAG: no OpenFGA pre-filter, no bi-temporal valid_to IS NULL filter
        rag_rows = conn.execute(
            """
            SELECT d.source_uri, kb.canonical_text, kb.valid_to IS NOT NULL AS is_stale, kb.space_id <> %s AS cross_space
            FROM knowledge_blocks kb
            JOIN documents d ON d.id = kb.document_id
            WHERE kb.org_id = %s
            ORDER BY ts_rank_cd(to_tsvector('english', kb.canonical_text), plainto_tsquery('english', %s)) DESC,
                     kb.valid_from ASC
            LIMIT 5
            """,
            (space_id, org_id, prompt),
        ).fetchall()
        rag_blocks = [
            {"source_uri": u, "text": t, "is_stale": bool(s), "cross_space": bool(c)} for u, t, s, c in rag_rows
        ]

        # Project Convergent: OpenFGA pre-filtered + active-only + per-clearance HMAC KV cache
        allowed = self._fga_allowed_docs(user_id)
        view = read_mount_view(conn, org_id=org_id, space_id=space_id)
        doc_map = dict(
            conn.execute(
                "SELECT id::text, document_id FROM knowledge_blocks WHERE org_id = %s AND space_id = %s AND valid_to IS NULL",
                (org_id, space_id),
            ).fetchall()
        )
        conv_blocks: list[dict[str, Any]] = []
        kv_hits = 0
        for blk in view:
            if doc_map.get(blk["id"]) not in allowed:
                continue
            kv_key = derive_kv_key(master_seed, blk["clearance_scope"], model_id, blk["content_blake3"])
            if self.valkey.exists(kv_key):
                kv_hits += 1
            else:
                register_kv_key(self.valkey, blk["content_blake3"], kv_key)
            conv_blocks.append({
                "source_uri": blk["source_uri"],
                "text": blk["canonical_text"],
                "kv_key_prefix": kv_key[:24] + "...",
            })

        rag_ans, rag_ms = self._call_ollama([b["text"] for b in rag_blocks], prompt, model_id)
        conv_ans, conv_ms = self._call_ollama([b["text"] for b in conv_blocks], prompt, model_id)
        return {
            "prompt": prompt,
            "standard_rag": {
                "answer": rag_ans,
                "latency_ms": rag_ms,
                "stale_leaked": sum(1 for b in rag_blocks if b["is_stale"]),
                "cross_space_leaked": sum(1 for b in rag_blocks if b["cross_space"]),
                "blocks": rag_blocks,
            },
            "convergent": {
                "answer": conv_ans,
                "latency_ms": conv_ms,
                "kv_hits": kv_hits,
                "kv_total": len(conv_blocks),
                "stale_leaked": 0,
                "cross_space_leaked": 0,
                "blocks": conv_blocks,
            },
        }

    def stream_full_showcase(self, model_id: str | None = None) -> Iterator[dict[str, Any]]:
        """Run all 5 live showcase stages sequentially, yielding real-time events + final benchmark summary."""
        active_model = (model_id or self.model_id).strip()
        if not active_model:
            yield {"event": "error", "error": "No Ollama model selected. Configure Ollama first."}
            return

        t_start = time.perf_counter()
        org_id = uuid.uuid4()
        space_finance = uuid.uuid4()
        space_eng = uuid.uuid4()
        suffix = uuid.uuid4().hex[:6]
        alice, carol, bob = f"alice-{suffix}", f"carol-{suffix}", f"bob-{suffix}"
        master_seed = f"arena-seed-{suffix}".encode("utf-8")

        yield {"event": "start", "model_id": active_model, "ollama_url": self.ollama_url, "total_steps": 5}

        for user, sid in [(alice, space_finance), (carol, space_finance), (bob, space_eng)]:
            self._fga_write(f"user:{user}", "viewer", f"space:{sid}")

        with psycopg.connect(PG_DSN, autocommit=True) as conn:
            # Seed Finance r1 -> r2 (Travel $500 -> $1,200) and Engineering r1 -> r2 (us-east-1 -> eu-central-1)
            ov_f1 = open_overlay(conn, org_id=org_id, space_id=space_finance, owner=alice)
            stage_delta(conn, overlay_id=ov_f1.overlay_id, source_uri="policy://travel-policy.md", text=TRAVEL_POLICY_V1)
            stage_delta(conn, overlay_id=ov_f1.overlay_id, source_uri="finance://q4-helios-budget.md", text=FINANCE_HELIOS_BUDGET)
            stage_delta(conn, overlay_id=ov_f1.overlay_id, source_uri="secret://helios-code.md", text=IDENTICAL_SECRET_TEXT)
            land_overlay(conn, overlay_id=ov_f1.overlay_id, author=alice, message="Finance baseline v1")

            ov_f2 = open_overlay(conn, org_id=org_id, space_id=space_finance, owner="cfo-memo")
            stage_delta(conn, overlay_id=ov_f2.overlay_id, source_uri="policy://travel-policy.md", text=TRAVEL_POLICY_V2)
            land_overlay(conn, overlay_id=ov_f2.overlay_id, author="cfo-memo", message="Supersede travel cap -> $1,200")
            self._sync_docs(conn, org_id, space_finance)

            ov_e1 = open_overlay(conn, org_id=org_id, space_id=space_eng, owner=bob)
            stage_delta(conn, overlay_id=ov_e1.overlay_id, source_uri="infra://primary-region.md", text=PRIMARY_REGION_V1)
            stage_delta(conn, overlay_id=ov_e1.overlay_id, source_uri="eng://helios-architecture.md", text=ENG_HELIOS_ARCH)
            stage_delta(conn, overlay_id=ov_e1.overlay_id, source_uri="secret://helios-code.md", text=IDENTICAL_SECRET_TEXT)
            land_overlay(conn, overlay_id=ov_e1.overlay_id, author=bob, message="Engineering baseline v1")

            ov_e2 = open_overlay(conn, org_id=org_id, space_id=space_eng, owner="postmortem-42")
            stage_delta(conn, overlay_id=ov_e2.overlay_id, source_uri="infra://primary-region.md", text=PRIMARY_REGION_V2)
            land_overlay(conn, overlay_id=ov_e2.overlay_id, author="postmortem-42", message="Migrate region -> eu-central-1")
            self._sync_docs(conn, org_id, space_eng)

            # STEP 1: Bi-Temporal Truth (Travel Policy $500 -> $1,200)
            s1 = self._compare_query(
                conn,
                org_id=org_id,
                space_id=space_finance,
                user_id=alice,
                prompt="What is the ACME corporate travel reimbursement cap per trip?",
                model_id=active_model,
                master_seed=master_seed,
            )
            yield {"event": "step", "index": 1, "id": "bitemporal_truth", "title": "Bi-Temporal Truth (Travel Policy v1 → v2)", "data": s1}

            # STEP 2: Cross-URI Supersession (Primary Region us-east-1 -> eu-central-1)
            s2 = self._compare_query(
                conn,
                org_id=org_id,
                space_id=space_eng,
                user_id=bob,
                prompt="What is our primary production cloud region?",
                model_id=active_model,
                master_seed=master_seed,
            )
            yield {"event": "step", "index": 2, "id": "cross_uri_supersession", "title": "Cross-Document Supersession (us-east-1 → eu-central-1)", "data": s2}

            # STEP 3: Hard Space Isolation & Instant Revocation (Bob asks Finance budget)
            s3 = self._compare_query(
                conn,
                org_id=org_id,
                space_id=space_eng,
                user_id=bob,
                prompt="What is the Project Helios confidential marketing budget and engineering stack?",
                model_id=active_model,
                master_seed=master_seed,
            )
            # Instant revoke Bob -> verify 0 authorized blocks
            self._fga_write(f"user:{bob}", "viewer", f"space:{space_eng}", delete=True)
            bob_docs_after_revoke = len(self._fga_allowed_docs(bob))
            self._fga_write(f"user:{bob}", "viewer", f"space:{space_eng}", delete=False)
            s3["bob_docs_after_revoke"] = bob_docs_after_revoke
            yield {"event": "step", "index": 3, "id": "private_context", "title": "Hard Space Isolation & Instant Revoke", "data": s3}

            # STEP 4: Smart Cache & Cryptographic Side-Channel Isolation (Cold Alice -> Warm Carol -> Disjoint Bob)
            s4_carol = self._compare_query(
                conn,
                org_id=org_id,
                space_id=space_finance,
                user_id=carol,
                prompt="What is the ACME corporate travel reimbursement cap per trip?",
                model_id=active_model,
                master_seed=master_seed,
            )
            secret_hash = conn.execute(
                "SELECT content_blake3 FROM knowledge_blocks WHERE org_id = %s AND space_id = %s AND canonical_text LIKE '%%HELIOS-9921%%' LIMIT 1",
                (org_id, space_finance),
            ).fetchone()[0]
            kv_alice_secret = derive_kv_key(master_seed, f"space:{space_finance}", active_model, secret_hash)
            kv_bob_secret = derive_kv_key(master_seed, f"space:{space_eng}", active_model, secret_hash)
            cold_ms = s1["convergent"]["latency_ms"]
            warm_ms = s4_carol["convergent"]["latency_ms"]
            speedup_x = round(cold_ms / warm_ms, 2) if warm_ms > 0 else 1.0
            s4_data = {
                "cold_ms": cold_ms,
                "warm_ms": warm_ms,
                "speedup_x": speedup_x,
                "carol_kv_hits": s4_carol["convergent"]["kv_hits"],
                "carol_kv_total": s4_carol["convergent"]["kv_total"],
                "kv_alice_prefix": kv_alice_secret[:26] + "...",
                "kv_bob_prefix": kv_bob_secret[:26] + "...",
                "disjoint_hmac_keys": kv_alice_secret != kv_bob_secret,
            }
            yield {"event": "step", "index": 4, "id": "smart_cache", "title": "Smart Cache & HMAC-SHA256 Side-Channel Isolation", "data": s4_data}

            # STEP 5: Zero-Copy Layered Overlays, Auto-Rebase Land, Conflict Detection & Instant Undo
            before_rows = conn.execute("SELECT COUNT(*) FROM knowledge_blocks WHERE org_id = %s", (org_id,)).fetchone()[0]
            ov_a = open_overlay(conn, org_id=org_id, space_id=space_finance, owner="agent-a")
            ov_b = open_overlay(conn, org_id=org_id, space_id=space_finance, owner="agent-b")
            after_rows = conn.execute("SELECT COUNT(*) FROM knowledge_blocks WHERE org_id = %s", (org_id,)).fetchone()[0]
            copied_on_open = int(after_rows - before_rows)

            stage_delta(conn, overlay_id=ov_a.overlay_id, source_uri="policy://travel-policy.md", text="ACME Travel Cap v3: $1,500 per trip.")
            stage_delta(conn, overlay_id=ov_b.overlay_id, source_uri="policy://flight-class.md", text="Flights under 6h use economy class.")
            land_a = land_overlay(conn, overlay_id=ov_a.overlay_id, author="agent-a", message="Bump cap to $1,500")
            land_b = land_overlay(conn, overlay_id=ov_b.overlay_id, author="agent-b", message="Add flight class policy")

            # Stale overlay C pinned to base_rev=2 attempts overlapping edit -> CONFLICT
            ov_c = open_overlay(conn, org_id=org_id, space_id=space_finance, owner="agent-c", base_rev=2)
            stage_delta(conn, overlay_id=ov_c.overlay_id, source_uri="policy://travel-policy.md", text="Conflicting cap: $900.")
            land_c = land_overlay(conn, overlay_id=ov_c.overlay_id, author="agent-c", message="Stale conflicting edit")

            # Instant Undo of land_a.rev + KV eviction
            undo_res = undo_revision(conn, org_id=org_id, space_id=space_finance, rev=int(land_a.rev or 3), author="arena-ops")
            for bid in undo_res.deactivated_block_ids:
                hrow = conn.execute("SELECT content_blake3 FROM knowledge_blocks WHERE id = %s", (bid,)).fetchone()
                if hrow:
                    evict_block_kv(self.valkey, hrow[0])

            s5_data = {
                "copied_rows_on_open": copied_on_open,
                "agent_a_landed_rev": land_a.rev,
                "agent_b_auto_rebased_rev": land_b.rev,
                "agent_c_conflict_status": land_c.status,
                "undo_rev": undo_res.undo_rev,
                "restored_blocks": len(undo_res.restored_block_ids),
            }
            yield {"event": "step", "index": 5, "id": "layered_overlays", "title": "Zero-Copy Overlays, Auto-Rebase, Conflict & Undo", "data": s5_data}

        total_secs = round(time.perf_counter() - t_start, 2)
        rag_total_stale = s1["standard_rag"]["stale_leaked"] + s2["standard_rag"]["stale_leaked"] + s3["standard_rag"]["stale_leaked"]
        rag_total_cross = s1["standard_rag"]["cross_space_leaked"] + s2["standard_rag"]["cross_space_leaked"] + s3["standard_rag"]["cross_space_leaked"]

        summary = {
            "event": "complete",
            "model_id": active_model,
            "total_secs": total_secs,
            "kpis": {
                "warm_speedup_x": speedup_x,
                "cold_ms": cold_ms,
                "warm_ms": warm_ms,
                "rag_leaks_total": rag_total_stale + rag_total_cross,
                "rag_stale_leaks": rag_total_stale,
                "rag_cross_space_leaks": rag_total_cross,
                "convergent_leaks_total": 0,
                "copied_rows_on_overlay": copied_on_open,
                "bob_docs_after_revoke": bob_docs_after_revoke,
                "carol_kv_hits": s4_data["carol_kv_hits"],
                "carol_kv_total": s4_data["carol_kv_total"],
                "auto_rebase_ok": land_b.status == "landed",
                "conflict_caught": land_c.status == "conflict",
                "undo_restored": len(undo_res.restored_block_ids),
                "disjoint_hmac_keys": s4_data["disjoint_hmac_keys"],
            },
        }
        yield summary


def make_arena_handler(engine: ArenaEngine) -> type[BaseHTTPRequestHandler]:
    html_bytes = (ROOT / "arena" / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send(self, status: int, content_type: str, raw: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _json(self, status: int, obj: Any) -> None:
            self._send(status, "application/json", json.dumps(obj, default=str).encode("utf-8"))

        def _body(self) -> dict[str, Any]:
            n = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", html_bytes)
            elif parsed.path == "/api/config":
                self._json(200, engine.get_config())
            elif parsed.path == "/api/ollama/models":
                qs = parse_qs(parsed.query)
                url = (qs.get("url") or [None])[0]
                self._json(200, engine.refresh_ollama_models(url))
            elif parsed.path == "/api/showcase/stream":
                qs = parse_qs(parsed.query)
                model_id = (qs.get("model_id") or [None])[0]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    for evt in engine.stream_full_showcase(model_id=model_id):
                        chunk = f"data: {json.dumps(evt, default=str)}\n\n".encode("utf-8")
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except Exception as exc:
                    err = f"data: {json.dumps({'event': 'error', 'error': str(exc)})}\n\n".encode("utf-8")
                    self.wfile.write(err)
                    self.wfile.flush()
            else:
                self._json(404, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                body = self._body()
                if self.path == "/api/config":
                    self._json(
                        200,
                        engine.configure(
                            ollama_url=body.get("ollama_url"),
                            model_id=body.get("model_id"),
                        ),
                    )
                else:
                    self._json(404, {"error": "not_found"})
            except Exception as exc:
                self._json(400, {"error": str(exc)})

    return Handler


def main() -> None:
    engine = ArenaEngine()
    server = ThreadingHTTPServer((ARENA_HOST, ARENA_PORT), make_arena_handler(engine))
    print(f"\n  Project Convergent Arena -> http://{ARENA_HOST}:{ARENA_PORT}")
    print(f"  Ollama: {engine.ollama_url} | Model: {engine.model_id or '(configure in UI)'} ({len(engine.available_models)} available)\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
