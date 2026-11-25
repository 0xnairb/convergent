"""Convergent Studio: Minimal, interactive MVP Web GUI backend for Project Convergent.

Connects to live PostgreSQL 17, Valkey 8, OpenFGA 1.8.5, and any user-configured
Ollama instance (auto-discovering installed models via /api/tags — no hard-coded model).

Usage:
    cd convergent/worker-py && uv run python ../studio/server.py
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
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
    current_head_rev,
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
STUDIO_HOST = os.environ.get("CONVERGENT_STUDIO_HOST", "127.0.0.1")
STUDIO_PORT = int(os.environ.get("CONVERGENT_STUDIO_PORT", "8095"))

PRIMARY_REGION_V1 = "Production Infrastructure Spec (v1): Our primary production cloud region is us-east-1."
PRIMARY_REGION_V2 = "Postmortem 42 Migration Update: Our primary production cloud region is eu-central-1."


class StudioWorkspace:
    """Manages an isolated interactive workspace inside live Postgres, Valkey, and OpenFGA."""

    def __init__(self) -> None:
        self.master_seed = b"convergent-studio-seed"
        self.valkey = redis.Redis.from_url(VALKEY_URL, decode_responses=True)
        self.http = httpx.Client(timeout=90.0)
        self.store_id = resolve_openfga_store_id(self.http, OPENFGA_URL, "convergent")
        self.ollama_url = DEFAULT_OLLAMA_URL.rstrip("/")
        self.available_models: list[dict[str, Any]] = []
        self.ollama_connected = False
        self.ollama_error: str | None = None
        self.model_id = DEFAULT_MODEL_ID
        self.refresh_ollama_models(self.ollama_url)
        self.org_id = uuid.uuid4()
        self.spaces: dict[str, uuid.UUID] = {}
        self.principals: dict[str, dict[str, Any]] = {}
        self.reset()

    def refresh_ollama_models(self, url: str | None = None) -> dict[str, Any]:
        target = (url or self.ollama_url).strip().rstrip("/")
        if not target:
            target = "http://127.0.0.1:11434"
        try:
            resp = self.http.get(f"{target}/api/tags", timeout=5.0)
            resp.raise_for_status()
            raw_models = resp.json().get("models") or []
            models: list[dict[str, Any]] = []
            for m in raw_models:
                details = m.get("details") or {}
                size_bytes = int(m.get("size") or 0)
                models.append({
                    "name": str(m.get("name") or m.get("model") or ""),
                    "size_gb": round(size_bytes / (1024 ** 3), 1) if size_bytes else 0.0,
                    "parameter_size": str(details.get("parameter_size") or ""),
                    "quantization": str(details.get("quantization_level") or ""),
                    "family": str(details.get("family") or ""),
                })
            models = [m for m in models if m["name"]]
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
        return {
            "ollama_url": self.ollama_url,
            "connected": self.ollama_connected,
            "error": self.ollama_error,
            "model_id": self.model_id,
            "models": self.available_models,
        }

    def configure_ollama(self, *, ollama_url: str | None = None, model_id: str | None = None) -> dict[str, Any]:
        if ollama_url is not None and ollama_url.strip():
            self.refresh_ollama_models(ollama_url.strip())
        if model_id is not None and model_id.strip():
            self.model_id = model_id.strip()
        return self.get_state()

    def _fga_write(self, user: str, relation: str, obj: str, delete: bool = False) -> None:
        key = "deletes" if delete else "writes"
        resp = self.http.post(
            f"{OPENFGA_URL.rstrip('/')}/stores/{self.store_id}/write",
            json={key: {"tuple_keys": [{"user": user, "relation": relation, "object": obj}]}},
        )
        if resp.status_code == 400 and ("already exists" in resp.text.lower() or "does not exist" in resp.text.lower()):
            return
        resp.raise_for_status()

    def _fga_allowed_docs(self, user_id: str) -> list[uuid.UUID]:
        resp = self.http.post(
            f"{OPENFGA_URL.rstrip('/')}/stores/{self.store_id}/list-objects",
            json={"user": f"user:{user_id}", "relation": "viewer", "type": "document"},
        )
        resp.raise_for_status()
        out: list[uuid.UUID] = []
        for obj in resp.json().get("objects", []):
            if obj.startswith("document:"):
                try:
                    out.append(uuid.UUID(obj.split(":", 1)[1]))
                except ValueError:
                    pass
        return out

    def _sync_space_docs_to_openfga(self, conn: psycopg.Connection, space_id: uuid.UUID) -> None:
        rows = conn.execute(
            "SELECT id FROM documents WHERE org_id = %s AND space_id = %s",
            (self.org_id, space_id),
        ).fetchall()
        for (doc_id,) in rows:
            register_document_openfga(
                self.http,
                OPENFGA_URL,
                self.store_id,
                document_id=doc_id,
                space_id=space_id,
                org_id=self.org_id,
            )

    def reset(self) -> dict[str, Any]:
        self.org_id = uuid.uuid4()
        suffix = uuid.uuid4().hex[:6]
        self.spaces = {"finance": uuid.uuid4(), "engineering": uuid.uuid4()}
        self.principals = {
            "alice": {"id": f"alice-{suffix}", "label": "Alice (Finance)", "space": "finance", "active": True},
            "carol": {"id": f"carol-{suffix}", "label": "Carol (Finance)", "space": "finance", "active": True},
            "bob": {"id": f"bob-{suffix}", "label": "Bob (Engineering)", "space": "engineering", "active": True},
        }

        for p in self.principals.values():
            sid = self.spaces[p["space"]]
            self._fga_write(f"user:{p['id']}", "viewer", f"space:{sid}")

        with psycopg.connect(PG_DSN, autocommit=True) as conn:
            ov_f1 = open_overlay(conn, org_id=self.org_id, space_id=self.spaces["finance"], owner="alice")
            stage_delta(conn, overlay_id=ov_f1.overlay_id, source_uri="policy://travel-policy.md", text=TRAVEL_POLICY_V1)
            stage_delta(conn, overlay_id=ov_f1.overlay_id, source_uri="finance://q4-helios-budget.md", text=FINANCE_HELIOS_BUDGET)
            stage_delta(conn, overlay_id=ov_f1.overlay_id, source_uri="secret://helios-code.md", text=IDENTICAL_SECRET_TEXT)
            land_overlay(conn, overlay_id=ov_f1.overlay_id, author="alice", message="Initial Finance baseline (Travel v1 = $500)")

            ov_f2 = open_overlay(conn, org_id=self.org_id, space_id=self.spaces["finance"], owner="alice")
            stage_delta(conn, overlay_id=ov_f2.overlay_id, source_uri="policy://travel-policy.md", text=TRAVEL_POLICY_V2)
            land_overlay(conn, overlay_id=ov_f2.overlay_id, author="cfo-memo", message="Q4 Update: Raise travel cap to $1,200")
            self._sync_space_docs_to_openfga(conn, self.spaces["finance"])

            ov_e1 = open_overlay(conn, org_id=self.org_id, space_id=self.spaces["engineering"], owner="bob")
            stage_delta(conn, overlay_id=ov_e1.overlay_id, source_uri="infra://primary-region.md", text=PRIMARY_REGION_V1)
            stage_delta(conn, overlay_id=ov_e1.overlay_id, source_uri="eng://helios-architecture.md", text=ENG_HELIOS_ARCH)
            stage_delta(conn, overlay_id=ov_e1.overlay_id, source_uri="secret://helios-code.md", text=IDENTICAL_SECRET_TEXT)
            land_overlay(conn, overlay_id=ov_e1.overlay_id, author="bob", message="Initial Engineering baseline (us-east-1)")

            ov_e2 = open_overlay(conn, org_id=self.org_id, space_id=self.spaces["engineering"], owner="bob")
            stage_delta(conn, overlay_id=ov_e2.overlay_id, source_uri="infra://primary-region.md", text=PRIMARY_REGION_V2)
            land_overlay(conn, overlay_id=ov_e2.overlay_id, author="postmortem-42", message="Migrate primary prod region to eu-central-1")
            self._sync_space_docs_to_openfga(conn, self.spaces["engineering"])

        return self.get_state()

    def get_state(self) -> dict[str, Any]:
        with psycopg.connect(PG_DSN, autocommit=True) as conn:
            spaces_state: dict[str, Any] = {}
            for name, sid in self.spaces.items():
                head_rev = current_head_rev(conn, org_id=self.org_id, space_id=sid)
                active_blocks = read_mount_view(conn, org_id=self.org_id, space_id=sid)
                revs = conn.execute(
                    """
                    SELECT rev, author, message, action, created_at
                    FROM space_revisions
                    WHERE org_id = %s AND space_id = %s
                    ORDER BY rev DESC
                    LIMIT 12
                    """,
                    (self.org_id, sid),
                ).fetchall()
                ovs = conn.execute(
                    """
                    SELECT id, base_rev, owner, status, landed_rev
                    FROM overlays
                    WHERE org_id = %s AND space_id = %s AND status = 'open'
                    ORDER BY created_at DESC
                    """,
                    (self.org_id, sid),
                ).fetchall()
                open_overlays = []
                for oid, base_rev, owner, status, landed_rev in ovs:
                    deltas = conn.execute(
                        "SELECT op, source_uri, canonical_text FROM overlay_deltas WHERE overlay_id = %s ORDER BY source_uri",
                        (oid,),
                    ).fetchall()
                    open_overlays.append({
                        "overlay_id": str(oid),
                        "base_rev": int(base_rev),
                        "owner": owner,
                        "status": status,
                        "landed_rev": int(landed_rev) if landed_rev is not None else None,
                        "deltas": [{"op": op, "source_uri": uri, "canonical_text": txt} for op, uri, txt in deltas],
                    })
                spaces_state[name] = {
                    "space_id": str(sid),
                    "head_rev": head_rev,
                    "active_blocks": active_blocks,
                    "revisions": [
                        {"rev": int(r), "author": a, "message": m, "action": act, "created_at": ts.isoformat()}
                        for r, a, m, act, ts in revs
                    ],
                    "open_overlays": open_overlays,
                }
        return {
            "org_id": str(self.org_id),
            "ollama": {
                "url": self.ollama_url,
                "connected": self.ollama_connected,
                "error": self.ollama_error,
                "model_id": self.model_id,
                "models": self.available_models,
            },
            "principals": self.principals,
            "spaces": spaces_state,
        }

    def toggle_principal(self, persona: str) -> dict[str, Any]:
        p = self.principals.get(persona)
        if not p:
            raise ValueError(f"unknown persona: {persona}")
        sid = self.spaces[p["space"]]
        new_active = not p["active"]
        self._fga_write(f"user:{p['id']}", "viewer", f"space:{sid}", delete=not new_active)
        p["active"] = new_active
        return self.get_state()

    def api_open_overlay(self, space_name: str, owner: str, base_rev: int | None = None) -> dict[str, Any]:
        sid = self.spaces[space_name]
        with psycopg.connect(PG_DSN, autocommit=True) as conn:
            ov = open_overlay(conn, org_id=self.org_id, space_id=sid, owner=owner, base_rev=base_rev)
        return {"overlay_id": str(ov.overlay_id), "base_rev": ov.base_rev, "owner": ov.owner, "state": self.get_state()}

    def api_stage_delta(self, overlay_id: str, source_uri: str, text: str, op: str = "upsert") -> dict[str, Any]:
        with psycopg.connect(PG_DSN, autocommit=True) as conn:
            did = stage_delta(conn, overlay_id=uuid.UUID(overlay_id), source_uri=source_uri, text=text, op=op)
        return {"delta_id": str(did), "state": self.get_state()}

    def api_land_overlay(self, overlay_id: str, author: str, message: str, force: bool = False) -> dict[str, Any]:
        oid = uuid.UUID(overlay_id)
        with psycopg.connect(PG_DSN, autocommit=True) as conn:
            row = conn.execute("SELECT space_id FROM overlays WHERE id = %s", (oid,)).fetchone()
            if row is None:
                raise ValueError("overlay not found")
            sid = row[0]
            res = land_overlay(conn, overlay_id=oid, author=author, message=message, force=force)
            if res.status == "landed":
                self._sync_space_docs_to_openfga(conn, sid)
                for old_id in res.superseded_block_ids:
                    hrow = conn.execute("SELECT content_blake3 FROM knowledge_blocks WHERE id = %s", (old_id,)).fetchone()
                    if hrow:
                        evict_block_kv(self.valkey, hrow[0])
        return {
            "status": res.status,
            "rev": res.rev,
            "applied_count": len(res.applied_block_ids),
            "superseded_count": len(res.superseded_block_ids),
            "conflicts": res.conflicts,
            "state": self.get_state(),
        }

    def api_undo_revision(self, space_name: str, rev: int, author: str) -> dict[str, Any]:
        sid = self.spaces[space_name]
        with psycopg.connect(PG_DSN, autocommit=True) as conn:
            res = undo_revision(conn, org_id=self.org_id, space_id=sid, rev=rev, author=author)
            for bid in res.deactivated_block_ids:
                hrow = conn.execute("SELECT content_blake3 FROM knowledge_blocks WHERE id = %s", (bid,)).fetchone()
                if hrow:
                    evict_block_kv(self.valkey, hrow[0])
        return {
            "undo_rev": res.undo_rev,
            "undone_rev": res.undone_rev,
            "deactivated": len(res.deactivated_block_ids),
            "restored": len(res.restored_block_ids),
            "state": self.get_state(),
        }

    def _call_ollama(self, context_blocks: list[str], prompt: str, model_id: str) -> tuple[str, float]:
        if not model_id:
            return "[No Ollama model selected — configure Ollama in the top bar.]", 0.0
        t0 = time.perf_counter()
        ctx = "\n".join(f"- {b}" for b in context_blocks) if context_blocks else "(No authorized context available.)"
        try:
            resp = self.http.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": model_id,
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0.0, "num_predict": 160},
                    "messages": [
                        {
                            "role": "system",
                            "content": "Answer concisely in 1-2 sentences using ONLY the provided context blocks. If no relevant context is provided, say you do not have access to that information.",
                        },
                        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {prompt}"},
                    ],
                },
            )
            resp.raise_for_status()
            ans = str(resp.json()["message"]["content"]).strip()
        except Exception as exc:
            ans = f"[Ollama error on {model_id} @ {self.ollama_url}: {exc}]"
        return ans, round((time.perf_counter() - t0) * 1000.0, 1)

    def run_arena(
        self,
        *,
        persona: str,
        prompt: str,
        overlay_id: str | None = None,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        p = self.principals.get(persona)
        if not p:
            raise ValueError(f"unknown persona: {persona}")
        active_model = (model_id or self.model_id).strip()
        space_name = p["space"]
        sid = self.spaces[space_name]

        with psycopg.connect(PG_DSN, autocommit=True) as conn:
            rag_rows = conn.execute(
                """
                SELECT d.source_uri, kb.canonical_text, kb.valid_to IS NOT NULL AS is_stale, kb.space_id <> %s AS cross_space
                FROM knowledge_blocks kb
                JOIN documents d ON d.id = kb.document_id
                WHERE kb.org_id = %s
                ORDER BY ts_rank_cd(to_tsvector('english', kb.canonical_text), plainto_tsquery('english', %s)) DESC,
                         kb.valid_from ASC
                LIMIT 6
                """,
                (sid, self.org_id, prompt),
            ).fetchall()
            rag_blocks = [
                {"source_uri": uri, "text": txt, "is_stale": bool(stale), "cross_space": bool(cross)}
                for uri, txt, stale, cross in rag_rows
            ]

            allowed_docs = set(self._fga_allowed_docs(p["id"]))
            oid = uuid.UUID(overlay_id) if overlay_id else None
            view = read_mount_view(conn, org_id=self.org_id, space_id=sid, overlay_id=oid)

            doc_map = dict(
                conn.execute(
                    "SELECT id::text, document_id FROM knowledge_blocks WHERE org_id = %s AND space_id = %s AND valid_to IS NULL",
                    (self.org_id, sid),
                ).fetchall()
            )
            conv_blocks: list[dict[str, Any]] = []
            kv_hits = 0
            for blk in view:
                if blk["layer"] == "base":
                    doc_id = doc_map.get(blk["id"])
                    if doc_id not in allowed_docs:
                        continue
                kv_key = derive_kv_key(self.master_seed, blk["clearance_scope"], active_model or "default", blk["content_blake3"])
                if self.valkey.exists(kv_key):
                    kv_hits += 1
                else:
                    register_kv_key(self.valkey, blk["content_blake3"], kv_key)
                conv_blocks.append({
                    "source_uri": blk["source_uri"],
                    "text": blk["canonical_text"],
                    "layer": blk["layer"],
                    "clearance_scope": blk["clearance_scope"],
                    "kv_key_prefix": kv_key[:24] + "...",
                    "citations": blk["citations"],
                })

        rag_answer, rag_ms = self._call_ollama([b["text"] for b in rag_blocks], prompt, active_model)
        conv_answer, conv_ms = self._call_ollama([b["text"] for b in conv_blocks], prompt, active_model)

        return {
            "persona": p,
            "model_id": active_model,
            "prompt": prompt,
            "overlay_id": overlay_id,
            "standard_rag": {
                "answer": rag_answer,
                "latency_ms": rag_ms,
                "stale_blocks_leaked": sum(1 for b in rag_blocks if b["is_stale"]),
                "cross_space_leaked": sum(1 for b in rag_blocks if b["cross_space"]),
                "blocks": rag_blocks,
            },
            "convergent": {
                "answer": conv_answer,
                "latency_ms": conv_ms,
                "kv_cache_hits": kv_hits,
                "kv_total_blocks": len(conv_blocks),
                "stale_blocks_leaked": 0,
                "cross_space_leaked": 0,
                "blocks": conv_blocks,
            },
        }


def make_studio_handler(ws: StudioWorkspace) -> type[BaseHTTPRequestHandler]:
    html_bytes = (ROOT / "studio" / "index.html").read_bytes()

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
            elif parsed.path == "/api/state":
                self._json(200, ws.get_state())
            elif parsed.path == "/api/ollama/models":
                qs = parse_qs(parsed.query)
                url = (qs.get("url") or [None])[0]
                self._json(200, ws.refresh_ollama_models(url))
            else:
                self._json(404, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                body = self._body()
                if self.path == "/api/reset":
                    self._json(200, ws.reset())
                elif self.path == "/api/config":
                    self._json(
                        200,
                        ws.configure_ollama(
                            ollama_url=body.get("ollama_url"),
                            model_id=body.get("model_id"),
                        ),
                    )
                elif self.path == "/api/arena":
                    self._json(
                        200,
                        ws.run_arena(
                            persona=body.get("persona", "alice"),
                            prompt=body.get("prompt", ""),
                            overlay_id=body.get("overlay_id") or None,
                            model_id=body.get("model_id") or None,
                        ),
                    )
                elif self.path == "/api/principals/toggle":
                    self._json(200, ws.toggle_principal(body["persona"]))
                elif self.path == "/api/overlays/open":
                    self._json(
                        200,
                        ws.api_open_overlay(
                            space_name=body.get("space", "finance"),
                            owner=body.get("owner", "agent-1"),
                            base_rev=body.get("base_rev"),
                        ),
                    )
                elif self.path == "/api/overlays/stage":
                    self._json(
                        200,
                        ws.api_stage_delta(
                            overlay_id=body["overlay_id"],
                            source_uri=body["source_uri"],
                            text=body.get("text", ""),
                            op=body.get("op", "upsert"),
                        ),
                    )
                elif self.path == "/api/overlays/land":
                    self._json(
                        200,
                        ws.api_land_overlay(
                            overlay_id=body["overlay_id"],
                            author=body.get("author", "agent-1"),
                            message=body.get("message", "Land overlay"),
                            force=bool(body.get("force", False)),
                        ),
                    )
                elif self.path == "/api/revisions/undo":
                    self._json(
                        200,
                        ws.api_undo_revision(
                            space_name=body["space"],
                            rev=int(body["rev"]),
                            author=body.get("author", "studio-user"),
                        ),
                    )
                else:
                    self._json(404, {"error": "not_found"})
            except Exception as exc:
                self._json(400, {"error": str(exc)})

    return Handler


def main() -> None:
    ws = StudioWorkspace()
    server = ThreadingHTTPServer((STUDIO_HOST, STUDIO_PORT), make_studio_handler(ws))
    print(f"\n  Project Convergent Studio -> http://{STUDIO_HOST}:{STUDIO_PORT}")
    print(f"  Ollama: {ws.ollama_url} | Active Model: {ws.model_id or '(none detected)'} ({len(ws.available_models)} available)\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
