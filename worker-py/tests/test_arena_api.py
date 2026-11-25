"""Integration test for Convergent Arena single-page showcase & SSE streaming benchmark."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "arena"))

from server import ArenaEngine, make_arena_handler  # noqa: E402


def _start_mock_ollama() -> tuple[ThreadingHTTPServer, str]:
    class MockOllama(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send(self, obj: Any) -> None:
            raw = json.dumps(obj).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/api/tags":
                self._send({
                    "models": [
                        {"name": "mock-small:3b", "size": 2_200_000_000, "details": {"parameter_size": "3B", "quantization_level": "Q4_K_M"}},
                        {"name": "mock-large:27b", "size": 18_000_000_000, "details": {"parameter_size": "27B", "quantization_level": "Q4_K_M"}},
                    ]
                })

        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
            user_msg = body.get("messages", [{}])[-1].get("content", "")
            self._send({"message": {"role": "assistant", "content": f"mock-ok ({len(user_msg)} chars)"}})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), MockOllama)
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, url


def test_arena_config_and_full_showcase_stream() -> None:
    mock_srv, mock_ollama_url = _start_mock_ollama()
    engine = ArenaEngine()
    engine.configure(ollama_url=mock_ollama_url)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_arena_handler(engine))
    url = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()

    try:
        with httpx.Client(timeout=30.0) as http:
            # 1. Single-page HTML served
            html = http.get(f"{url}/")
            assert html.status_code == 200
            assert "Project Convergent" in html.text
            assert "Run Full Showcase" in html.text
            assert "Configure Ollama" in html.text

            # 2. Ollama discovery sorts smallest-first & allows switching models
            cfg = http.get(f"{url}/api/ollama/models?url={mock_ollama_url}").raise_for_status().json()
            assert cfg["connected"] is True
            assert cfg["models"][0]["name"] == "mock-small:3b"
            updated = http.post(f"{url}/api/config", json={"model_id": "mock-large:27b"}).raise_for_status().json()
            assert updated["model_id"] == "mock-large:27b"

            # 3. Single-button SSE showcase streams all 5 stages + real KPIs
            events = []
            with http.stream("GET", f"{url}/api/showcase/stream?model_id=mock-small:3b") as resp:
                assert resp.status_code == 200
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        events.append(json.loads(line[6:]))

            traces = [e for e in events if e.get("event") == "trace"]
            steps = [e for e in events if e.get("event") == "step"]
            assert len(traces) >= 45
            assert {t["stage"] for t in traces} == {1, 2, 3, 4, 5}
            assert len(steps) == 5
            complete = next(e for e in events if e.get("event") == "complete")
            kpis = complete["kpis"]
            assert kpis["convergent_leaks_total"] == 0
            assert kpis["rag_leaks_total"] > 0
            assert kpis["copied_rows_on_overlay"] == 0
            assert kpis["bob_docs_after_revoke"] == 0
            assert kpis["auto_rebase_ok"] is True
            assert kpis["conflict_caught"] is True
            assert kpis["undo_restored"] >= 1
            assert kpis["disjoint_hmac_keys"] is True
    finally:
        server.shutdown()
        server.server_close()
        mock_srv.shutdown()
        mock_srv.server_close()
