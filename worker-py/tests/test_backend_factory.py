"""Tests for the Inference Backend Factory (CUDA vLLM/SGLang, Apple Silicon MLX, Ollama, Dev)."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
from typing import Any
import uuid

import httpx
import pytest
import redis

from convergent_worker.backends import create_backend
from convergent_worker.kv_evictor import derive_kv_key, prewarm_block

VALKEY_URL = os.environ.get("CONVERGENT_VALKEY_URL", "redis://127.0.0.1:56379/0")


def test_factory_resolves_cuda_mlx_ollama_sglang_dev() -> None:
    assert create_backend("cuda").name == "vllm"
    assert create_backend("vllm").hardware_target == "cuda"
    assert create_backend("sglang").name == "sglang"
    assert create_backend("mlx").hardware_target == "apple-silicon-metal"
    assert create_backend("apple-silicon").name == "mlx"
    assert create_backend("ollama").name == "ollama"
    assert create_backend("dev").name == "dev"
    with pytest.raises(ValueError, match="Unsupported CONVERGENT_INFERENCE_BACKEND"):
        create_backend("nonexistent-engine")


def test_prewarm_dispatches_backend_specific_endpoints_and_tracks_valkey() -> None:
    seen: list[tuple[str, dict[str, Any]]] = []

    class CaptureHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            seen.append((self.path, body))
            raw = json.dumps({"ok": True}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    valkey = redis.Redis.from_url(VALKEY_URL, decode_responses=True)

    try:
        with httpx.Client(timeout=5.0) as http:
            for name, expected_path in [
                ("vllm", "/v1/completions"),
                ("sglang", "/v1/completions"),
                ("mlx", "/v1/chat/completions"),
                ("ollama", "/api/generate"),
            ]:
                backend = create_backend(name)
                bhash = f"hash-{name}-{uuid.uuid4().hex[:6]}"
                kv_key = derive_kv_key(b"factory-seed", "space:demo", f"model-{name}", bhash)
                prewarm_block(http, url, f"model-{name}", "Block content", kv_key, valkey, bhash, backend=backend)
                assert valkey.exists(kv_key) == 1
                path, payload = seen[-1]
                assert path == expected_path
                if name == "ollama":
                    assert payload["options"]["num_predict"] == 1
                    assert payload["keep_alive"] == "10m"
                elif name == "mlx":
                    assert "extra_body" not in payload
                    assert payload["max_tokens"] == 1
                else:
                    assert payload["extra_body"]["kv_cache_key"] == kv_key
    finally:
        server.shutdown()
        server.server_close()
        valkey.close()
