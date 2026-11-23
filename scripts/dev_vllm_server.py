"""Local Apple Silicon dev OpenAI-compatible inference server simulating vLLM + LMCache KV tracking.

Zero extra dependencies (uses Python stdlib http.server + redis + httpx from worker-py).
Optionally forwards generation to a real local Ollama or mlx_lm.server when
CONVERGENT_UPSTREAM_LLM_URL is set.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import time
from typing import Any

import httpx
import redis

VALKEY_URL = os.environ.get("CONVERGENT_VALKEY_URL", "redis://127.0.0.1:56379/0")
MODEL_ID = os.environ.get("CONVERGENT_MODEL_ID", "Qwen/Qwen3-32B-FP8")
BIND_HOST = os.environ.get("CONVERGENT_DEV_VLLM_HOST", "127.0.0.1")
BIND_PORT = int(os.environ.get("CONVERGENT_DEV_VLLM_PORT", "8000"))
UPSTREAM_LLM_URL = os.environ.get("CONVERGENT_UPSTREAM_LLM_URL", "").strip()


def make_handler(valkey_url: str, model_id: str, upstream_llm_url: str = "") -> type[BaseHTTPRequestHandler]:
    valkey = redis.Redis.from_url(valkey_url, decode_responses=True)

    class DevVllmHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send_json(self, status: int, body: dict[str, Any], extra_headers: dict[str, str] | None = None) -> None:
            raw = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            for k, v in (extra_headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send_json(200, {"status": "ok", "mode": "apple-silicon-dev-simulator"})
            elif self.path == "/v1/models":
                self._send_json(
                    200,
                    {"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "convergent-dev"}]},
                )
            else:
                self._send_json(404, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}

            if self.path == "/v1/completions":
                extra = body.get("extra_body") or {}
                kv_key = extra.get("kv_cache_key")
                if kv_key:
                    valkey.set(kv_key, "dev-warm", ex=3600)
                self._send_json(
                    200,
                    {
                        "id": f"cmpl-dev-{int(time.time() * 1000)}",
                        "object": "text_completion",
                        "model": body.get("model", model_id),
                        "choices": [{"index": 0, "text": "", "finish_reason": "length"}],
                    },
                )
                return

            if self.path == "/v1/chat/completions":
                extra = body.get("extra_body") or {}
                kv_keys = extra.get("kv_cache_keys") or []
                hits = 0
                for key in kv_keys:
                    if valkey.exists(key):
                        hits += 1
                    else:
                        valkey.set(key, "dev-warm", ex=3600)

                if upstream_llm_url:
                    with httpx.Client(timeout=60.0) as client:
                        up = client.post(
                            f"{upstream_llm_url.rstrip('/')}/v1/chat/completions",
                            json={k: v for k, v in body.items() if k != "extra_body"},
                        )
                    self.send_response(up.status_code)
                    self.send_header("Content-Type", up.headers.get("content-type", "application/json"))
                    self.send_header("x-convergent-dev-cache-hits", str(hits))
                    self.end_headers()
                    self.wfile.write(up.content)
                    return

                messages = body.get("messages") or []
                user_msg = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
                sys_msg = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
                reply = (
                    f"[convergent-dev | kv_hits={hits}/{len(kv_keys)}] "
                    f"Context blocks loaded: {sys_msg.count('[BLOCK ')}. Query: {user_msg}"
                )

                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("x-convergent-dev-cache-hits", str(hits))
                    self.end_headers()
                    chunk = {
                        "id": f"chatcmpl-dev-{int(time.time() * 1000)}",
                        "object": "chat.completion.chunk",
                        "model": body.get("model", model_id),
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode("utf-8"))
                    self.wfile.flush()
                    return

                self._send_json(
                    200,
                    {
                        "id": f"chatcmpl-dev-{int(time.time() * 1000)}",
                        "object": "chat.completion",
                        "model": body.get("model", model_id),
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
                    },
                    extra_headers={"x-convergent-dev-cache-hits": str(hits)},
                )
                return

            self._send_json(404, {"error": "not_found"})

    return DevVllmHandler


def main() -> None:
    server = ThreadingHTTPServer((BIND_HOST, BIND_PORT), make_handler(VALKEY_URL, MODEL_ID, UPSTREAM_LLM_URL))
    print(f"convergent dev vLLM simulator listening on http://{BIND_HOST}:{BIND_PORT} (model={MODEL_ID})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
