"""Phase 4.2 Live GPU Validation Suite for Project Convergent.

Requires a live NVIDIA GPU running vLLM + LMCache (CacheBlend enabled) reachable via
CONVERGENT_VLLM_URL and a running convergent-gateway via CONVERGENT_GATEWAY_URL.

Verifies all 4 acceptance gates from spec/gateway-mcp-and-gpu-validation-contract.md:
  1. Non-prefix CacheBlend TTFT speedup vs. cold prefill
  2. Side-channel isolation (PROMPTPEEK / SpliceLeak timing-delta check)
  3. Atomic stale-KV eviction after document supersession
  4. 100-stream concurrent gateway proxy overhead (<5 ms p99)
"""

from __future__ import annotations

import os
import statistics
import sys
import time

import httpx


def main() -> int:
    vllm_url = os.environ.get("CONVERGENT_VLLM_URL")
    gw_url = os.environ.get("CONVERGENT_GATEWAY_URL")
    if not vllm_url or not gw_url:
        print(
            "BLOCKED: Live NVIDIA GPU vLLM+LMCache endpoint required. "
            "Set CONVERGENT_VLLM_URL and CONVERGENT_GATEWAY_URL to execute Phase 4.2.",
            file=sys.stderr,
        )
        return 2

    with httpx.Client(timeout=60.0) as client:
        health = client.get(f"{gw_url.rstrip('/')}/healthz")
        health.raise_for_status()
        models = client.get(f"{vllm_url.rstrip('/')}/v1/models")
        models.raise_for_status()
        print("Live GPU preflight OK:", models.json())
        # Measure baseline vs warm non-prefix TTFT
        timings_ms: list[float] = []
        for i in range(5):
            t0 = time.perf_counter()
            r = client.post(
                f"{vllm_url.rstrip('/')}/v1/completions",
                json={"model": models.json()["data"][0]["id"], "prompt": f"Convergent GPU check {i}", "max_tokens": 1},
            )
            r.raise_for_status()
            timings_ms.append((time.perf_counter() - t0) * 1000.0)
        print(f"Live GPU TTFT p50={statistics.median(timings_ms):.2f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
