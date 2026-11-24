"""Inference Backend Factory for Project Convergent workers.

Allows running prefill warming and model swaps against CUDA (vLLM, SGLang),
Apple Silicon (MLX, Ollama), or local dev simulators via CONVERGENT_INFERENCE_BACKEND.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import os

import httpx


class InferenceBackend(ABC):
    name: str
    hardware_target: str
    supports_native_kv_keys: bool

    @abstractmethod
    def prewarm(
        self,
        http: httpx.Client,
        endpoint_url: str,
        model_id: str,
        block_text: str,
        kv_key: str,
    ) -> None:
        """Warm KV cache for `block_text` on the target inference engine."""


class VllmBackend(InferenceBackend):
    name = "vllm"
    hardware_target = "cuda"
    supports_native_kv_keys = True

    def prewarm(self, http: httpx.Client, endpoint_url: str, model_id: str, block_text: str, kv_key: str) -> None:
        resp = http.post(
            f"{endpoint_url.rstrip('/')}/v1/completions",
            json={
                "model": model_id,
                "prompt": block_text,
                "max_tokens": 1,
                "temperature": 0.0,
                "extra_body": {"kv_cache_key": kv_key},
            },
        )
        resp.raise_for_status()


class SglangBackend(InferenceBackend):
    name = "sglang"
    hardware_target = "cuda"
    supports_native_kv_keys = True

    def prewarm(self, http: httpx.Client, endpoint_url: str, model_id: str, block_text: str, kv_key: str) -> None:
        resp = http.post(
            f"{endpoint_url.rstrip('/')}/v1/completions",
            json={
                "model": model_id,
                "prompt": block_text,
                "max_tokens": 1,
                "temperature": 0.0,
                "extra_body": {"radix_cache_key": kv_key, "kv_cache_key": kv_key},
            },
        )
        resp.raise_for_status()


class MlxBackend(InferenceBackend):
    """Apple Silicon Metal backend (mlx_lm.server OpenAI-compatible API)."""

    name = "mlx"
    hardware_target = "apple-silicon-metal"
    supports_native_kv_keys = False

    def prewarm(self, http: httpx.Client, endpoint_url: str, model_id: str, block_text: str, kv_key: str) -> None:
        resp = http.post(
            f"{endpoint_url.rstrip('/')}/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "system", "content": block_text}, {"role": "user", "content": "ack"}],
                "max_tokens": 1,
                "temperature": 0.0,
            },
        )
        resp.raise_for_status()


class OllamaBackend(InferenceBackend):
    """Ollama backend (Apple Silicon Metal or CUDA) using /api/generate with keep_alive."""

    name = "ollama"
    hardware_target = "metal-or-cuda"
    supports_native_kv_keys = False

    def prewarm(self, http: httpx.Client, endpoint_url: str, model_id: str, block_text: str, kv_key: str) -> None:
        resp = http.post(
            f"{endpoint_url.rstrip('/')}/api/generate",
            json={
                "model": model_id,
                "prompt": block_text,
                "stream": False,
                "keep_alive": "10m",
                "options": {"num_predict": 1, "temperature": 0.0},
            },
        )
        resp.raise_for_status()


class OpenAiProxyBackend(InferenceBackend):
    """Cloud OpenAI proxy backend (uses prompt_cache_key; prewarm is a no-op to avoid cloud token spend)."""

    name = "openai"
    hardware_target = "cloud-proxy"
    supports_native_kv_keys = False

    def prewarm(self, http: httpx.Client, endpoint_url: str, model_id: str, block_text: str, kv_key: str) -> None:
        return


class AnthropicProxyBackend(InferenceBackend):
    """Cloud Anthropic proxy backend (uses ephemeral cache_control; prewarm is a no-op to avoid cloud token spend)."""

    name = "anthropic"
    hardware_target = "cloud-proxy"
    supports_native_kv_keys = False

    def prewarm(self, http: httpx.Client, endpoint_url: str, model_id: str, block_text: str, kv_key: str) -> None:
        return


class DevBackend(VllmBackend):
    name = "dev"
    hardware_target = "local-simulator"


_REGISTRY: dict[str, type[InferenceBackend]] = {
    "vllm": VllmBackend,
    "cuda": VllmBackend,
    "vllm-cuda": VllmBackend,
    "sglang": SglangBackend,
    "sglang-cuda": SglangBackend,
    "mlx": MlxBackend,
    "apple-silicon": MlxBackend,
    "metal": MlxBackend,
    "ollama": OllamaBackend,
    "openai": OpenAiProxyBackend,
    "openai-proxy": OpenAiProxyBackend,
    "anthropic": AnthropicProxyBackend,
    "claude": AnthropicProxyBackend,
    "dev": DevBackend,
    "simulator": DevBackend,
}


def create_backend(name: str | None = None) -> InferenceBackend:
    key = (name or os.environ.get("CONVERGENT_INFERENCE_BACKEND") or "vllm").strip().lower()
    cls = _REGISTRY.get(key)
    if cls is None:
        supported = ", ".join(sorted({"vllm (cuda)", "sglang", "mlx", "ollama", "dev"}))
        raise ValueError(f"Unsupported CONVERGENT_INFERENCE_BACKEND '{key}'; expected one of: {supported}")
    return cls()
