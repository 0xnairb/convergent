"""Apple Silicon Metal GPU (MLX) Tensor-Level KV-Block Cache & CacheBlend Splicer.

Implements:
  - Per-clearance HMAC-SHA256 KV key derivation (byte-identical to gateway-rs/src/kv_keys.rs)
  - Real Apple Silicon Metal GPU (`mlx.core`) multi-layer KV prefill & scaled dot-product attention
  - Non-prefix CacheBlend KV splicing with selective boundary token recomputation (~15%)
  - Two-tier storage (Unified Memory GPU tier + `.safetensors` disk tier) with O(k) physical eviction
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
from pathlib import Path
import time
from typing import Any

import mlx.core as mx


def derive_kv_key(master_seed: bytes, clearance_scope: str, model_id: str, block_hash: str) -> str:
    """Byte-identical to gateway-rs/src/kv_keys.rs::derive_kv_key."""
    scope_secret = hmac.new(master_seed, b"cvg:scope:v1:" + clearance_scope.encode("utf-8"), hashlib.sha256).digest()
    digest = hmac.new(scope_secret, f"{model_id}:{block_hash}".encode("utf-8"), hashlib.sha256).hexdigest()
    return f"cvg_kv_v1_{digest}"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BlockSpec:
    clearance_scope: str
    block_hash: str
    tokens: mx.array  # shape: [seq_len, hidden_dim], float16


@dataclass(frozen=True)
class BlendResult:
    output: mx.array
    cache_hits: int
    cache_misses: int
    recomputed_tokens: int
    total_context_tokens: int
    ttft_ms: float


class MlxKvCacheEngine:
    """Real Apple Silicon Metal GPU KV-block engine with per-clearance isolation and CacheBlend splicing."""

    def __init__(
        self,
        *,
        model_id: str,
        master_seed: bytes,
        storage_dir: Path,
        num_layers: int = 16,
        num_heads: int = 8,
        head_dim: int = 128,
        blend_recompute_ratio: float = 0.15,
    ) -> None:
        self.model_id = model_id
        self.master_seed = master_seed
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = num_heads * head_dim
        self.blend_recompute_ratio = blend_recompute_ratio

        # Deterministic model weights resident on Apple Silicon Metal GPU (float16)
        mx.random.seed(42)
        scale = (1.0 / self.hidden_dim) ** 0.5
        self.w_q = [mx.random.normal((self.hidden_dim, self.hidden_dim), dtype=mx.float16) * scale for _ in range(num_layers)]
        self.w_k = [mx.random.normal((self.hidden_dim, self.hidden_dim), dtype=mx.float16) * scale for _ in range(num_layers)]
        self.w_v = [mx.random.normal((self.hidden_dim, self.hidden_dim), dtype=mx.float16) * scale for _ in range(num_layers)]
        self.w_o = [mx.random.normal((self.hidden_dim, self.hidden_dim), dtype=mx.float16) * scale for _ in range(num_layers)]
        mx.eval(self.w_q, self.w_k, self.w_v, self.w_o)

        # Tier 1: Unified Memory GPU cache: kv_key -> list[(K_l, V_l)] per layer
        self._mem_tier: dict[str, list[tuple[mx.array, mx.array]]] = {}
        # Reverse index for O(k) physical eviction: block_hash -> set[kv_key]
        self._block_index: dict[str, set[str]] = {}

    def _disk_path(self, kv_key: str) -> Path:
        return self.storage_dir / f"{kv_key}.safetensors"

    def _prefill_block_gpu(self, tokens: mx.array) -> list[tuple[mx.array, mx.array]]:
        """Full L-layer Transformer prefill on Metal GPU producing per-layer (K, V) tensors."""
        seq_len = tokens.shape[0]
        h = tokens
        kv_layers: list[tuple[mx.array, mx.array]] = []
        attn_scale = self.head_dim ** -0.5

        for layer in range(self.num_layers):
            q = (h @ self.w_q[layer]).reshape(1, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            k = (h @ self.w_k[layer]).reshape(1, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            v = (h @ self.w_v[layer]).reshape(1, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn_scale, mask="causal")
            attn_flat = attn.transpose(0, 2, 1, 3).reshape(seq_len, self.hidden_dim)
            h = h + (attn_flat @ self.w_o[layer])
            kv_layers.append((k, v))

        mx.eval([t for pair in kv_layers for t in pair])
        return kv_layers

    def _save_block(self, block_hash: str, kv_key: str, kv_layers: list[tuple[mx.array, mx.array]]) -> None:
        self._mem_tier[kv_key] = kv_layers
        self._block_index.setdefault(block_hash, set()).add(kv_key)
        tensors: dict[str, mx.array] = {}
        for i, (k, v) in enumerate(kv_layers):
            tensors[f"k_{i}"] = k
            tensors[f"v_{i}"] = v
        mx.save_safetensors(str(self._disk_path(kv_key)), tensors)

    def _load_block(self, kv_key: str) -> list[tuple[mx.array, mx.array]] | None:
        if kv_key in self._mem_tier:
            return self._mem_tier[kv_key]
        path = self._disk_path(kv_key)
        if path.exists():
            loaded = mx.load(str(path))
            kv_layers = [(loaded[f"k_{i}"], loaded[f"v_{i}"]) for i in range(self.num_layers)]
            self._mem_tier[kv_key] = kv_layers
            return kv_layers
        return None

    def evict_block(self, block_hash: str) -> dict[str, int]:
        """Physically evict all Unified Memory tensors and `.safetensors` files for a superseded block."""
        keys = list(self._block_index.pop(block_hash, set()))
        mem_evicted = 0
        disk_evicted = 0
        for kv_key in keys:
            if kv_key in self._mem_tier:
                del self._mem_tier[kv_key]
                mem_evicted += 1
            path = self._disk_path(kv_key)
            if path.exists():
                path.unlink()
                disk_evicted += 1
        return {"mem_evicted": mem_evicted, "disk_evicted": disk_evicted}

    def compose_and_attend(
        self,
        blocks: list[BlockSpec],
        query_tokens: mx.array,
        *,
        use_cache: bool = True,
    ) -> BlendResult:
        """Compose authorized Knowledge Blocks (including non-prefix subsets) with CacheBlend splicing."""
        t0 = time.perf_counter()
        hits = 0
        misses = 0
        per_block_kv: list[list[tuple[mx.array, mx.array]]] = []
        total_ctx_tokens = sum(int(b.tokens.shape[0]) for b in blocks)
        recomputed_tokens = 0

        if not use_cache:
            # Cold baseline: concatenate all raw tokens and run full L-layer prefill from scratch
            full_ctx = mx.concatenate([b.tokens for b in blocks], axis=0)
            full_kv = self._prefill_block_gpu(full_ctx)
            misses = len(blocks)
            recomputed_tokens = total_ctx_tokens
            spliced_kv = full_kv
        else:
            for b in blocks:
                kv_key = derive_kv_key(self.master_seed, b.clearance_scope, self.model_id, b.block_hash)
                cached = self._load_block(kv_key)
                if cached is not None:
                    hits += 1
                    per_block_kv.append(cached)
                else:
                    misses += 1
                    recomputed_tokens += int(b.tokens.shape[0])
                    computed = self._prefill_block_gpu(b.tokens)
                    self._save_block(b.block_hash, kv_key, computed)
                    per_block_kv.append(computed)

            # Splice per-block (K, V) tensors along sequence dimension for each layer
            spliced_kv = [
                (
                    mx.concatenate([blk[l][0] for blk in per_block_kv], axis=2),
                    mx.concatenate([blk[l][1] for blk in per_block_kv], axis=2),
                )
                for l in range(self.num_layers)
            ]

            # CacheBlend selective boundary recomputation (~15% boundary tokens) when splicing >=2 cached blocks
            if hits >= 2 and self.blend_recompute_ratio > 0:
                boundary_n = max(1, int(total_ctx_tokens * self.blend_recompute_ratio))
                boundary_slice = mx.concatenate([b.tokens for b in blocks], axis=0)[-boundary_n:]
                recomputed_kv = self._prefill_block_gpu(boundary_slice)
                spliced_kv = [
                    (
                        mx.concatenate([spliced_kv[l][0][:, :, :-boundary_n, :], recomputed_kv[l][0]], axis=2),
                        mx.concatenate([spliced_kv[l][1][:, :, :-boundary_n, :], recomputed_kv[l][1]], axis=2),
                    )
                    for l in range(self.num_layers)
                ]
                recomputed_tokens += boundary_n

        # Attend query tokens against the composed KV cache on Metal GPU
        q_len = int(query_tokens.shape[0])
        h = query_tokens
        attn_scale = self.head_dim ** -0.5
        for l in range(self.num_layers):
            q = (h @ self.w_q[l]).reshape(1, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            k_q = (h @ self.w_k[l]).reshape(1, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            v_q = (h @ self.w_v[l]).reshape(1, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            k_all = mx.concatenate([spliced_kv[l][0], k_q], axis=2)
            v_all = mx.concatenate([spliced_kv[l][1], v_q], axis=2)
            attn = mx.fast.scaled_dot_product_attention(q, k_all, v_all, scale=attn_scale)
            h = h + (attn.transpose(0, 2, 1, 3).reshape(q_len, self.hidden_dim) @ self.w_o[l])

        mx.eval(h)
        ttft_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return BlendResult(
            output=h,
            cache_hits=hits,
            cache_misses=misses,
            recomputed_tokens=recomputed_tokens,
            total_context_tokens=total_ctx_tokens,
            ttft_ms=ttft_ms,
        )
