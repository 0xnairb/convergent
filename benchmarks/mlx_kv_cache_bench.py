"""Apple Silicon Metal GPU (MLX) Proof Gate for Project Convergent Smart Cache & Side-Channel Isolation.

Executes three live Metal GPU proofs on Apple Silicon (`mlx.core` Device(gpu, 0)):
  1. Non-Prefix CacheBlend TTFT Speedup: [A, C] after caching [A, B, C] vs cold full prefill
  2. Statistical Side-Channel Proof (Welch's t-test): Cross-clearance identical text has 0 cache hits
     and statistically indistinguishable TTFT vs unseen text, while same-clearance teammate gets 100% hits
  3. Physical Tensor Eviction: evict_block(B) deletes both Metal Unified Memory tensors and `.safetensors`
     files on disk so stale tensors can never be reused

Usage:
    python3 benchmarks/mlx_kv_cache_bench.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
import sys
import tempfile

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mlx-kv"))

from kv_store import BlockSpec, MlxKvCacheEngine, content_hash  # noqa: E402


def _welch_t_stat(a: list[float], b: list[float]) -> float:
    mean_a, mean_b = statistics.mean(a), statistics.mean(b)
    var_a = statistics.variance(a) if len(a) > 1 else 0.0
    var_b = statistics.variance(b) if len(b) > 1 else 0.0
    denom = math.sqrt((var_a / len(a)) + (var_b / len(b)))
    if denom == 0.0:
        return 0.0
    return round((mean_a - mean_b) / denom, 3)


def _make_block(scope: str, text: str, seq_len: int, hidden_dim: int, seed: int) -> BlockSpec:
    mx.random.seed(seed)
    tokens = mx.random.normal((seq_len, hidden_dim), dtype=mx.float16)
    mx.eval(tokens)
    return BlockSpec(clearance_scope=scope, block_hash=content_hash(text), tokens=tokens)


def run_mlx_proof() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="convergent_mlx_kv_") as tmpdir:
        engine = MlxKvCacheEngine(
            model_id="mlx-community/Granite-4.2-3B-FP16",
            master_seed=b"convergent-apple-silicon-metal-seed",
            storage_dir=Path(tmpdir),
            num_layers=16,
            num_heads=8,
            head_dim=128,
            blend_recompute_ratio=0.15,
        )
        seq_len = 512
        hidden_dim = engine.hidden_dim
        query = mx.random.normal((16, hidden_dim), dtype=mx.float16)
        mx.eval(query)

        # Warm up Metal shaders once
        _warm = _make_block("space:warmup", "shader-warmup", 64, hidden_dim, 1)
        engine.compose_and_attend([_warm], query, use_cache=False)

        # =====================================================================
        # PROOF 1: Non-Prefix CacheBlend TTFT Speedup ([A, C] after [A, B, C])
        # =====================================================================
        blk_a = _make_block("space:finance", "Finance Block A: Q4 budget rules", seq_len, hidden_dim, 10)
        blk_b = _make_block("space:finance", "Finance Block B: Travel policy v1", seq_len, hidden_dim, 11)
        blk_c = _make_block("space:finance", "Finance Block C: Audit compliance", seq_len, hidden_dim, 12)

        # Populate cache with [A, B, C]
        engine.compose_and_attend([blk_a, blk_b, blk_c], query, use_cache=True)

        # Benchmark non-prefix subset [A, C]: cold full prefill vs warm CacheBlend splicing
        trials = 8
        cold_times = [engine.compose_and_attend([blk_a, blk_c], query, use_cache=False).ttft_ms for _ in range(trials)]
        blend_results = [engine.compose_and_attend([blk_a, blk_c], query, use_cache=True) for _ in range(trials)]
        blend_times = [r.ttft_ms for r in blend_results]

        cold_med = round(statistics.median(cold_times), 2)
        blend_med = round(statistics.median(blend_times), 2)
        speedup = round(cold_med / blend_med, 2) if blend_med > 0 else 0.0

        # =====================================================================
        # PROOF 2: Statistical Side-Channel Proof (Welch's t-test)
        # =====================================================================
        secret_text = "Confidential launch code HELIOS-9921"
        alice_secret = _make_block("space:finance", secret_text, seq_len, hidden_dim, 20)
        carol_secret = _make_block("space:finance", secret_text, seq_len, hidden_dim, 20)
        bob_secret = _make_block("space:engineering", secret_text, seq_len, hidden_dim, 20)

        # Alice warms her finance secret
        engine.compose_and_attend([alice_secret], query, use_cache=True)

        # Carol (same clearance) queries identical secret -> warm hit
        carol_res = engine.compose_and_attend([carol_secret], query, use_cache=True)

        # Bob (different clearance) queries identical secret vs unseen baseline (evicting Bob's key each trial to measure cross-clearance cold path)
        bob_identical_times: list[float] = []
        bob_identical_hits = 0
        bob_unseen_times: list[float] = []
        for i in range(trials):
            # Bob queries identical secret text (must NEVER hit Alice's cached finance tensor)
            bob_trial = _make_block(f"space:engineering-trial-{i}", secret_text, seq_len, hidden_dim, 20)
            res_id = engine.compose_and_attend([bob_trial], query, use_cache=True)
            bob_identical_hits += res_id.cache_hits
            bob_identical_times.append(res_id.ttft_ms)

            # Bob queries completely unseen text
            bob_unseen = _make_block(f"space:engineering-unseen-{i}", f"Unseen random doc {i}", seq_len, hidden_dim, 100 + i)
            res_un = engine.compose_and_attend([bob_unseen], query, use_cache=True)
            bob_unseen_times.append(res_un.ttft_ms)

        t_stat = _welch_t_stat(bob_identical_times, bob_unseen_times)

        # =====================================================================
        # PROOF 3: Physical Metal GPU + Disk Tensor Eviction
        # =====================================================================
        evict_stats = engine.evict_block(blk_b.block_hash)
        after_evict = engine.compose_and_attend([blk_b], query, use_cache=True)
        disk_files_for_b = [p.name for p in Path(tmpdir).glob("*.safetensors") if blk_b.block_hash in p.name]

        report = {
            "hardware": {
                "mlx_version": mx.__version__,
                "device": str(mx.default_device()),
                "kv_geometry": f"{engine.num_layers}L x {engine.num_heads}H x {engine.head_dim}D (float16), {seq_len} tokens/block",
            },
            "non_prefix_cacheblend": {
                "cold_prefill_median_ms": cold_med,
                "cacheblend_spliced_median_ms": blend_med,
                "ttft_speedup_x": speedup,
                "cache_hits_per_query": blend_results[0].cache_hits,
                "recomputed_boundary_tokens": blend_results[0].recomputed_tokens,
                "total_context_tokens": blend_results[0].total_context_tokens,
            },
            "side_channel_isolation": {
                "carol_same_clearance_hits": carol_res.cache_hits,
                "carol_same_clearance_ttft_ms": carol_res.ttft_ms,
                "bob_cross_clearance_hits": bob_identical_hits,
                "bob_identical_median_ms": round(statistics.median(bob_identical_times), 2),
                "bob_unseen_median_ms": round(statistics.median(bob_unseen_times), 2),
                "welch_t_statistic": t_stat,
                "side_channel_leak_detected": abs(t_stat) > 2.5 or bob_identical_hits > 0,
            },
            "physical_tensor_eviction": {
                "unified_memory_tensors_evicted": evict_stats["mem_evicted"],
                "safetensors_files_evicted": evict_stats["disk_evicted"],
                "post_evict_cache_hits": after_evict.cache_hits,
                "post_evict_cache_misses": after_evict.cache_misses,
                "stale_disk_files_remaining": len(disk_files_for_b),
            },
        }
        print(json.dumps(report, indent=2))
        assert speedup > 1.5, f"Expected real CacheBlend speedup > 1.5x, got {speedup}x"
        assert bob_identical_hits == 0
        assert carol_res.cache_hits == 1
        assert evict_stats["mem_evicted"] == 1 and evict_stats["disk_evicted"] == 1
        assert after_evict.cache_hits == 0 and after_evict.cache_misses == 1
        return report


if __name__ == "__main__":
    run_mlx_proof()
