"""Full-stack live integration test proving Smart Cache, Smart Private Context, and Smart Memory."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))

from demo_three_layers import run_three_layer_proof  # noqa: E402


def test_full_stack_three_layers_proof() -> None:
    report = run_three_layer_proof(verbose=True)

    # Layer 3 · Smart Memory
    mem = report["smart_memory"]
    assert mem["active_contains_v2_only"] is True
    assert mem["as_of_v1_recovered"] is True
    assert mem["as_of_v1_superseded_by"] == mem["v2_block_id"]
    assert len(mem["v2_citations"]) >= 1
    assert "source_uri" in mem["v2_citations"][0]

    # Layer 2 · Smart Private Context
    priv = report["smart_private_context"]
    assert priv["alice_sees_only_finance"] is True
    assert priv["bob_sees_only_engineering"] is True
    assert priv["identical_text_same_blake3"] is True
    assert priv["disjoint_hmac_kv_keys"] is True
    assert priv["bob_cross_clearance_cache_hits"] == 0
    assert priv["carol_same_clearance_cache_hits"] >= 1
    assert priv["bob_blocks_after_revoke"] == 0

    # Layer 1 · Smart Cache
    cache = report["smart_cache"]
    assert cache["cold_cache_hits"] == 0
    assert cache["warm_same_scope_hits"] >= 1
    assert cache["stale_v1_kv_evicted"] is True
    assert cache["v2_kv_prewarmed_hit"] is True
    assert cache["model_swap_rewarmed_blocks"] >= 1
    assert cache["active_model_after_swap"] == "DeepSeek/DeepSeek-V3.2-Local"
