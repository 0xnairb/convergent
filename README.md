# Project Convergent

Clean-slate **3-Layer Sovereign Cognitive Engine** for enterprises running open-weight LLMs in customer VPCs.

## Architecture
- **Layer 1 · Smart Cache:** `vLLM` + `LMCache` (`CacheBlend`) tiered to `Valkey` + NATS-driven atomic KV eviction (`Valkey DEL`) & `max_tokens=1` prefill warming + zero-downtime `model_registry` cutover.
- **Layer 2 · Smart Private Context:** Deterministic `OpenFGA` Zanzibar ReBAC pre-filtering + per-clearance `HMAC-SHA256` KV key isolation + opt-in Policy Ethical Walls (`Withheld[]`) webhook.
- **Layer 3 · Smart Memory:** Single-ACID `PostgreSQL 17` (`pgvector` + `pg_search` BM25 + bi-temporal `valid_from`/`valid_to`/`superseded_by`) + `NATS JetStream` transactional outbox & Python (`uv`) workers.
- **Hot Data Plane (`gateway-rs`):** Rust (`Axum`) serving **MCP Streamable HTTP (`/mcp`)** with **OAuth 2.1 + PKCE (`S256`)**, **OpenAI-compatible `/v1/chat/completions`** (JSON & SSE streaming), `/v1/spaces/{space_id}/grants`, and `/v1/memory/history`.

## Quickstart (macOS Apple Silicon / Local Dev)

```bash
# 1. Start PostgreSQL 17 (ParadeDB), Valkey 8, OpenFGA 1.8.5, and NATS JetStream
docker compose -f deploy/compose.yml up -d

# 2. Run idempotent Postgres, OpenFGA, and NATS JetStream bootstrap
cd worker-py && uv run python ../scripts/bootstrap.py

# 3. Start the local Apple Silicon dev vLLM simulator (optional: set CONVERGENT_UPSTREAM_LLM_URL for Ollama/MLX)
uv run python ../scripts/dev_vllm_server.py

# 4. Start the async memory/ingest/KV-evictor worker daemon (in a second terminal)
uv run convergent-worker

# 5. Start the Rust gateway (in a third terminal)
cd ../gateway-rs && cargo run
```

## Running the 3-Layer Live Sample & Integration Proof

```bash
# Boots the real Rust gateway binary, Python worker daemon, and local dev vLLM simulator
# against live Postgres 17, Valkey 8, OpenFGA 1.8.5, and NATS JetStream:
cd worker-py && uv run python ../examples/demo_three_layers.py
```

## Running Tests

```bash
# Rust integration tests (infra, OpenFGA/HMAC/Ethical Walls, OAuth2.1/MCP/SSE, Admin/History)
cd gateway-rs && cargo test -- --nocapture

# Python worker & E2E tests (infra, bi-temporal memory/outbox, KV eviction/swap, E2E ingest->dev_vllm)
cd worker-py && uv run pytest -v
```

## Production NVIDIA GPU VPC Deployment

```bash
# Launch infrastructure + self-hosted vLLM + LMCache (CacheBlend)
docker compose -f deploy/compose.yml -f deploy/compose.gpu.yml up -d

# Execute live GPU benchmark & side-channel proof gate
CONVERGENT_VLLM_URL=http://127.0.0.1:8000 \
CONVERGENT_GATEWAY_URL=http://127.0.0.1:8090 \
  uv run --project worker-py python scripts/gpu_validation_suite.py
```
