# Project Convergent

Clean-slate 3-Layer Sovereign Cognitive Engine for self-hosted open-weight AI:
- **Layer 1 · Smart Cache:** `vLLM + LMCache` (`CacheBlend`) + tiered `Valkey`/NVMe KV store + zero-downtime model-swap re-warming.
- **Layer 2 · Smart Private Context:** `OpenFGA` ReBAC pre-filtering + per-clearance `HMAC-SHA256` KV key isolation (`PROMPTPEEK` / `SpliceLeak` defense) + opt-in OpenBox `Withheld[]` webhook.
- **Layer 3 · Smart Memory:** Single-ACID `PostgreSQL 17` (`pgvector` + `pg_search` BM25 + bi-temporal supersession) + `NATS JetStream` + Python (`uv`) async workers for cited fact extraction and atomic KV eviction.

## Layout
- `gateway-rs/` — Rust (`Axum` + `Hyper` + `Tokio`) hot data plane: OAuth 2.1 + PKCE, `/mcp` Streamable HTTP, `/v1/chat/completions`, OpenFGA pre-filtering, HMAC KV key derivation.
- `worker-py/` — Python (`uv`) async workers: ingestion, cited fact distillation, transactional supersession outbox relay, `Valkey DEL` eviction, and `max_tokens=1` prefill warmup.
- `deploy/compose.yml` — Local/VPC stack (ParadeDB PG17, Valkey 8, OpenFGA, NATS JetStream).
- `migrations/` — Idempotent PostgreSQL & OpenFGA schemas.
