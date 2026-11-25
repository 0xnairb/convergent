# Project Convergent

**The local-first World-State & Execution Plane for multi-agent teams.**

Own your versioned institutional memory and context boundaries; swap any self-hosted (`vLLM`, `SGLang`, `MLX`, `Ollama`) or cloud (`OpenAI`, `Anthropic`) model underneath.

## Documentation
- **[Manifesto & Philosophy](docs/manifesto.md)** — Why tokens → `$0`, why Git-style branching breaks at agent scale, and what survives in the AGI era.
- **[Architecture](docs/architecture.md)** — Layered Overlays (`open_overlay`, `stage_delta`, `land_overlay`, `undo_revision`), Mount Stacks, MCP/HTTP API, and the Inference Backend Factory.
- **[Verified Benchmarks](docs/benchmarks.md)** — Real Apple Silicon Metal (`MLX`), local Ollama (`granite4.2:3b`), and semantic supersession benchmark results.

---

## Convergent Arena (Single-Page Showcase Dashboard)

Launch **Convergent Arena** (`http://127.0.0.1:8095`) to configure any local or remote Ollama endpoint/model and run all 5 live proofs with a single button — updating every comparison card and benchmark row with real execution data:

```bash
cd worker-py && uv run python ../arena/server.py
```

## Quickstart (Disposable CLI Demo on macOS Apple Silicon)

Requires Docker and local Ollama (`granite4.2:3b`). Spawns isolated disposable containers, runs all predefined fixtures end-to-end, and cleans up with `0` leftover containers:

```bash
cd worker-py && uv run python ../examples/demo_convergent.py
```

## Local Development Stack

```bash
# 1. Start PostgreSQL 17 (ParadeDB), Valkey 8, OpenFGA 1.8.5, and NATS JetStream
docker compose -f deploy/compose.yml up -d

# 2. Run idempotent Postgres, OpenFGA, and NATS JetStream migrations
cd worker-py && uv run python ../scripts/bootstrap.py

# 3. Start the async worker daemon (terminal 1)
uv run convergent-worker

# 4. Start the Rust gateway with your chosen backend (terminal 2)
cd ../gateway-rs
CONVERGENT_INFERENCE_BACKEND=ollama CONVERGENT_VLLM_URL=http://127.0.0.1:11434 CONVERGENT_MODEL_ID=granite4.2:3b cargo run
```

Switch backends anytime via `CONVERGENT_INFERENCE_BACKEND`: `vllm` (`cuda`), `sglang`, `mlx`, `ollama`, `openai`, `anthropic`, or `dev` (see [docs/architecture.md](docs/architecture.md)).

## Running Tests & Benchmarks

```bash
# Rust gateway unit + integration suites
cd gateway-rs && cargo test -- --nocapture

# Python worker + overlay MVCC + E2E suites
cd worker-py && uv run pytest -v

# Apple Silicon Metal GPU (MLX) KV-cache & Welch's t-test side-channel proof
python3 benchmarks/mlx_kv_cache_bench.py

# Cross-document semantic supersession benchmark (Ollama granite4.2:3b)
cd worker-py && uv run python ../benchmarks/semantic_supersession_bench.py
```
