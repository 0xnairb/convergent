# Project Convergent

Clean-slate **Sovereign Cognitive Engine** for enterprises running open-weight LLMs in customer VPCs.

## Core Pillars (AGI-Era Focus)
- **Git-for-World-State ([branches.py](worker-py/src/convergent_worker/branches.py), [003_world_state_branches.sql](migrations/003_world_state_branches.sql)):** `branch`, `commit`, `diff`, `merge`, `revert`, and bi-temporal `as_of` replay over primary sources & explicit decisions in single-ACID `PostgreSQL 17`.
- **Multiplayer Shared Spaces + Hard Isolation:** Private developer/agent branches, shared team `main` brain with real-time NATS merge broadcasts, and deterministic `OpenFGA` + per-branch/per-space `HMAC-SHA256` KV isolation.
- **Hybrid Inference & Smart Cache Base:** Pluggable backend factory running the exact same Brain across self-hosted (`vLLM`/`SGLang` CUDA, `MLX`/`Ollama` Apple Silicon) and cloud proxies (`OpenAI` `prompt_cache_key`, `Anthropic` `cache_control: ephemeral`), with atomic KV eviction & pre-warming on merge/revert.
- **Hot Data Plane (`gateway-rs`):** Rust (`Axum`) serving **MCP Streamable HTTP (`/mcp`)** with **OAuth 2.1 + PKCE (`S256`)**, **OpenAI-compatible `/v1/chat/completions`**, `/v1/spaces/{space_id}/grants`, and `/v1/memory/history`.

## Pluggable Inference Backend Factory (`CONVERGENT_INFERENCE_BACKEND`)

Switch engines without changing application code or migrating memory:

| `CONVERGENT_INFERENCE_BACKEND` | Target | Engine | Example Endpoint |
|---|---|---|---|
| `vllm` (or `cuda`) | NVIDIA CUDA | `vLLM + LMCache` (`CacheBlend`) | `http://127.0.0.1:8000` |
| `sglang` | NVIDIA CUDA | `SGLang + RadixAttention` | `http://127.0.0.1:30000` |
| `mlx` (or `apple-silicon`) | Apple Silicon Metal | `mlx_lm.server` / `mlx-kv` | `http://127.0.0.1:8080` |
| `ollama` | Metal / CUDA / CPU | `Ollama` | `http://127.0.0.1:11434` |
| `openai` | Cloud Proxy | OpenAI API (`prompt_cache_key`) | `https://api.openai.com` |
| `anthropic` | Cloud Proxy | Anthropic API (`cache_control: ephemeral`) | `https://api.anthropic.com` |
| `dev` | Any | `scripts/dev_vllm_server.py` | `http://127.0.0.1:8000` |

Example on macOS Apple Silicon with Ollama or MLX:
```bash
CONVERGENT_INFERENCE_BACKEND=ollama CONVERGENT_VLLM_URL=http://127.0.0.1:11434 CONVERGENT_MODEL_ID=qwen3:8b cargo run
# or
CONVERGENT_INFERENCE_BACKEND=mlx CONVERGENT_VLLM_URL=http://127.0.0.1:8080 CONVERGENT_MODEL_ID=mlx-community/Qwen3-8B-4bit cargo run
```

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

## Running the Project Convergent Live Sample & Experiment

[examples/demo_convergent.py](examples/demo_convergent.py) automatically spawns **disposable Docker Compose infrastructure** (PostgreSQL 17 ParadeDB, Valkey 8, OpenFGA 1.8.5, NATS JetStream on ephemeral ports), runs all predefined fixtures in [examples/fixtures.py](examples/fixtures.py) against **local Ollama (`granite4.2:3b`)**, prints the full JSON proof report, and tears down all disposable containers and volumes on exit (`0` leftover containers).

```bash
cd worker-py && uv run python ../examples/demo_convergent.py
```

### Verified Run Results (Apple Silicon · Local Ollama `granite4.2:3b` · Disposable Infra)

Captured from an actual end-to-end execution of `uv run python ../examples/demo_convergent.py` (`compose_project: convergent-demo-f773c6`, infra boot: **`6.4s`**, leftover containers after teardown: **`0`**):

#### 1. Smart Cache
| Metric | Observed Value | Proof |
|---|---|---|
| Backend header (`x-convergent-backend`) | `ollama` (`granite4.2:3b`) | Factory-routed to local Ollama Metal |
| Cold request (Alice, `cache_hits=0`) | **`3,453.46 ms`** | First prefill & KV registration |
| Warm same-scope request (Carol, `cache_hits=1`) | **`864.33 ms`** (**4.0x faster**) | Shared warm KV within `space:finance` |
| Stale v1 KV evicted after supersession (`Valkey DEL`) | **`true`** | Zero stale tensor reuse |
| v2 pre-warmed hit (`post_supersede_latency_ms`) | **`true`** (`1,226.02 ms`) | Worker `num_predict=1` prefill warmup |
| Rust gateway proxy overhead | **`5.75 ms`** | Measured via `x-convergent-proxy-overhead-ms` |
| Zero-downtime model swap | **`5` blocks re-warmed → `hostlab-granite:3b`** | Active model flipped atomically in `model_registry` |

#### 2. Smart Private Context
| Metric | Observed Value | Proof |
|---|---|---|
| `alice` sees only `space:finance` | **`true`** | OpenFGA pre-filtered retrieval before Top-K |
| `bob` sees only `space:engineering` | **`true`** | Zero cross-space leakage |
| `bob` (`granite4.2:3b`) answer | *"Engineering stack: Rust Axum, PostgreSQL 17, Valkey 8. Authorization code: HELIOS-9921. No marketing budget is indicated."* | Confirmed Finance's `$4.2M` budget never reached Engineering |
| Identical secret (`HELIOS-9921`) same `BLAKE3` | **`true`** | Byte-identical content in both spaces |
| Per-clearance `HMAC-SHA256` KV keys disjoint | **`true`** (`cvg:kv:v1:476b0b65...` vs `cvg:kv:v1:8f9a60fe...`) | Cryptographic KV cache isolation |
| Cross-clearance cache hits (`bob` after `alice`) | **`0`** | Zero cross-clearance KV side-channel hit |
| Same-clearance cache hits (`carol` after `alice`) | **`1`** | Authorized teammates share warm KV |
| `bob` authorized blocks after instant revoke | **`0`** | OpenFGA tuple delete + immediate Valkey ACL invalidation |

#### 3. Smart Memory
| Metric | Observed Value | Proof |
|---|---|---|
| `granite4.2:3b` answer on v1 fixture | **`"$500 per trip."`** | Grounded in `TRAVEL_POLICY_V1` |
| `granite4.2:3b` answer after v2 supersession | **`"$1,200 per trip."`** | Grounded in `TRAVEL_POLICY_V2` |
| Active search contains v2 only (0 v1 blocks) | **`true`** | Atomic Postgres bi-temporal supersession |
| Exact source citation on v2 block | `source_uri: policy://travel-policy-401de8.md`, `offset: 0`, `length: 152`, `quote_hash: f446b6507ec9f626` | Content-addressed provenance |
| Historical `as_of=<t_v1>` query recovers v1 | **`true`** (`superseded_by: f2c8464d-a3a9-4d00-ba89-c57c70154a55`) | Full auditability without polluting active context |

### Semantic Supersession Benchmark (Standard RAG vs. Project Convergent)

Run the 12-scenario cross-document contradiction & additive-control benchmark ([benchmarks/semantic_supersession_bench.py](benchmarks/semantic_supersession_bench.py)) on local Ollama (`granite4.2:3b`):

```bash
cd worker-py && uv run python ../benchmarks/semantic_supersession_bench.py
```

Observed results (`granite4.2:3b`, `40.21s`, 8 cross-URI contradictions + 4 additive false-positive controls):

| Metric | Standard RAG (No Semantic Supersession) | Project Convergent (`CONVERGENT_SEMANTIC_INGEST=1`) |
|---|---|---|
| Cross-Document Contradiction Recall (different `source_uri`) | `0/8 (0.0%)` | **`8/8 (100.0%)`** |
| False-Positive Supersession Rate (additive controls) | N/A | **`0/4 (0.0%)`** |
| Old Superseded Block Retrieved in Active Context | **`8/8 (100.0%)`** | **`0/8 (0.0%)`** *(all 8 old blocks marked `valid_to`)* |
| Example Failure Prevented (`infra://primary-region.md` vs `incident://postmortem-42-migration.md`) | Hallucinated stale **`us-east-1`** | Answered **`eu-central-1`** |

- **Source-Authority Guardrail & Conflict Queue ([002_semantic_authority.sql](migrations/002_semantic_authority.sql)):** Lower-authority sources (e.g., `slack://`, authority `1`) contradicting higher-authority policies (e.g., `legal://`, authority `4`) are **quarantined** in `conflict_reviews` instead of silently overwriting active policy until explicitly approved via `resolve_conflict()`.

### Apple Silicon Metal GPU (`MLX`) Tensor-Level KV Cache & Side-Channel Proof

Run the real Metal GPU tensor-level KV-block splicing, physical `.safetensors` eviction, and statistical Welch's t-test side-channel proof ([mlx-kv/kv_store.py](mlx-kv/kv_store.py), [benchmarks/mlx_kv_cache_bench.py](benchmarks/mlx_kv_cache_bench.py)):

```bash
python3 benchmarks/mlx_kv_cache_bench.py
```

Observed results (`mlx 0.32.4` on `Device(gpu, 0)`, `16L x 8H x 128D float16`, `512` tokens/block):

| Proof | Observed Value | Verdict |
|---|---|---|
| Non-prefix `CacheBlend` (`[A, C]` after `[A, B, C]`) | **`11.81 ms`** vs **`29.81 ms`** cold (**`2.52x` TTFT speedup**, `153/1024` boundary tokens recomputed) | Real non-prefix GPU KV reuse |
| Same-clearance hit (`carol` after `alice`, `space:finance`) | **`1` hit**, **`3.67 ms`** TTFT | Shared warm Metal KV |
| Cross-clearance identical secret (`bob`, `space:engineering`) | **`0` hits**, **`50.74 ms`** vs **`52.13 ms`** unseen (**Welch's `t = -0.137`**) | Zero timing side-channel (`leak=false`) |
| Physical tensor eviction (`evict_block(B)`) | **`1` GPU tensor + `1` `.safetensors` file deleted**, **`0`** post-evict hits | 100% stale tensor purge |

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
