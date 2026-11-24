# Verified Benchmarks & Proofs

All benchmarks below were executed locally on **Apple Silicon macOS** against live infrastructure — zero mocked databases or fabricated latencies.

---

## 1. End-to-End Disposable Infra Proof (Local Ollama `granite4.2:3b`)

Spawns isolated disposable Docker Compose containers (`PostgreSQL 17`, `Valkey 8`, `OpenFGA 1.8.5`, `NATS JetStream`), runs all predefined fixtures in [examples/fixtures.py](../examples/fixtures.py) through the compiled Rust `convergent-gateway` and Python `convergent-worker` against local Ollama `granite4.2:3b`, and tears down cleanly (`0` leftover containers).

```bash
cd worker-py && uv run python ../examples/demo_convergent.py
```

**Run metadata:** `compose_project: convergent-demo-f773c6`, infra boot **`6.4s`**, leftover containers **`0`**.

### Smart Cache
| Metric | Observed Value | Proof |
|---|---|---|
| Backend header (`x-convergent-backend`) | `ollama` (`granite4.2:3b`) | Factory-routed to local Ollama Metal |
| Cold request (Alice, `cache_hits=0`) | **`3,453.46 ms`** | First prefill & KV registration |
| Warm same-scope request (Carol, `cache_hits=1`) | **`864.33 ms`** (**4.0x faster**) | Shared warm KV within `space:finance` |
| Stale v1 KV evicted (`Valkey DEL`) | **`true`** | Zero stale tensor reuse |
| v2 pre-warmed hit (`post_supersede_latency_ms`) | **`true`** (`1,226.02 ms`) | Worker `num_predict=1` prefill warmup |
| Rust gateway proxy overhead | **`5.75 ms`** | Measured via `x-convergent-proxy-overhead-ms` |
| Zero-downtime model swap | **`5` blocks re-warmed → `hostlab-granite:3b`** | Active model flipped atomically in `model_registry` |

### Smart Private Context
| Metric | Observed Value | Proof |
|---|---|---|
| `alice` sees only `space:finance` | **`true`** | OpenFGA pre-filtered retrieval before Top-K |
| `bob` sees only `space:engineering` | **`true`** | Zero cross-space leakage |
| `bob` (`granite4.2:3b`) answer | *"Engineering stack: Rust Axum, PostgreSQL 17, Valkey 8. Authorization code: HELIOS-9921. No marketing budget is indicated."* | Finance's `$4.2M` budget never reached Engineering |
| Identical secret (`HELIOS-9921`) same `BLAKE3` | **`true`** | Byte-identical content in both spaces |
| Per-clearance `HMAC-SHA256` KV keys disjoint | **`true`** (`cvg:kv:v1:476b0b65...` vs `cvg:kv:v1:8f9a60fe...`) | Cryptographic KV cache isolation |
| Cross-clearance cache hits (`bob` after `alice`) | **`0`** | Zero cross-clearance KV side-channel hit |
| Same-clearance cache hits (`carol` after `alice`) | **`1`** | Authorized teammates share warm KV |
| `bob` authorized blocks after instant revoke | **`0`** | OpenFGA tuple delete + immediate Valkey ACL invalidation |

### Smart Memory
| Metric | Observed Value | Proof |
|---|---|---|
| `granite4.2:3b` answer on v1 fixture | **`"$500 per trip."`** | Grounded in `TRAVEL_POLICY_V1` |
| `granite4.2:3b` answer after v2 supersession | **`"$1,200 per trip."`** | Grounded in `TRAVEL_POLICY_V2` |
| Active search contains v2 only | **`true`** | Atomic Postgres bi-temporal supersession |
| Exact citation on v2 block | `source_uri: policy://travel-policy-401de8.md`, `offset: 0`, `length: 152`, `quote_hash: f446b6507ec9f626` | Content-addressed provenance |
| Historical `as_of=<t_v1>` query recovers v1 | **`true`** | Full auditability without polluting active context |

---

## 2. Apple Silicon Metal GPU (`MLX`) Tensor KV Cache & Side-Channel Proof

Executes real multi-layer Transformer KV prefill, non-prefix `CacheBlend` boundary recomputation (`15%`), physical `.safetensors` eviction, and a statistical Welch's t-test side-channel check on Metal GPU (`mlx 0.32.4`, `Device(gpu, 0)`).

```bash
python3 benchmarks/mlx_kv_cache_bench.py
```

**Geometry:** `16L x 8H x 128D (float16)`, `512` tokens/block.

| Proof | Observed Value | Verdict |
|---|---|---|
| Non-prefix `CacheBlend` (`[A, C]` after `[A, B, C]`) | **`11.81 ms`** vs **`29.81 ms`** cold (**`2.52x` TTFT speedup**, `153/1024` boundary tokens recomputed) | Real non-prefix Metal GPU KV reuse |
| Same-clearance hit (`carol` after `alice`, `space:finance`) | **`1` hit**, **`3.67 ms`** TTFT | Shared warm Metal KV |
| Cross-clearance identical secret (`bob`, `space:engineering`) | **`0` hits**, **`50.74 ms`** vs **`52.13 ms`** unseen (**Welch's `t = -0.137`**) | Zero timing side-channel (`leak=false`) |
| Physical tensor eviction (`evict_block(B)`) | **`1` GPU tensor + `1` `.safetensors` file deleted**, **`0`** post-evict hits | 100% stale tensor purge |

---

## 3. Cross-Document Semantic Supersession Benchmark (`granite4.2:3b`)

Compares **Standard RAG** against **Project Convergent** across 12 enterprise scenarios (8 cross-URI contradictions + 4 additive false-positive controls).

```bash
cd worker-py && uv run python ../benchmarks/semantic_supersession_bench.py
```

| Metric | Standard RAG | Project Convergent |
|---|---|---|
| Cross-Document Contradiction Recall (different `source_uri`) | `0/8 (0.0%)` | **`8/8 (100.0%)`** |
| False-Positive Supersession Rate (4 additive controls) | N/A | **`0/4 (0.0%)`** |
| Old Superseded Block Remaining Active in Retrieval | **`8/8 (100.0%)`** | **`0/8 (0.0%)`** |
| Example Failure Prevented (`infra://primary-region.md` vs `incident://postmortem-42-migration.md`) | Hallucinated stale **`us-east-1`** | Answered **`eu-central-1`** |

---

## 4. NVIDIA GPU VPC Validation Suite

For production Linux NVIDIA GPU nodes running `vLLM + LMCache` (`CacheBlend`):

```bash
docker compose -f deploy/compose.yml -f deploy/compose.gpu.yml up -d
CONVERGENT_VLLM_URL=http://127.0.0.1:8000 \
CONVERGENT_GATEWAY_URL=http://127.0.0.1:8090 \
  uv run --project worker-py python scripts/gpu_validation_suite.py
```
