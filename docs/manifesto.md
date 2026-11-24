# Project Convergent Manifesto

**Own the World-State. Swap the Model.**

---

## 1. The Five-Year Premise

Over the next five years, three things will become true simultaneously:

1. **Tokens approach zero cost.** Inference hardware gets an order of magnitude cheaper every ~24 months. Selling middleware to "save 50% on API bills" is a depreciating business.
2. **Models become interchangeable CPUs.** Open-weight models (`Qwen`, `DeepSeek`, `Llama`, `Granite`) and frontier cloud APIs leapfrog each other every quarter. Trapping your team's institutional state inside a single vendor's cloud is a strategic mistake.
3. **Autonomous agents outnumber humans 100:1.** Instead of one human chatting with one bot, teams run dozens or thousands of concurrent agents executing tens of thousands of reads and mutations per day.

---

## 2. What Dies in the AGI Era

As models become smarter, faster, and cheaper, entire categories of 2023–2025 AI tooling become obsolete:

- **Naive chunk-and-pray vector RAG:** Long-context, agentic models reading structured primary sources beat brittle 512-token cosine chunks.
- **Lossy summarization wrappers:** Having a weak model compress documents into lossy notes destroys nuance that stronger models need.
- **Copy-heavy Git branching for agents:** Decentralized, full-checkout branching was designed for offline humans making 5 commits a day — not always-online agent swarms making 30,000 changes a day.
- **Compliance-checkbox governance theater:** Security bolted on as bureaucratic friction kills developer velocity. Real isolation must be an effortless architectural primitive.

---

## 3. What Survives by Physics and Game Theory

Even with superintelligent models at near-zero token cost, **four fundamental constraints never disappear**:

### I. Distributed Coordination (Server-Authoritative World-State)
Higher intelligence does not solve concurrent writes. Ten brilliant engineers still collide without version control; a hundred autonomous agents collide in seconds.

Convergent replaces copy-heavy branches with **Layered Overlays & Server-Authoritative Changesets**:
- **Zero-copy `Overlay` sandboxes:** Spawning 10,000 agent sandboxes copies **0 blocks**.
- **Optimistic `Land` with auto-rebase:** Non-overlapping agent updates land atomically at monotonic `rev + 1`; true conflicts surface cleanly.
- **Instant `Undo(rev)` & `as_of` replay:** Any bad agent mutation is reversed in milliseconds without corrupting history.

### II. Hard Context Boundaries (Mount Composition)
Information boundaries are game theory, not bureaucracy. A multi-client studio cannot mix Client A and Client B; a developer's scratchpad should not pollute a team's shared state.

In Convergent, isolation is **mount composition** backed by deterministic `OpenFGA` pre-filtering and per-scope `HMAC-SHA256` cache keys. An agent only mounts the spaces and overlays it is authorized to see.

### III. Memory-Bandwidth Physics (Smart KV Cache)
Even if tokens are free, HBM bandwidth and the speed of light are not. Re-prefilling 100k–1M tokens of shared team context on every agent turn destroys interactive speed.

Because every agent mounts the **same immutable base revisions**, Convergent keeps base KV tensors hot across the entire swarm — splicing non-prefix blocks (`CacheBlend`) and physically evicting only the exact keys touched by a `Land` or `Undo`.

### IV. Model Sovereignty (Hybrid Execution)
Your versioned institutional world-state is 100x more valuable than any single model's weights. Convergent runs the **exact same Brain** across self-hosted engines (`vLLM`, `SGLang`, `MLX`, `Ollama`) and cloud proxies (`OpenAI`, `Anthropic`) — so you can pin sensitive spaces to local hardware while routing general work anywhere.

---

## 4. The Goal

Build the open, local-first **World-State & Execution Plane** for multi-agent teams: fast enough to feel invisible, isolated enough to trust with real work, and portable enough to outlive every model generation.
