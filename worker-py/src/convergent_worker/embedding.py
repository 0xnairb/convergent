"""Lightweight deterministic embeddings for local/VPC operation, pluggable via HTTP."""

from __future__ import annotations

import hashlib
import math
import re

EMBED_DIM = 64
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def embed_text(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Compute a deterministic L2-normalized bag-of-tokens embedding vector."""
    vec = [0.0] * dim
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return vec
    for tok in tokens:
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0.0:
        vec = [round(v / norm, 6) for v in vec]
    return vec


def pgvector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in vec) + "]"
