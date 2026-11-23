"""Semantic Supersession Benchmark: Standard RAG vs. Project Convergent (local Ollama).

Runs 12 realistic cross-document enterprise scenarios (8 cross-URI contradictions +
4 additive non-conflicting controls) against a live PostgreSQL 17 instance and local
Ollama (`granite4.2:3b` by default) to prove:
  1. Contradiction Recall (true positives across different source_uris)
  2. False-Positive Rate (must NOT supersede additive/compatible facts)
  3. Stale-Fact Elimination Rate & End-to-End QA Accuracy: Standard RAG vs. Project Convergent

Usage:
    cd convergent/worker-py && uv run python ../benchmarks/semantic_supersession_bench.py
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import uuid

import httpx
import psycopg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker-py" / "src"))

from convergent_worker.memory import hybrid_search, ingest_document  # noqa: E402
from convergent_worker.semantic import (  # noqa: E402
    classify_relationship_llm,
    extract_claims_llm,
    ingest_document_semantic,
)

PG_DSN = os.environ.get(
    "CONVERGENT_PG_DSN",
    "postgresql://convergent:convergent_dev_password@127.0.0.1:55432/convergent",
)
OLLAMA_URL = os.environ.get("CONVERGENT_VLLM_URL", "http://127.0.0.1:11434")
MODEL_ID = os.environ.get("CONVERGENT_MODEL_ID", "granite4.2:3b")


@dataclass(frozen=True)
class ContradictionCase:
    name: str
    old_uri: str
    old_text: str
    new_uri: str
    new_text: str
    query: str
    expected_new_token: str
    forbidden_old_token: str


@dataclass(frozen=True)
class AdditiveCase:
    name: str
    base_uri: str
    base_text: str
    add_uri: str
    add_text: str
    query: str
    must_keep_tokens: tuple[str, str]


CONTRADICTION_CASES: list[ContradictionCase] = [
    ContradictionCase(
        name="travel_cap_pdf_vs_cfo_memo",
        old_uri="policy://travel-2025.pdf",
        old_text="ACME corporate travel reimbursement cap is $500 per trip.",
        new_uri="memo://cfo-q4-update.md",
        new_text="Effective Q4, ACME corporate travel reimbursement cap is raised to $1,200 per trip.",
        query="What is the ACME corporate travel reimbursement cap per trip?",
        expected_new_token="1,200",
        forbidden_old_token="500",
    ),
    ContradictionCase(
        name="database_rfc_vs_adr",
        old_uri="rfc://001-database.md",
        old_text="All new backend services must use MongoDB as their primary database.",
        new_uri="adr://014-storage-decision.md",
        new_text="Architecture Decision 014: All new backend services must use PostgreSQL 17 as their primary database; MongoDB is deprecated.",
        query="Which primary database must new backend services use?",
        expected_new_token="PostgreSQL",
        forbidden_old_token="MongoDB",
    ),
    ContradictionCase(
        name="pto_handbook_vs_slack",
        old_uri="handbook://pto-policy-2025.md",
        old_text="Full-time employees receive 15 days of paid time off (PTO) per year.",
        new_uri="slack://hr-announcements-2026.txt",
        new_text="HR update: Effective January 1, full-time employees receive 20 days of paid time off (PTO) per year.",
        query="How many days of paid time off (PTO) do full-time employees receive per year?",
        expected_new_token="20",
        forbidden_old_token="15",
    ),
    ContradictionCase(
        name="password_rotation_policy_vs_nist",
        old_uri="security://password-policy-v1.md",
        old_text="Employee passwords must be rotated every 90 days.",
        new_uri="infosec://nist-2026-update.md",
        new_text="InfoSec update: Mandatory 90-day password rotation is abolished; passwords are only rotated upon suspected compromise.",
        query="Do employees still have to rotate passwords every 90 days?",
        expected_new_token="compromise",
        forbidden_old_token="every 90 days",
    ),
    ContradictionCase(
        name="sla_pdf_vs_msa_addendum",
        old_uri="sales://enterprise-sla-2025.pdf",
        old_text="Enterprise P1 critical incident response SLA is 24 hours.",
        new_uri="legal://msa-addendum-2026.md",
        new_text="Legal Addendum: Enterprise P1 critical incident response SLA is tightened to 1 hour.",
        query="What is the Enterprise P1 critical incident response SLA?",
        expected_new_token="1 hour",
        forbidden_old_token="24 hours",
    ),
    ContradictionCase(
        name="deploy_window_ops_vs_cicd",
        old_uri="ops://deploy-window.md",
        old_text="Production deployments are restricted to Tuesdays only.",
        new_uri="eng://ci-cd-charter.md",
        new_text="Engineering Charter: Continuous daily production deployments are now enabled; the Tuesday-only restriction is lifted.",
        query="When are production deployments allowed?",
        expected_new_token="daily",
        forbidden_old_token="Tuesdays only",
    ),
    ContradictionCase(
        name="receipt_threshold_finance_vs_email",
        old_uri="finance://expense-receipts.md",
        old_text="Expense receipts are mandatory for any purchase over $75.",
        new_uri="email://controller-update.txt",
        new_text="Controller notice: The mandatory expense receipt threshold has been raised to purchases over $100.",
        query="Over what dollar amount are expense receipts mandatory?",
        expected_new_token="100",
        forbidden_old_token="75",
    ),
    ContradictionCase(
        name="primary_region_infra_vs_postmortem",
        old_uri="infra://primary-region.md",
        old_text="Our primary production cloud region is us-east-1.",
        new_uri="incident://postmortem-42-migration.md",
        new_text="Postmortem 42: Primary production cloud region has been migrated from us-east-1 to eu-central-1.",
        query="What is our primary production cloud region?",
        expected_new_token="eu-central-1",
        forbidden_old_token="us-east-1",
    ),
]

ADDITIVE_CASES: list[AdditiveCase] = [
    AdditiveCase(
        name="travel_cap_plus_flight_class",
        base_uri="policy://travel-2025.pdf",
        base_text="ACME corporate travel hotel and meal reimbursement cap is $1,200 per trip.",
        add_uri="policy://flight-booking.md",
        add_text="ACME corporate travel flights under 6 hours must be booked in economy class.",
        query="corporate travel",
        must_keep_tokens=("1,200", "economy"),
    ),
    AdditiveCase(
        name="postgres_memory_plus_valkey_cache",
        base_uri="adr://014-storage-decision.md",
        base_text="Primary durable memory store is PostgreSQL 17.",
        add_uri="adr://015-cache-tier.md",
        add_text="Ephemeral KV cache tier uses Valkey 8.",
        query="memory cache store",
        must_keep_tokens=("PostgreSQL 17", "Valkey 8"),
    ),
    AdditiveCase(
        name="sla_time_plus_support_channel",
        base_uri="legal://msa-addendum-2026.md",
        base_text="Enterprise P1 critical incident response SLA is 1 hour.",
        add_uri="sales://support-channels.md",
        add_text="Enterprise customers receive a dedicated Slack Connect support channel.",
        query="Enterprise support SLA",
        must_keep_tokens=("1 hour", "Slack Connect"),
    ),
    AdditiveCase(
        name="primary_region_plus_dr_region",
        base_uri="incident://postmortem-42-migration.md",
        base_text="Primary production cloud region is eu-central-1.",
        add_uri="infra://dr-region.md",
        add_text="Disaster recovery warm standby cloud region is ap-southeast-1.",
        query="production cloud region",
        must_keep_tokens=("eu-central-1", "ap-southeast-1"),
    ),
]


def _ask_llm(http: httpx.Client, context_blocks: list[str], query: str) -> str:
    ctx = "\n".join(f"- {b}" for b in context_blocks)
    resp = http.post(
        f"{OLLAMA_URL.rstrip('/')}/api/chat",
        json={
            "model": MODEL_ID,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.0, "num_predict": 80},
            "messages": [
                {
                    "role": "system",
                    "content": "Answer concisely in one short sentence using ONLY the provided context blocks. Do not mention superseded or conflicting options if only one is present.",
                },
                {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {query}"},
            ],
        },
    )
    resp.raise_for_status()
    return str(resp.json()["message"]["content"]).strip()


def run_benchmark() -> dict[str, Any]:
    t0 = time.monotonic()
    org_id = uuid.uuid4()

    rag_stale_leaks = 0
    conv_stale_leaks = 0
    conv_tp = 0
    rag_qa_correct = 0
    conv_qa_correct = 0
    conv_fp = 0
    details: list[dict[str, Any]] = []

    with psycopg.connect(PG_DSN, autocommit=True) as conn, httpx.Client(timeout=60.0) as http:
        extractor = lambda text: extract_claims_llm(http, ollama_url=OLLAMA_URL, model_id=MODEL_ID, text=text)
        classifier = lambda old_c, new_c: classify_relationship_llm(
            http, ollama_url=OLLAMA_URL, model_id=MODEL_ID, existing_claim=old_c, new_claim=new_c
        )

        # 1. Cross-document contradiction benchmark
        for case in CONTRADICTION_CASES:
            space_rag = uuid.uuid4()
            space_conv = uuid.uuid4()

            # Baseline Standard RAG (same-URI only; different URIs both remain active!)
            ingest_document(conn, org_id=org_id, space_id=space_rag, clearance_scope=f"space:{space_rag}", source_uri=case.old_uri, text=case.old_text)
            ingest_document(conn, org_id=org_id, space_id=space_rag, clearance_scope=f"space:{space_rag}", source_uri=case.new_uri, text=case.new_text)
            rag_blocks = [str(b["canonical_text"]) for b in hybrid_search(conn, org_id=org_id, space_id=space_rag, query=case.query, top_k=5)]
            rag_has_stale = any(case.forbidden_old_token.lower() in b.lower() for b in rag_blocks)
            if rag_has_stale:
                rag_stale_leaks += 1
            rag_ans = _ask_llm(http, rag_blocks, case.query)
            rag_ok = (case.expected_new_token.lower() in rag_ans.lower()) and (case.forbidden_old_token.lower() not in rag_ans.lower())
            if rag_ok:
                rag_qa_correct += 1

            # Project Convergent Semantic Supersession
            ingest_document_semantic(
                conn, org_id=org_id, space_id=space_conv, clearance_scope=f"space:{space_conv}",
                source_uri=case.old_uri, text=case.old_text, extractor=extractor, classifier=classifier,
            )
            res2 = ingest_document_semantic(
                conn, org_id=org_id, space_id=space_conv, clearance_scope=f"space:{space_conv}",
                source_uri=case.new_uri, text=case.new_text, extractor=extractor, classifier=classifier,
            )
            if res2.cross_doc_superseded_ids:
                conv_tp += 1
            conv_blocks = [str(b["canonical_text"]) for b in hybrid_search(conn, org_id=org_id, space_id=space_conv, query=case.query, top_k=5)]
            conv_has_stale = any(case.forbidden_old_token.lower() in b.lower() for b in conv_blocks)
            if conv_has_stale:
                conv_stale_leaks += 1
            conv_ans = _ask_llm(http, conv_blocks, case.query)
            conv_ok = (case.expected_new_token.lower() in conv_ans.lower()) and (case.forbidden_old_token.lower() not in conv_ans.lower())
            if conv_ok:
                conv_qa_correct += 1

            details.append({
                "case": case.name,
                "rag_stale_retrieved": rag_has_stale,
                "rag_answer": rag_ans,
                "rag_qa_clean": rag_ok,
                "convergent_cross_doc_superseded": bool(res2.cross_doc_superseded_ids),
                "convergent_stale_retrieved": conv_has_stale,
                "convergent_answer": conv_ans,
                "convergent_qa_clean": conv_ok,
            })

        # 2. Additive non-conflicting control benchmark (false-positive check)
        additive_details: list[dict[str, Any]] = []
        for ac in ADDITIVE_CASES:
            space_ctrl = uuid.uuid4()
            ingest_document_semantic(
                conn, org_id=org_id, space_id=space_ctrl, clearance_scope=f"space:{space_ctrl}",
                source_uri=ac.base_uri, text=ac.base_text, extractor=extractor, classifier=classifier,
            )
            res_add = ingest_document_semantic(
                conn, org_id=org_id, space_id=space_ctrl, clearance_scope=f"space:{space_ctrl}",
                source_uri=ac.add_uri, text=ac.add_text, extractor=extractor, classifier=classifier,
            )
            false_pos = len(res_add.cross_doc_superseded_ids) > 0
            if false_pos:
                conv_fp += 1
            active = [str(b["canonical_text"]) for b in hybrid_search(conn, org_id=org_id, space_id=space_ctrl, query=ac.query, top_k=5)]
            both_kept = all(any(tok.lower() in b.lower() for b in active) for tok in ac.must_keep_tokens)
            additive_details.append({"case": ac.name, "false_positive_supersession": false_pos, "both_facts_preserved": both_kept})

    n_contra = len(CONTRADICTION_CASES)
    n_add = len(ADDITIVE_CASES)
    summary = {
        "model": MODEL_ID,
        "elapsed_secs": round(time.monotonic() - t0, 2),
        "contradiction_cases": n_contra,
        "additive_control_cases": n_add,
        "standard_rag": {
            "stale_context_leakage_rate": f"{rag_stale_leaks}/{n_contra} ({100.0 * rag_stale_leaks / n_contra:.1f}%)",
            "clean_qa_accuracy": f"{rag_qa_correct}/{n_contra} ({100.0 * rag_qa_correct / n_contra:.1f}%)",
        },
        "project_convergent": {
            "cross_doc_contradiction_recall": f"{conv_tp}/{n_contra} ({100.0 * conv_tp / n_contra:.1f}%)",
            "false_positive_supersession_rate": f"{conv_fp}/{n_add} ({100.0 * conv_fp / n_add:.1f}%)",
            "stale_context_leakage_rate": f"{conv_stale_leaks}/{n_contra} ({100.0 * conv_stale_leaks / n_contra:.1f}%)",
            "clean_qa_accuracy": f"{conv_qa_correct}/{n_contra} ({100.0 * conv_qa_correct / n_contra:.1f}%)",
        },
        "contradiction_details": details,
        "additive_details": additive_details,
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_benchmark()
