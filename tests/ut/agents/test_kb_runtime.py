from __future__ import annotations

import sqlite3

from vllm_ascend.agent_runtime.kb import pack


def test_resolve_exact_remote_tuple(exact_resolve_result: dict) -> None:
    assert exact_resolve_result["match_level"] == "exact"
    assert exact_resolve_result["runtime_tuple"]["soc"] == "A2"
    assert exact_resolve_result["runtime_tuple"]["cann"] == "8.5.0"


def test_build_local_creates_sqlite(built_sqlite) -> None:
    assert built_sqlite.exists()
    conn = sqlite3.connect(built_sqlite)
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        }
        assert "pack_meta" in tables
        assert "facts" in tables
        assert "capsules" in tables
    finally:
        conn.close()


def test_pack_model_expectation(exact_resolve_result: dict, built_sqlite, agent_repo_root) -> None:
    request = {
        "schema_version": "kb-pack-request/v2",
        "request_id": "req-pack-exp",
        "created_at": "2026-03-13T13:03:00Z",
        "intent": "model_expectation",
        "repo_root": ".",
        "resolve_policy": "auto",
        "logical_domains": ["validation_evidence", "deployment_config", "ascend_foundation"],
        "physical_shard_hints": ["validation", "repo_semantics", "hw_runtime_caps"],
        "selectors": {
            "files": [],
            "symbols": [],
            "entities": ["baseline.expectation.qwen3-next-32b.a2", "repo.policy.prefill"],
            "errors": [],
            "models": ["qwen3-next-32b"],
            "features": ["prefill", "decode", "tp4", "bf16", "ctx8k"],
            "hw": ["A2"],
            "commits": [],
            "prs": [],
            "versions": ["vllm-ascend@0.13.0"],
            "configs": ["tp4_bf16_ctx8k"],
        },
        "must_have": ["expected TTFT range", "expected throughput range", "memory headroom assumptions"],
        "nice_to_have": ["closest comparable baseline", "top sensitivity factors"],
        "evidence_refs": [],
        "budget_token_cap": 1500,
        "max_atoms": 10,
        "max_hops": 1,
        "include_evidence_stubs": True,
        "stop_after_first_sufficient": True,
        "emit_path": ".agents/kb/local/capsules/req-pack-exp.json",
    }
    response = pack(agent_repo_root, request=request, resolve_result=exact_resolve_result, merged_pack=built_sqlite)
    assert response["match_level"] == "exact"
    assert response["capsule_type"] == "atomic_capsule"
    assert "TTFT" in response["capsule_text"]
    assert response["unknowns"]


def test_pack_validation_selection_returns_tests(exact_resolve_result: dict, built_sqlite, agent_repo_root) -> None:
    request = {
        "schema_version": "kb-pack-request/v2",
        "request_id": "req-pack-val",
        "created_at": "2026-03-13T13:03:00Z",
        "intent": "validation_selection",
        "repo_root": ".",
        "resolve_policy": "auto",
        "logical_domains": ["validation_evidence"],
        "physical_shard_hints": ["validation"],
        "selectors": {
            "files": ["vllm_ascend/core/scheduler_dynamic_batch.py"],
            "symbols": [],
            "entities": [],
            "errors": [],
            "models": [],
            "features": ["dynamic_batching"],
            "hw": ["A2"],
            "commits": [],
            "prs": [],
            "versions": [],
            "configs": [],
        },
        "must_have": ["minimum required tests"],
        "nice_to_have": ["supplemental smoke"],
        "evidence_refs": [],
        "budget_token_cap": 1200,
        "max_atoms": 10,
        "max_hops": 1,
        "include_evidence_stubs": True,
        "stop_after_first_sufficient": True,
        "emit_path": ".agents/kb/local/capsules/req-pack-val.json",
    }
    response = pack(agent_repo_root, request=request, resolve_result=exact_resolve_result, merged_pack=built_sqlite)
    refs = [ref for atom in response["atoms"] for ref in atom["source_refs"]]
    assert "tests/ut/core/test_scheduler_dynamic_batch.py" in refs
    assert "tests/e2e/singlecard/test_async_scheduling.py" in refs
