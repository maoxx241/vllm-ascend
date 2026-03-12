from __future__ import annotations

import sqlite3

import pytest

import vllm_ascend.agent_runtime.kb as kb_mod
from vllm_ascend.agent_runtime.contracts import ContractError
from vllm_ascend.agent_runtime.kb import build_local, pack, resolve


def _request(intent: str, request_id: str, budget: int = 1500) -> dict:
    return {
        "schema_version": "kb-pack-request/v2",
        "request_id": request_id,
        "created_at": "2026-03-13T13:03:00Z",
        "intent": intent,
        "repo_root": ".",
        "resolve_policy": "auto",
        "logical_domains": ["validation_evidence", "deployment_config"],
        "physical_shard_hints": ["validation", "repo_semantics"],
        "selectors": {
            "files": [],
            "symbols": [],
            "entities": [],
            "errors": [],
            "models": ["qwen3-next-32b"],
            "features": ["prefill", "decode", "tp4", "bf16", "ctx8k"],
            "hw": ["A2"],
            "commits": [],
            "prs": [],
            "versions": ["vllm-ascend@feature/agent-runtime-v33-p3-mvp"],
            "configs": ["tp4_bf16_ctx8k"],
        },
        "must_have": ["baseline comparison"],
        "nice_to_have": ["closest comparable baseline"],
        "evidence_refs": ["profile:baseline", "profile:current"] if intent == "perf_breakdown" else [],
        "budget_token_cap": budget,
        "max_atoms": 10,
        "max_hops": 1,
        "include_evidence_stubs": True,
        "stop_after_first_sufficient": True,
        "emit_path": f".agents/kb/local/capsules/{request_id}.json",
    }


def test_c1_resolve_exact_tuple(exact_resolve_result: dict) -> None:
    assert exact_resolve_result["match_level"] == "exact"
    assert exact_resolve_result["runtime_tuple"]["soc"] == "A2"
    assert exact_resolve_result["runtime_tuple"]["cann"] == "8.5.0"


def test_c2_resolve_fallback_tuple(agent_repo_root) -> None:
    result = resolve(
        agent_repo_root,
        request_id="req-c2",
        overrides={
            "soc": "A2",
            "cann": "8.5.0",
            "torch": "2.9.0",
            "torch_npu": "2.9.0",
            "python": "3.11",
            "repo_sha": "deadbeef",
            "paired_vllm_ref": "e39257a552d18ae9abb6ba1bbe65865d385ea764",
        },
    )
    assert result["match_level"] == "compatible"
    assert result["warnings"]


def test_c3_resolve_miss_returns_unknown(agent_repo_root) -> None:
    result = resolve(
        agent_repo_root,
        request_id="req-c3",
        overrides={
            "soc": "unknown",
            "cann": "unknown",
            "torch": "unknown",
            "torch_npu": "unknown",
            "python": "unknown",
            "repo_sha": "deadbeef",
            "paired_vllm_ref": "unknown",
        },
    )
    assert result["match_level"] == "unknown"
    assert "soc" in result["missing"]


def test_c4_build_local_repo_only_success(agent_repo_root, exact_resolve_result: dict, tmp_path) -> None:
    emit_sqlite = tmp_path / "current.sqlite"
    build_local(agent_repo_root, resolve_result=exact_resolve_result, emit_sqlite=emit_sqlite)
    conn = sqlite3.connect(emit_sqlite)
    try:
        source_ids = {row[0] for row in conn.execute("SELECT source_id FROM sources")}
        assert "source-custom-ops-register" in source_ids
        assert "source-val-async" in source_ids
    finally:
        conn.close()


def test_c5_pack_under_budget_returns_capsule(exact_resolve_result: dict, built_sqlite, agent_repo_root) -> None:
    request = _request("model_expectation", "req-c5")
    request["logical_domains"] = ["validation_evidence", "deployment_config", "ascend_foundation"]
    request["physical_shard_hints"] = ["validation", "repo_semantics", "hw_runtime_caps"]
    request["must_have"] = ["expected TTFT range", "expected throughput range", "memory headroom assumptions"]
    response = pack(agent_repo_root, request=request, resolve_result=exact_resolve_result, merged_pack=built_sqlite)
    assert response["atoms"]
    assert response["deep_reference_stubs"]


def test_c6_pack_over_budget_returns_smaller_result_or_explicit_miss(exact_resolve_result: dict, built_sqlite, agent_repo_root) -> None:
    request = _request("model_expectation", "req-c6", budget=180)
    request["logical_domains"] = ["validation_evidence", "deployment_config", "ascend_foundation"]
    request["physical_shard_hints"] = ["validation", "repo_semantics", "hw_runtime_caps"]
    request["must_have"] = ["expected TTFT range", "expected throughput range", "memory headroom assumptions"]
    response = pack(agent_repo_root, request=request, resolve_result=exact_resolve_result, merged_pack=built_sqlite)
    assert response["warnings"]
    assert response["estimated_tokens"] <= 180
    assert not response["atoms"] or "budget cap prevented a sufficient capsule" in response["unknowns"]


def test_c7_pack_miss_returns_unknowns(exact_resolve_result: dict, built_sqlite, agent_repo_root) -> None:
    request = _request("design_lookup", "req-c7")
    request["logical_domains"] = ["knowledge_governance"]
    request["physical_shard_hints"] = []
    response = pack(agent_repo_root, request=request, resolve_result=exact_resolve_result, merged_pack=built_sqlite)
    assert response["unknowns"]


def test_c8_model_expectation_is_deterministic(exact_resolve_result: dict, built_sqlite, agent_repo_root) -> None:
    request = _request("model_expectation", "req-c8")
    request["logical_domains"] = ["validation_evidence", "deployment_config", "ascend_foundation"]
    request["physical_shard_hints"] = ["validation", "repo_semantics", "hw_runtime_caps"]
    request["must_have"] = ["expected TTFT range", "expected throughput range", "memory headroom assumptions"]
    left = pack(agent_repo_root, request=request, resolve_result=exact_resolve_result, merged_pack=built_sqlite)
    right = pack(agent_repo_root, request=request, resolve_result=exact_resolve_result, merged_pack=built_sqlite)
    assert left == right


def test_doctor_returns_nonzero_on_failure(monkeypatch) -> None:
    def _boom(root=None):
        raise ContractError("doctor failed")

    monkeypatch.setattr(kb_mod, "run_contract_checks", _boom)
    assert kb_mod.main(["doctor"]) == 1


def test_f4_exact_miss_is_not_silent(agent_repo_root) -> None:
    result = resolve(
        agent_repo_root,
        request_id="req-f4",
        overrides={
            "soc": "A2",
            "cann": "8.5.0",
            "torch": "2.9.0",
            "torch_npu": "unknown",
            "python": "3.11",
            "repo_sha": "deadbeef",
            "paired_vllm_ref": "e39257a552d18ae9abb6ba1bbe65865d385ea764",
        },
    )
    assert result["match_level"] == "compatible"
    assert result["warnings"]
