from __future__ import annotations

from pathlib import Path

import pytest

from vllm_ascend.agent_runtime import (
    RawRequest,
    change_impact_test_selector,
    comparative_profile_breakdown,
    coverage_gap_analyzer,
    deployment_artifact_packager,
    deployment_config_synthesizer,
    deployment_intake,
    feature_policy_resolver,
    model_expected_performance_estimator,
    perf_intake,
    single_profile_breakdown,
    validation_strategy_intake,
    vllm_ascend_assistant,
)
from vllm_ascend.agent_runtime.contracts import ContractError, copy_example, validate_instance
from vllm_ascend.agent_runtime.kb import pack


def _pack_request(selector_plan: dict, request_id: str, intent: str, domains: list[str]) -> dict:
    return {
        "schema_version": "kb-pack-request/v2",
        "request_id": request_id,
        "created_at": "2026-03-13T13:05:00Z",
        "intent": intent,
        "repo_root": ".",
        "resolve_policy": "auto",
        "logical_domains": domains,
        "physical_shard_hints": ["repo_semantics", "validation"],
        "selectors": selector_plan["selectors"],
        "must_have": ["baseline comparison"],
        "nice_to_have": ["closest comparable baseline"],
        "evidence_refs": ["profile:baseline", "profile:current"] if intent == "perf_breakdown" else [],
        "budget_token_cap": 1500,
        "max_atoms": 10,
        "max_hops": 1,
        "include_evidence_stubs": True,
        "stop_after_first_sufficient": True,
        "emit_path": f".agents/kb/local/capsules/{request_id}.json",
    }


def test_e1_qwen3_next_a2_baseline_deployment(exact_resolve_result, built_sqlite, agent_repo_root) -> None:
    public = vllm_ascend_assistant(
        RawRequest(
            request_id="req-e1",
            user_text="在 A2 上确认 qwen3-next 的 baseline 和 prefill policy，并给出最小部署交付物",
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=[],
            created_at_hint="2026-03-13T13:05:00Z",
        )
    )
    result = deployment_intake(
        RawRequest(
            request_id="req-e1",
            user_text="在 A2 上确认 qwen3-next 的 baseline 和 prefill policy，并给出最小部署交付物",
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=[],
            created_at_hint="2026-03-13T13:05:00Z",
        )
    )
    assert public["selector_seed"]["task_family_candidates"][0] == "deployment_execution"
    assert result["selector_plan"]["task_family"] == "deployment_execution"
    assert result["selector_plan"]["execution_mode"] == "direct_atomic_workflow"
    request = _pack_request(result["selector_plan"], "req-e1-pack", "intake_lookup", ["deployment_config"])
    response = pack(agent_repo_root, request=request, resolve_result=exact_resolve_result, merged_pack=built_sqlite)
    card = feature_policy_resolver(result["selector_plan"], response)
    assert card["result_status"] == "complete"


def test_e2_single_profile_breakdown(exact_resolve_result, built_sqlite, agent_repo_root) -> None:
    intake = perf_intake(
        RawRequest(
            request_id="req-e2",
            user_text="用户给出了一次 prefill regression profile，需要解释 TTFT 差异",
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=["prefill regression"],
            created_at_hint="2026-03-13T13:05:01Z",
        )
    )
    assert intake["selector_plan"]["task_family"] == "performance_analysis"
    assert intake["selector_plan"]["consumer_id"] == "single-profile-breakdown"
    request = _pack_request(intake["selector_plan"], "req-e2-pack", "perf_breakdown", ["validation_evidence", "deployment_config"])
    request["evidence_refs"] = []
    response = pack(agent_repo_root, request=request, resolve_result=exact_resolve_result, merged_pack=built_sqlite)
    card = single_profile_breakdown(intake["selector_plan"], response)
    assert card["result_status"] == "partial"


def test_e3_model_expected_performance_envelope(exact_resolve_result, built_sqlite, agent_repo_root) -> None:
    intake = perf_intake(
        RawRequest(
            request_id="req-e3",
            user_text="估算 qwen3-next-32b 在 A2 TP4 BF16 8k 下的预期 TTFT、吞吐和显存范围",
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=[],
            created_at_hint="2026-03-13T13:05:02Z",
        )
    )
    assert intake["selector_plan"]["consumer_id"] == "model-expected-performance-estimator"
    request = _pack_request(
        intake["selector_plan"],
        "req-e3-pack",
        "model_expectation",
        ["validation_evidence", "deployment_config", "ascend_foundation"],
    )
    response = pack(agent_repo_root, request=request, resolve_result=exact_resolve_result, merged_pack=built_sqlite)
    card = model_expected_performance_estimator(intake["selector_plan"], response)
    assert card["result_status"] == "complete"
    assert "TTFT" in card["finding_summary"]


def test_e4_minimal_test_selection_from_diff(exact_resolve_result, built_sqlite, agent_repo_root) -> None:
    intake = validation_strategy_intake(
        RawRequest(
            request_id="req-e4",
            user_text="根据 diff 收口 dynamic batching 相关最小必跑集",
            attachment_refs=[],
            inline_paths=["vllm_ascend/core/scheduler_dynamic_batch.py"],
            inline_symbols=[],
            inline_errors=[],
            created_at_hint="2026-03-13T13:05:03Z",
        )
    )
    assert intake["selector_plan"]["task_family"] == "validation_strategy"
    assert intake["selector_plan"]["consumer_id"] == "change-impact-test-selector"
    request = _pack_request(intake["selector_plan"], "req-e4-pack", "validation_selection", ["validation_evidence"])
    response = pack(agent_repo_root, request=request, resolve_result=exact_resolve_result, merged_pack=built_sqlite)
    card = change_impact_test_selector(intake["selector_plan"], response)
    assert card["result_status"] == "complete"


def test_p3_deployment_follow_on_skills_produce_artifacts(exact_resolve_result, built_sqlite, agent_repo_root) -> None:
    intake = deployment_intake(
        RawRequest(
            request_id="req-p3-deploy",
            user_text="在 A2 上确认 qwen3-next 的 baseline 和 prefill policy，并给出最小部署交付物",
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=[],
            created_at_hint="2026-03-13T13:05:04Z",
        )
    )
    request = _pack_request(intake["selector_plan"], "req-p3-deploy-pack", "deployment_lookup", ["deployment_config", "validation_evidence"])
    response = pack(agent_repo_root, request=request, resolve_result=exact_resolve_result, merged_pack=built_sqlite)
    config_card = deployment_config_synthesizer(intake["selector_plan"], response)
    artifact_card = deployment_artifact_packager(intake["selector_plan"], response)
    assert config_card["produced_artifacts"]
    assert artifact_card["next_action"]["kind"] == "answer_user"


def test_p3_comparative_profile_breakdown_complete_with_baseline(exact_resolve_result, built_sqlite, agent_repo_root) -> None:
    intake = perf_intake(
        RawRequest(
            request_id="req-p3-compare",
            user_text="对照 baseline/current profile，解释 prefill 差异",
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=["prefill regression"],
            created_at_hint="2026-03-13T13:05:05Z",
        )
    )
    assert intake["selector_plan"]["consumer_id"] == "comparative-profile-breakdown"
    request = _pack_request(intake["selector_plan"], "req-p3-compare-pack", "perf_breakdown", ["validation_evidence", "deployment_config"])
    response = pack(agent_repo_root, request=request, resolve_result=exact_resolve_result, merged_pack=built_sqlite)
    card = comparative_profile_breakdown(intake["selector_plan"], response)
    assert card["result_status"] == "complete"


def test_p3_coverage_gap_analyzer_returns_low_confidence_gap(exact_resolve_result, built_sqlite, agent_repo_root) -> None:
    intake = validation_strategy_intake(
        RawRequest(
            request_id="req-p3-gap",
            user_text="分析 coverage gap，并给出补采建议",
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=[],
            created_at_hint="2026-03-13T13:05:06Z",
        )
    )
    assert intake["selector_plan"]["consumer_id"] == "coverage-gap-analyzer"
    request = _pack_request(intake["selector_plan"], "req-p3-gap-pack", "validation_selection", ["validation_evidence"])
    response = pack(agent_repo_root, request=request, resolve_result=exact_resolve_result, merged_pack=built_sqlite)
    card = coverage_gap_analyzer(intake["selector_plan"], response)
    assert card["result_status"] == "needs_more_evidence"
    assert card["confidence"] == "low"


def test_f1_canonical_skill_runtimes_do_not_read_raw_sqlite(agent_repo_root) -> None:
    skill_paths = [
        agent_repo_root / ".agents" / "skills" / "feature-policy-resolver" / "runtime.py",
        agent_repo_root / ".agents" / "skills" / "deployment-config-synthesizer" / "runtime.py",
        agent_repo_root / ".agents" / "skills" / "deployment-artifact-packager" / "runtime.py",
        agent_repo_root / ".agents" / "skills" / "single-profile-breakdown" / "runtime.py",
        agent_repo_root / ".agents" / "skills" / "comparative-profile-breakdown" / "runtime.py",
        agent_repo_root / ".agents" / "skills" / "model-expected-performance-estimator" / "runtime.py",
        agent_repo_root / ".agents" / "skills" / "change-impact-test-selector" / "runtime.py",
        agent_repo_root / ".agents" / "skills" / "coverage-gap-analyzer" / "runtime.py",
    ]
    for runtime_path in skill_paths:
        text = runtime_path.read_text(encoding="utf-8")
        assert "sqlite3" not in text
        assert ".sqlite" not in text


def test_f3_atomic_does_not_expand_unrelated_domains(exact_resolve_result, built_sqlite, agent_repo_root) -> None:
    intake = validation_strategy_intake(
        RawRequest(
            request_id="req-f3",
            user_text="根据 diff 收口 dynamic batching 相关最小必跑集",
            attachment_refs=[],
            inline_paths=["vllm_ascend/core/scheduler_dynamic_batch.py"],
            inline_symbols=[],
            inline_errors=[],
            created_at_hint="2026-03-13T13:05:07Z",
        )
    )
    request = _pack_request(intake["selector_plan"], "req-f3-pack", "validation_selection", ["validation_evidence"])
    response = pack(agent_repo_root, request=request, resolve_result=exact_resolve_result, merged_pack=built_sqlite)
    assert response["logical_domains"] == ["validation_evidence"]
    card = change_impact_test_selector(intake["selector_plan"], response)
    assert all("repo_semantics" not in source_ref for source_ref in card["source_refs"])


def test_f5_code_change_family_with_no_analysis_depth_fails(agent_repo_root) -> None:
    plan = copy_example("selector-plan.performance.atomic.json", root=agent_repo_root)
    plan["task_family"] = "adaptation"
    plan["deliverable_contract"] = "code_change_pack"
    plan["analysis_depth"] = "none"
    with pytest.raises(ContractError):
        validate_instance(plan, "selector-plan.schema.json", root=agent_repo_root)
