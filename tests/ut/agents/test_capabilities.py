from __future__ import annotations

from vllm_ascend.agent_runtime.capabilities import (
    change_impact_test_selector,
    feature_policy_resolver,
    model_expected_performance_estimator,
    single_profile_breakdown,
)
from vllm_ascend.agent_runtime.shared import RawRequest, generic_task_intake
from vllm_ascend.agent_runtime.kb import pack


def test_deployment_capability_card(exact_resolve_result, built_sqlite, agent_repo_root) -> None:
    intake = generic_task_intake(
        RawRequest(
            request_id="req-cap-deploy",
            user_text="在 A2 上确认 qwen3-next 的默认 prefill policy，并给出最小部署交付物",
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=[],
        )
    )
    request = {
        "schema_version": "kb-pack-request/v2",
        "request_id": "req-cap-deploy",
        "created_at": "2026-03-13T13:03:00Z",
        "intent": "deployment_lookup",
        "repo_root": ".",
        "resolve_policy": "auto",
        "logical_domains": ["deployment_config", "validation_evidence"],
        "physical_shard_hints": ["repo_semantics", "validation"],
        "selectors": intake["selector_plan"]["selectors"],
        "must_have": ["默认策略", "已知限制", "最小交付物"],
        "nice_to_have": ["相关验证记录"],
        "evidence_refs": [],
        "budget_token_cap": 1200,
        "max_atoms": 10,
        "max_hops": 1,
        "include_evidence_stubs": True,
        "stop_after_first_sufficient": True,
        "emit_path": ".agents/kb/local/capsules/req-cap-deploy.json",
    }
    response = pack(agent_repo_root, request=request, resolve_result=exact_resolve_result, merged_pack=built_sqlite)
    card = feature_policy_resolver(intake["selector_plan"], response)
    assert card["result_status"] == "complete"
    assert card["task_family"] == "deployment_execution"
    assert card["reroute"] is None


def test_performance_expectation_capability_card(exact_resolve_result, built_sqlite, agent_repo_root) -> None:
    intake = generic_task_intake(
        RawRequest(
            request_id="req-cap-exp",
            user_text="估算 qwen3-next-32b 在 A2 TP4 BF16 8k 下的预期 TTFT、吞吐和显存范围",
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=[],
        )
    )
    request = {
        "schema_version": "kb-pack-request/v2",
        "request_id": "req-cap-exp",
        "created_at": "2026-03-13T13:03:00Z",
        "intent": "model_expectation",
        "repo_root": ".",
        "resolve_policy": "auto",
        "logical_domains": ["validation_evidence", "deployment_config", "ascend_foundation"],
        "physical_shard_hints": ["validation", "repo_semantics", "hw_runtime_caps"],
        "selectors": intake["selector_plan"]["selectors"],
        "must_have": ["expected TTFT range", "expected throughput range", "memory headroom assumptions"],
        "nice_to_have": ["closest comparable baseline", "top sensitivity factors"],
        "evidence_refs": [],
        "budget_token_cap": 1500,
        "max_atoms": 10,
        "max_hops": 1,
        "include_evidence_stubs": True,
        "stop_after_first_sufficient": True,
        "emit_path": ".agents/kb/local/capsules/req-cap-exp.json",
    }
    response = pack(agent_repo_root, request=request, resolve_result=exact_resolve_result, merged_pack=built_sqlite)
    card = model_expected_performance_estimator(intake["selector_plan"], response)
    assert card["result_status"] == "complete"
    assert card["atomic_skill"] == "model-expected-performance-estimator"
    assert "TTFT" in card["finding_summary"]


def test_single_profile_breakdown_partial_card(exact_resolve_result, built_sqlite, agent_repo_root) -> None:
    intake = generic_task_intake(
        RawRequest(
            request_id="req-cap-perf",
            user_text="用户给出了一次 prefill regression profile，需要解释 TTFT 差异",
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=["prefill regression"],
        )
    )
    request = {
        "schema_version": "kb-pack-request/v2",
        "request_id": "req-cap-perf",
        "created_at": "2026-03-13T13:03:00Z",
        "intent": "perf_breakdown",
        "repo_root": ".",
        "resolve_policy": "auto",
        "logical_domains": ["validation_evidence", "deployment_config"],
        "physical_shard_hints": ["validation", "repo_semantics"],
        "selectors": intake["selector_plan"]["selectors"],
        "must_have": ["局部瓶颈解释"],
        "nice_to_have": ["baseline profile"],
        "evidence_refs": [],
        "budget_token_cap": 1500,
        "max_atoms": 10,
        "max_hops": 1,
        "include_evidence_stubs": True,
        "stop_after_first_sufficient": True,
        "emit_path": ".agents/kb/local/capsules/req-cap-perf.json",
    }
    response = pack(agent_repo_root, request=request, resolve_result=exact_resolve_result, merged_pack=built_sqlite)
    card = single_profile_breakdown(intake["selector_plan"], response)
    assert card["result_status"] == "partial"
    assert "baseline" in "".join(card["residual_unknowns"])


def test_validation_capability_card(exact_resolve_result, built_sqlite, agent_repo_root) -> None:
    intake = generic_task_intake(
        RawRequest(
            request_id="req-cap-val",
            user_text="根据 diff 收口 dynamic batching 相关最小必跑集",
            attachment_refs=[],
            inline_paths=["vllm_ascend/core/scheduler_dynamic_batch.py"],
            inline_symbols=[],
            inline_errors=[],
        )
    )
    request = {
        "schema_version": "kb-pack-request/v2",
        "request_id": "req-cap-val",
        "created_at": "2026-03-13T13:03:00Z",
        "intent": "validation_selection",
        "repo_root": ".",
        "resolve_policy": "auto",
        "logical_domains": ["validation_evidence"],
        "physical_shard_hints": ["validation"],
        "selectors": intake["selector_plan"]["selectors"],
        "must_have": ["minimum required tests"],
        "nice_to_have": ["extra smoke"],
        "evidence_refs": [],
        "budget_token_cap": 1200,
        "max_atoms": 10,
        "max_hops": 1,
        "include_evidence_stubs": True,
        "stop_after_first_sufficient": True,
        "emit_path": ".agents/kb/local/capsules/req-cap-val.json",
    }
    response = pack(agent_repo_root, request=request, resolve_result=exact_resolve_result, merged_pack=built_sqlite)
    card = change_impact_test_selector(intake["selector_plan"], response)
    assert card["result_status"] == "complete"
    assert card["atomic_skill"] == "change-impact-test-selector"
