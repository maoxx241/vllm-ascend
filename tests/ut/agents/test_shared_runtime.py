from __future__ import annotations

import inspect

from vllm_ascend.agent_runtime.contracts import copy_example
from vllm_ascend.agent_runtime.shared import (
    RawRequest,
    evaluate_governor,
    generic_spec,
    generic_task_intake,
)


def test_generic_task_intake_deployment_flow() -> None:
    result = generic_task_intake(
        RawRequest(
            request_id="req-deploy-flow",
            user_text="在 A2 上确认 qwen3-next 的默认 prefill policy，并给出最小部署交付物",
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=[],
        )
    )
    assert result["blocked"] is False
    assert result["selector_seed"]["schema_version"] == "selector-seed/v3"
    assert result["selector_plan"]["schema_version"] == "selector-plan/v4"
    assert result["selector_plan"]["task_family"] == "deployment_execution"


def test_generic_task_intake_blocks_pending_confirmation() -> None:
    result = generic_task_intake(
        RawRequest(
            request_id="req-adapt-flow",
            user_text="需要做 code change 才能适配一个新模型，请给我 adaptation 方案",
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=[],
        )
    )
    assert result["blocked"] is True
    assert result["selector_seed"]["confirmation_required"] is True
    assert result["selector_seed"]["confirmation_status"] == "pending"
    assert result["selector_plan"] is None


def test_governor_signature_has_no_stage_parameter() -> None:
    signature = inspect.signature(evaluate_governor)
    assert "stage" not in signature.parameters


def test_governor_blocks_pending_flush() -> None:
    plan = copy_example("selector-plan.design.spec.json")
    seed = copy_example("selector-seed.performance.expectation.json")
    decision = evaluate_governor(
        selector_seed=seed,
        selector_plan=plan,
        continuation_state=None,
        progress_state={
            "bundle_exists": True,
            "has_unflushed_findings": True,
            "query_count_in_stage": 1,
            "opened_deep_refs_in_stage": 0,
            "seen_dedupe_keys": [],
            "last_flush_at": None,
            "session_budget_used": 0,
        },
    )
    assert decision["allow_query"] is False
    assert decision["denial_reason_code"] == "pending_flush"
    assert decision["flush_required"] is True


def test_governor_blocks_duplicate_plan() -> None:
    result = generic_task_intake(
        RawRequest(
            request_id="req-dup-flow",
            user_text="估算 qwen3-next-32b 在 A2 TP4 BF16 8k 下的预期 TTFT 和吞吐",
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=[],
        )
    )
    plan = result["selector_plan"]
    decision = evaluate_governor(
        selector_seed=result["selector_seed"],
        selector_plan=plan,
        continuation_state=None,
        progress_state={
            "bundle_exists": False,
            "has_unflushed_findings": False,
            "query_count_in_stage": 0,
            "opened_deep_refs_in_stage": 0,
            "seen_dedupe_keys": [f"{plan['request_id']}:{plan['query_stage']}:{plan['consumer_id']}"],
            "last_flush_at": None,
            "session_budget_used": 0,
        },
    )
    assert decision["allow_query"] is False
    assert decision["denial_reason_code"] == "duplicate_plan"


def test_generic_spec_creates_full_bundle() -> None:
    plan = copy_example("selector-plan.design.spec.json")
    state = generic_spec(plan)
    assert state["schema_version"] == "continuation-state/v4"
    assert state["persistence_mode"] == "full_bundle"
    assert len(state["canonical_source_files"]) == 4
