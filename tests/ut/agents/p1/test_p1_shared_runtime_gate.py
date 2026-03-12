from __future__ import annotations

import inspect

import pytest

import vllm_ascend.agent_runtime.shared as shared_mod
from vllm_ascend.agent_runtime.contracts import ContractError, copy_example
from vllm_ascend.agent_runtime.entrypoints import public_entry, vllm_ascend_assistant
from vllm_ascend.agent_runtime.shared import (
    RawRequest,
    build_selector_seed,
    evaluate_governor,
    generic_task_intake,
    intake_from_seed,
)


def test_b1_raw_request_to_selector_seed_is_deterministic() -> None:
    request = RawRequest(
        request_id="req-b1",
        user_text="估算 qwen3-next-32b 在 A2 TP4 BF16 8k 下的预期 TTFT 和吞吐",
        attachment_refs=[],
        inline_paths=[],
        inline_symbols=[],
        inline_errors=[],
        created_at_hint="2026-03-13T00:00:00Z",
    )
    assert build_selector_seed(request) == build_selector_seed(request)


def test_b2_pending_confirmation_blocks_atomic_plan() -> None:
    result = generic_task_intake(
        RawRequest(
            request_id="req-b2",
            user_text="需要做 code change 才能适配一个新模型，请给我 adaptation 方案",
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=[],
            created_at_hint="2026-03-13T00:00:01Z",
        )
    )
    assert result["blocked"] is True
    assert result["response_kind"] == "confirmation_gate"
    assert result["selector_plan"] is None


def test_b3_user_declined_blocks_plan() -> None:
    seed = copy_example("selector-seed.upstream.user-declined.json")
    result = intake_from_seed(seed)
    assert result["blocked"] is True
    assert result["response_kind"] == "confirmation_gate"
    assert result["selector_plan"] is None


def test_b4_and_f2_public_entry_direct_answer_skips_governor(monkeypatch) -> None:
    def _explode(**_: object) -> None:
        raise AssertionError("governor should not run on no-query paths")

    monkeypatch.setattr(shared_mod, "evaluate_governor", _explode)
    seed = public_entry(
        RawRequest(
            request_id="req-b4",
            user_text="解释 direct_answer 和 spec_plan_workflow 的区别",
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=[],
            created_at_hint="2026-03-13T00:00:02Z",
        )
    )
    result = vllm_ascend_assistant(
        RawRequest(
            request_id="req-b4",
            user_text="解释 direct_answer 和 spec_plan_workflow 的区别",
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=[],
            created_at_hint="2026-03-13T00:00:02Z",
        )
    )
    assert seed["execution_mode_hint"] == "direct_answer"
    assert result["response_kind"] == "direct_answer"
    assert result["selector_plan"] is None


def test_b5_query_stage_matrix_enforced() -> None:
    seed = copy_example("selector-seed.deployment.json")
    plan = copy_example("selector-plan.deployment.intake.json")
    plan["budget_class"] = "atomic"
    plan["capsule_type"] = "atomic_capsule"
    with pytest.raises(ContractError):
        evaluate_governor(
            selector_seed=seed,
            selector_plan=plan,
            continuation_state=None,
            progress_state={
                "bundle_exists": False,
                "has_unflushed_findings": False,
                "query_count_in_stage": 0,
                "opened_deep_refs_in_stage": 0,
                "seen_dedupe_keys": [],
                "last_flush_at": None,
                "session_budget_used": 0,
            },
        )


def test_b6_pending_flush_before_second_capsule_denied() -> None:
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


def test_b7_duplicate_plan_deduped() -> None:
    result = generic_task_intake(
        RawRequest(
            request_id="req-b7",
            user_text="估算 qwen3-next-32b 在 A2 TP4 BF16 8k 下的预期 TTFT 和吞吐",
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=[],
            created_at_hint="2026-03-13T00:00:03Z",
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


def test_b8_origin_query_execution_mode_mismatch_rejected() -> None:
    seed = copy_example("selector-seed.performance.expectation.json")
    plan = copy_example("selector-plan.performance.atomic.json")
    plan["execution_mode"] = "spec_plan_workflow"
    with pytest.raises(ContractError):
        evaluate_governor(
            selector_seed=seed,
            selector_plan=plan,
            continuation_state=None,
            progress_state={
                "bundle_exists": False,
                "has_unflushed_findings": False,
                "query_count_in_stage": 0,
                "opened_deep_refs_in_stage": 0,
                "seen_dedupe_keys": [],
                "last_flush_at": None,
                "session_budget_used": 0,
            },
        )


def test_b9_governor_stage_source_only_comes_from_selector_plan() -> None:
    signature = inspect.signature(evaluate_governor)
    assert "stage" not in signature.parameters


def test_f6_pending_confirmation_cannot_query_atomic() -> None:
    seed = copy_example("selector-seed.adaptation.pending-confirmation.json")
    plan = copy_example("selector-plan.performance.atomic.json")
    decision = evaluate_governor(
        selector_seed=seed,
        selector_plan=plan,
        continuation_state=None,
        progress_state={
            "bundle_exists": False,
            "has_unflushed_findings": False,
            "query_count_in_stage": 0,
            "opened_deep_refs_in_stage": 0,
            "seen_dedupe_keys": [],
            "last_flush_at": None,
            "session_budget_used": 0,
        },
    )
    assert decision["allow_query"] is False
    assert decision["denial_reason_code"] == "stage_disallows_query"


def test_f7_user_declined_cannot_query() -> None:
    seed = copy_example("selector-seed.upstream.user-declined.json")
    plan = copy_example("selector-plan.performance.atomic.json")
    decision = evaluate_governor(
        selector_seed=seed,
        selector_plan=plan,
        continuation_state=None,
        progress_state={
            "bundle_exists": False,
            "has_unflushed_findings": False,
            "query_count_in_stage": 0,
            "opened_deep_refs_in_stage": 0,
            "seen_dedupe_keys": [],
            "last_flush_at": None,
            "session_budget_used": 0,
        },
    )
    assert decision["allow_query"] is False
    assert decision["denial_reason_code"] == "stage_disallows_query"
