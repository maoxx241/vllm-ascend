from __future__ import annotations

from vllm_ascend.agent_runtime import (
    RawRequest,
    adaptation_intake,
    compile_pack_request,
    design_analysis_intake,
    evaluate_governor,
    generic_spec,
    operator_development_intake,
    upstream_sync_intake,
)
from vllm_ascend.agent_runtime.contracts import copy_example


def _governor(plan: dict, root) -> dict:
    seed = copy_example("selector-seed.performance.expectation.json", root=root)
    seed["request_id"] = plan["request_id"]
    seed["task_family_candidates"] = [plan["task_family"]]
    seed["execution_mode_hint"] = plan["execution_mode"]
    seed["analysis_depth_hint"] = plan["analysis_depth"]
    seed["deliverable_contract_hint"] = plan["deliverable_contract"]
    seed["confirmation_required"] = False
    seed["confirmation_status"] = "not_needed"
    return evaluate_governor(
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
        root=root,
    )


def test_f501_adaptation_requires_confirmation_gate() -> None:
    result = adaptation_intake(
        RawRequest(
            request_id="req-p5-adapt",
            user_text="需要改代码适配一个新模型，请给 adaptation 方案",
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=[],
            created_at_hint="2026-03-13T14:20:00Z",
        )
    )
    assert result["blocked"] is True
    assert result["response_kind"] == "confirmation_gate"
    assert result["selector_seed"]["task_family_candidates"][0] == "adaptation"
    assert result["selector_plan"] is None


def test_f501_route_unknown_cases_fall_back_to_design_analysis() -> None:
    result = design_analysis_intake(
        RawRequest(
            request_id="req-p5-design-route",
            user_text="这个需求可能要同步上游也可能要本地适配，先帮我做路线分析",
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=[],
            created_at_hint="2026-03-13T14:20:01Z",
        )
    )
    assert result["selector_plan"]["task_family"] == "design_analysis"
    assert result["selector_plan"]["execution_mode"] == "spec_plan_workflow"
    assert result["selector_plan"]["query_stage"] == "spec_plan"


def test_f502_single_upstream_delta_can_be_direct_atomic(agent_repo_root) -> None:
    result = upstream_sync_intake(
        RawRequest(
            request_id="req-p5-upstream-atomic",
            user_text="同步单个 upstream delta，并说明影响面",
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=[],
            created_at_hint="2026-03-13T14:20:02Z",
        )
    )
    plan = result["selector_plan"]
    assert plan["task_family"] == "upstream_sync"
    assert plan["execution_mode"] == "direct_atomic_workflow"
    assert plan["query_stage"] == "atomic"
    request = compile_pack_request(plan, _governor(plan, agent_repo_root), root=agent_repo_root)
    assert request["intent"] == "upstream_delta"


def test_f502_release_sync_can_be_spec_plan(agent_repo_root) -> None:
    result = upstream_sync_intake(
        RawRequest(
            request_id="req-p5-upstream-spec",
            user_text="整理一个 release sync 方案，输出影响面和验证窗口",
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=[],
            created_at_hint="2026-03-13T14:20:03Z",
        )
    )
    plan = result["selector_plan"]
    assert plan["task_family"] == "upstream_sync"
    assert plan["execution_mode"] == "spec_plan_workflow"
    assert plan["query_stage"] == "spec_plan"
    state = generic_spec(plan, root=agent_repo_root)
    assert state["task_family"] == "upstream_sync"
    assert state["persistence_mode"] == "full_bundle"


def test_f503_design_analysis_spec_plan_writes_continuation_state(agent_repo_root) -> None:
    result = design_analysis_intake(
        RawRequest(
            request_id="req-p5-design-spec",
            user_text="prefill 路线和上游 feature 路线冲突，帮我做设计分析",
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=[],
            created_at_hint="2026-03-13T14:20:04Z",
        )
    )
    state = generic_spec(result["selector_plan"], root=agent_repo_root)
    assert state["task_family"] == "design_analysis"
    assert state["current_stage"] == "spec_plan"
    assert state["persistence_mode"] == "full_bundle"


def test_f504_operator_capability_gap_requires_confirmation_gate() -> None:
    result = operator_development_intake(
        RawRequest(
            request_id="req-p5-operator",
            user_text="需要新增自定义算子来支持这个路径，先评估 operator 能力缺口",
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=[],
            created_at_hint="2026-03-13T14:20:05Z",
        )
    )
    assert result["blocked"] is True
    assert result["response_kind"] == "confirmation_gate"
    assert result["selector_seed"]["task_family_candidates"][0] == "operator_development"
    assert result["selector_plan"] is None
