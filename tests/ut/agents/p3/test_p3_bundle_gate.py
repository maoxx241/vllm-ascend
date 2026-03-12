from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from vllm_ascend.agent_runtime import (
    RawRequest,
    deployment_intake,
    ensure_task_bundle,
    evaluate_governor,
    feature_policy_resolver,
    flush_atomic_result,
    generic_spec,
    pack,
    refresh_continuation_state,
)
from vllm_ascend.agent_runtime.contracts import copy_example


def _deployment_card(exact_resolve_result, built_sqlite, agent_repo_root, request_id: str):
    intake = deployment_intake(
        RawRequest(
            request_id=request_id,
            user_text="在 A2 上确认 qwen3-next 的默认 prefill policy，并给出最小部署交付物",
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=[],
            created_at_hint="2026-03-13T13:04:00Z",
        )
    )
    request = {
        "schema_version": "kb-pack-request/v2",
        "request_id": request_id,
        "created_at": "2026-03-13T13:04:00Z",
        "intent": "intake_lookup",
        "repo_root": ".",
        "resolve_policy": "auto",
        "logical_domains": ["deployment_config"],
        "physical_shard_hints": ["repo_semantics", "validation"],
        "selectors": intake["selector_plan"]["selectors"],
        "must_have": ["baseline", "policy"],
        "nice_to_have": ["minimal artifact"],
        "evidence_refs": [],
        "budget_token_cap": 1200,
        "max_atoms": 10,
        "max_hops": 1,
        "include_evidence_stubs": False,
        "stop_after_first_sufficient": True,
        "emit_path": f".agents/kb/local/capsules/{request_id}.json",
    }
    response = pack(agent_repo_root, request=request, resolve_result=exact_resolve_result, merged_pack=built_sqlite)
    card = feature_policy_resolver(intake["selector_plan"], response)
    return intake["selector_plan"], card


def test_d1_first_complex_task_turn_creates_bundle(agent_repo_root) -> None:
    plan = copy_example("selector-plan.design.spec.json", root=agent_repo_root)
    state = generic_spec(plan, root=agent_repo_root)
    bundle_root = agent_repo_root / state["bundle_root"]
    assert (bundle_root / "spec.md").exists()
    assert (bundle_root / "plan.md").exists()
    assert (bundle_root / "checklist.md").exists()
    assert (bundle_root / "progress.md").exists()
    assert (bundle_root / "runtime" / "plans" / f"{plan['plan_id']}.json").exists()


def test_d2_atomic_complete_updates_progress_before_next_query(exact_resolve_result, built_sqlite, agent_repo_root) -> None:
    plan, card = _deployment_card(exact_resolve_result, built_sqlite, agent_repo_root, f"req-d2-{uuid4().hex[:8]}")
    refs = flush_atomic_result(plan, card, root=agent_repo_root)
    bundle_root = agent_repo_root / ".agents" / "tasks" / card["task_id"]
    progress_text = (bundle_root / "progress.md").read_text(encoding="utf-8")
    assert card["finding_summary"] in progress_text
    assert refs["card_ref"].endswith(".json")
    decision = evaluate_governor(
        selector_seed=copy_example("selector-seed.performance.expectation.json", root=agent_repo_root),
        selector_plan=copy_example("selector-plan.performance.atomic.json", root=agent_repo_root),
        continuation_state=None,
        progress_state={
            "bundle_exists": True,
            "has_unflushed_findings": False,
            "query_count_in_stage": 1,
            "opened_deep_refs_in_stage": 0,
            "seen_dedupe_keys": [],
            "last_flush_at": "2026-03-13T13:04:30Z",
            "session_budget_used": 0,
        },
        root=agent_repo_root,
    )
    assert decision["allow_query"] is True


def test_d3_needs_reroute_card_has_required_payload(exact_resolve_result, built_sqlite, agent_repo_root) -> None:
    plan, _ = _deployment_card(exact_resolve_result, built_sqlite, agent_repo_root, f"req-d3-{uuid4().hex[:8]}")
    response = {
        "capsule_text": "当前问题已越过 deployment_execution 的不改代码边界。",
        "atoms": [],
        "unknowns": ["需要代码改动"],
        "match_level": "exact",
        "estimated_tokens": 320,
    }
    card = feature_policy_resolver(plan, response, code_change_required=True)
    assert card["result_status"] == "needs_reroute"
    assert card["resolution_code"].startswith("reroute_")
    assert card["reroute"] is not None
    assert card["next_action"]["kind"] == "reroute_task"
    assert card["flush_required"] is True


def test_d4_continuation_refresh_requires_flushed_bundle_then_succeeds(exact_resolve_result, built_sqlite, agent_repo_root) -> None:
    plan, card = _deployment_card(exact_resolve_result, built_sqlite, agent_repo_root, f"req-d4-{uuid4().hex[:8]}")
    state = generic_spec(plan, root=agent_repo_root)
    with pytest.raises(ValueError):
        refresh_continuation_state(state["task_id"], state, root=agent_repo_root)
    flush_atomic_result(plan, card, root=agent_repo_root)
    path = refresh_continuation_state(state["task_id"], state, root=agent_repo_root)
    assert path.endswith("continuation-state.json")


def test_d5_continuation_state_contains_canonical_source_files(agent_repo_root) -> None:
    plan = copy_example("selector-plan.design.spec.json", root=agent_repo_root)
    state = generic_spec(plan, root=agent_repo_root)
    assert state["persistence_mode"] == "full_bundle"
    assert len(state["canonical_source_files"]) == 4
    assert all(Path(path).name in {"spec.md", "plan.md", "checklist.md", "progress.md"} for path in state["canonical_source_files"])


def test_d6_continuation_state_persistence_mode_is_always_full_bundle() -> None:
    with pytest.raises(ValueError):
        ensure_task_bundle("task-d6", "none")
