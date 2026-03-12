from __future__ import annotations

from typing import Any

from .contracts import copy_example, now_utc, validate_instance
from .paths import repo_root


def _base_card(template_name: str, selector_plan: dict[str, Any], atomic_skill: str, root: Any | None = None) -> dict[str, Any]:
    root = root or repo_root()
    card = copy_example(template_name, root=root)
    card.update(
        {
            "card_id": f"card-{selector_plan['request_id']}",
            "request_id": selector_plan["request_id"],
            "task_id": f"task-{selector_plan['request_id']}",
            "created_at": now_utc(),
            "task_family": selector_plan["task_family"],
            "atomic_skill": atomic_skill,
            "work_package_id": selector_plan["work_package_id"],
            "source_plan_id": selector_plan["plan_id"],
        }
    )
    return card


def _apply_pack(card: dict[str, Any], pack_response: dict[str, Any]) -> dict[str, Any]:
    card.update(
        {
            "finding_summary": pack_response["capsule_text"],
            "evidence_summary": [atom["summary"] for atom in pack_response["atoms"]],
            "residual_unknowns": pack_response["unknowns"],
            "source_refs": [ref for atom in pack_response["atoms"] for ref in atom["source_refs"]],
            "confidence": "medium" if pack_response["unknowns"] or pack_response["match_level"] != "exact" else "high",
            "token_estimate": pack_response["estimated_tokens"],
        }
    )
    return card


def feature_policy_resolver(
    selector_plan: dict[str, Any],
    pack_response: dict[str, Any],
    *,
    code_change_required: bool = False,
    root: Any | None = None,
) -> dict[str, Any]:
    root = root or repo_root()
    if code_change_required:
        card = _base_card("atomic-result-card.reroute.json", selector_plan, "feature-policy-resolver", root=root)
        card["reroute"]["carry_over_unknowns"] = pack_response["unknowns"] or ["需要代码改动后再重新路由"]
    else:
        card = _base_card("atomic-result-card.complete.json", selector_plan, "feature-policy-resolver", root=root)
    card = _apply_pack(card, pack_response)
    validate_instance(card, "atomic-result-card.schema.json", root=root)
    return card


def deployment_config_synthesizer(selector_plan: dict[str, Any], pack_response: dict[str, Any], root: Any | None = None) -> dict[str, Any]:
    root = root or repo_root()
    card = _base_card("atomic-result-card.complete.json", selector_plan, "deployment-config-synthesizer", root=root)
    card = _apply_pack(card, pack_response)
    card.update(
        {
            "deliverable_fragment_summary": "已生成最小配置草案，可继续打包脚本与运行说明。",
            "produced_artifacts": ["artifacts/deploy/config.yaml", "artifacts/deploy/runtime-env.sh"],
            "next_action": {
                "kind": "continue_atomic",
                "owner_stage": "atomic",
                "summary": "调用 deployment-artifact-packager 组装最终交付物",
            },
        }
    )
    validate_instance(card, "atomic-result-card.schema.json", root=root)
    return card


def deployment_artifact_packager(selector_plan: dict[str, Any], pack_response: dict[str, Any], root: Any | None = None) -> dict[str, Any]:
    root = root or repo_root()
    card = _base_card("atomic-result-card.complete.json", selector_plan, "deployment-artifact-packager", root=root)
    card = _apply_pack(card, pack_response)
    card.update(
        {
            "deliverable_fragment_summary": "已收口配置、脚本和最小验证步骤，可直接回给用户。",
            "produced_artifacts": [
                "artifacts/deploy/config.yaml",
                "artifacts/deploy/launch.sh",
                "artifacts/deploy/minimal-validation.md",
            ],
            "next_action": {
                "kind": "answer_user",
                "owner_stage": "none",
                "summary": "返回 deployment artifact pack",
            },
        }
    )
    validate_instance(card, "atomic-result-card.schema.json", root=root)
    return card


def single_profile_breakdown(selector_plan: dict[str, Any], pack_response: dict[str, Any], root: Any | None = None) -> dict[str, Any]:
    root = root or repo_root()
    card = _base_card("atomic-result-card.performance.partial.json", selector_plan, "single-profile-breakdown", root=root)
    card = _apply_pack(card, pack_response)
    validate_instance(card, "atomic-result-card.schema.json", root=root)
    return card


def comparative_profile_breakdown(selector_plan: dict[str, Any], pack_response: dict[str, Any], root: Any | None = None) -> dict[str, Any]:
    root = root or repo_root()
    baseline_missing = any("baseline" in item for item in pack_response["unknowns"])
    card = _base_card("atomic-result-card.performance.partial.json", selector_plan, "comparative-profile-breakdown", root=root)
    card = _apply_pack(card, pack_response)
    if not baseline_missing:
        card.update(
            {
                "result_status": "complete",
                "resolution_code": "work_package_closed",
                "deliverable_fragment_summary": "baseline/current 对照已收口，可直接输出 comparative breakdown。",
                "next_action": {
                    "kind": "answer_user",
                    "owner_stage": "none",
                    "summary": "返回 comparative profile breakdown",
                },
                "produced_artifacts": ["artifacts/perf/comparative-breakdown.md"],
            }
        )
    validate_instance(card, "atomic-result-card.schema.json", root=root)
    return card


def model_expected_performance_estimator(
    selector_plan: dict[str, Any],
    pack_response: dict[str, Any],
    root: Any | None = None,
) -> dict[str, Any]:
    root = root or repo_root()
    card = _base_card(
        "atomic-result-card.performance.expectation.complete.json",
        selector_plan,
        "model-expected-performance-estimator",
        root=root,
    )
    card = _apply_pack(card, pack_response)
    card["confidence"] = "medium" if pack_response["unknowns"] else "high"
    validate_instance(card, "atomic-result-card.schema.json", root=root)
    return card


def change_impact_test_selector(
    selector_plan: dict[str, Any],
    pack_response: dict[str, Any],
    root: Any | None = None,
) -> dict[str, Any]:
    root = root or repo_root()
    card = _base_card(
        "atomic-result-card.validation.complete.json",
        selector_plan,
        "change-impact-test-selector",
        root=root,
    )
    card = _apply_pack(card, pack_response)
    validate_instance(card, "atomic-result-card.schema.json", root=root)
    return card


def coverage_gap_analyzer(selector_plan: dict[str, Any], pack_response: dict[str, Any], root: Any | None = None) -> dict[str, Any]:
    root = root or repo_root()
    card = _base_card("atomic-result-card.performance.partial.json", selector_plan, "coverage-gap-analyzer", root=root)
    card = _apply_pack(card, pack_response)
    card.update(
        {
            "task_family": "validation_strategy",
            "result_status": "needs_more_evidence",
            "resolution_code": "validation_gap",
            "confidence": "low",
            "impacted_surfaces": ["validation coverage", "asset completeness"],
            "deliverable_fragment_summary": "已给出低置信覆盖缺口与补采建议，但仍缺少完整资产。",
            "next_action": {
                "kind": "request_more_evidence",
                "owner_stage": "atomic",
                "summary": "补齐缺失验证资产后再刷新 coverage gap 结论",
            },
            "produced_artifacts": ["artifacts/validation/coverage-gaps.md"],
            "notes": "coverage-gap-analyzer 只输出缺口和补采建议，不做 root cause。",
        }
    )
    validate_instance(card, "atomic-result-card.schema.json", root=root)
    return card
