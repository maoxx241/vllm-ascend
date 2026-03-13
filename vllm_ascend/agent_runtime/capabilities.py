from __future__ import annotations

from typing import Any

from .contracts import copy_example, now_utc, validate_instance
from .paths import repo_root
from .topology import logical_npus_for_hw, requested_card_count_from_features, visible_devices


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


def _documented_qwen3_a3_topology() -> dict[str, int]:
    return {
        "physical_cards": 2,
        "logical_npus": 4,
        "tensor_parallel": 4,
    }


def _render_qwen3_a3_launch_block(
    *,
    model_path: str,
    model_name: str,
    logical_npus: int,
    is_quantized: bool,
    conservative: bool,
) -> str:
    quant_line = '  --quantization ascend \\\n' if is_quantized else ""
    env_lines = [
        f"export ASCEND_RT_VISIBLE_DEVICES={visible_devices(logical_npus)}",
        "export TASK_QUEUE_ENABLE=1",
    ]
    if not conservative and not is_quantized:
        env_lines.append('export OMP_PROC_BIND="false"')
    env_lines.append('export HCCL_OP_EXPANSION_MODE="AIV"')
    if is_quantized and not conservative:
        env_lines.append("export VLLM_ASCEND_ENABLE_FLASHCOMM1=1")
    if not is_quantized and not conservative:
        env_lines.append("export PAGED_ATTENTION_MASK_LEN=5500")

    lines = [*env_lines, f"vllm serve {model_path} \\", f"  --served-model-name {model_name} \\"]
    if is_quantized:
        lines.extend(
            [
                "  --trust-remote-code \\",
                "  --async-scheduling \\",
            ]
        )
    else:
        lines.extend(
            [
                "  --no-enable-prefix-caching \\",
            ]
        )
    if quant_line:
        lines.append(quant_line.rstrip())
    lines.extend(
        [
            "  --distributed-executor-backend mp \\",
            f"  --tensor-parallel-size {logical_npus} \\",
        ]
    )
    if not is_quantized:
        lines.append("  --trust-remote-code \\")
    if conservative:
        lines.extend(
            [
                "  --enforce-eager \\",
                "  --max-model-len 4096 \\",
                "  --max-num-batched-tokens 256 \\",
                "  --max-num-seqs 1 \\",
            ]
        )
    else:
        lines.extend(
            [
                f"  --max-model-len {'40960' if is_quantized else '36864'} \\",
                f"  --max-num-batched-tokens {'40960' if is_quantized else '36864'} \\",
            ]
        )
        if is_quantized:
            lines.append("  --reasoning-parser qwen3 \\")
    lines.extend(
        [
            "  --block-size 128 \\",
            "  --gpu-memory-utilization 0.9 \\",
            "  --port 8113 \\",
        ]
    )
    if is_quantized:
        lines.append(
            "  --additional-config '{\"weight_prefetch_config\":{\"enabled\":true}}'"
        )
    else:
        lines.append("  --additional-config '{\"enable_weight_nz_layout\":true}'")
    return "```bash\n" + "\n".join(lines) + "\n```"


def _deployment_notes(selector_plan: dict[str, Any]) -> str | None:
    selectors = selector_plan.get("selectors", {})
    models = selectors.get("models", [])
    hardware = selectors.get("hw", [])
    features = selectors.get("features", [])
    if not models or not hardware:
        return None

    primary_model = models[0]
    primary_hw = hardware[0]
    if primary_hw != "A3" or primary_model not in {"qwen3-32b", "qwen3-32b-w8a8"}:
        return None

    is_quantized = primary_model == "qwen3-32b-w8a8"
    model_path = "/model/Qwen3-32B-W8A8" if is_quantized else "/model/Qwen3-32B"
    model_name = "qwen3-32b-w8a8" if is_quantized else "qwen3-32b"
    documented_topology = _documented_qwen3_a3_topology()
    requested_card_count = requested_card_count_from_features(features)
    requested_logical_npus = logical_npus_for_hw(primary_hw, requested_card_count)
    if requested_card_count is not None and requested_card_count != documented_topology["physical_cards"]:
        conservative = requested_logical_npus is not None and requested_logical_npus < documented_topology["logical_npus"]
        assert requested_logical_npus is not None
        topology_label = "single-card" if requested_card_count == 1 else f"{requested_card_count}-card"
        requested_card_phrase = "1 card" if requested_card_count == 1 else f"{requested_card_count} cards"
        return (
            f"unverified {topology_label} attempt for the requested A3 topology. "
            f"On A3, 1 card = 2 logical NPUs, so {requested_card_phrase} = {requested_logical_npus} logical NPUs. "
            "The documented best-performance baseline is TP4 / 2 cards / 4 logical NPUs. "
            "The script below keeps the requested physical-card topology instead of silently collapsing it to the documented baseline, "
            "so it must be treated as inferred and unvalidated.\n\n"
            + _render_qwen3_a3_launch_block(
                model_path=model_path,
                model_name=model_name,
                logical_npus=requested_logical_npus,
                is_quantized=is_quantized,
                conservative=conservative,
            )
            + "\n\n"
            "documented best-performance baseline for comparison: TP4 / 2 cards / 4 logical NPUs on A3."
        )

    if is_quantized:
        return (
            "documented best-performance baseline for Qwen3-32B-W8A8 on A3 "
            "(TP4 / 2 cards / 4 logical NPUs):\n\n"
            + _render_qwen3_a3_launch_block(
                model_path=model_path,
                model_name=model_name,
                logical_npus=documented_topology["logical_npus"],
                is_quantized=True,
                conservative=False,
            )
        )

    return (
        "documented best-performance baseline for Qwen3-32B on A3 "
        "(TP4 / 2 cards / 4 logical NPUs):\n\n"
        + _render_qwen3_a3_launch_block(
            model_path=model_path,
            model_name=model_name,
            logical_npus=documented_topology["logical_npus"],
            is_quantized=False,
            conservative=False,
        )
    )


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
    notes = _deployment_notes(selector_plan)
    if notes:
        card["notes"] = notes
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
    notes = _deployment_notes(selector_plan)
    if notes:
        requested_card_count = requested_card_count_from_features(selector_plan.get("selectors", {}).get("features", []))
        if requested_card_count is not None and requested_card_count != _documented_qwen3_a3_topology()["physical_cards"]:
            card["deliverable_fragment_summary"] = (
                f"{requested_card_count} 卡请求未命中文档化基线；已基于用户要求返回未验证的 A3 拓扑推断脚本，并附带文档化 TP4 / 2 cards / 4 logical NPUs 对照基线。"
            )
        card["notes"] = notes
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


def log_triage(selector_plan: dict[str, Any], pack_response: dict[str, Any], root: Any | None = None) -> dict[str, Any]:
    root = root or repo_root()
    card = _base_card("atomic-result-card.complete.json", selector_plan, "log-triage", root=root)
    card = _apply_pack(card, pack_response)
    card["task_family"] = "debugging"
    validate_instance(card, "atomic-result-card.schema.json", root=root)
    return card


def cross_log_correlation(selector_plan: dict[str, Any], pack_response: dict[str, Any], root: Any | None = None) -> dict[str, Any]:
    root = root or repo_root()
    card = _base_card("atomic-result-card.complete.json", selector_plan, "cross-log-correlation", root=root)
    card = _apply_pack(card, pack_response)
    card["task_family"] = "debugging"
    card["deliverable_fragment_summary"] = "已对照多份日志并收口共同失败签名，可继续回给用户或进入下一轮 flush。"
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
