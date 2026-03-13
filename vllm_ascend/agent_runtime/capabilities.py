from __future__ import annotations

from typing import Any

from .contracts import copy_example, now_utc, validate_instance
from .paths import repo_root
from .strategy import (alternative_artifacts_from_atoms,
                       alternative_strategies_from_atoms,
                       documented_artifact_from_atoms,
                       documented_strategy_from_atoms,
                       selected_artifact_from_atoms,
                       selected_strategy_from_atoms)
from .topology import visible_devices

RENDER_PRESETS = {
    "qwen3_32b_a3_bf16": {
        "model_path": "/model/Qwen3-32B",
        "model_name": "qwen3-32b",
        "is_quantized": False,
    },
    "qwen3_32b_a3_w8a8": {
        "model_path": "/model/Qwen3-32B-W8A8",
        "model_name": "qwen3-32b-w8a8",
        "is_quantized": True,
    },
}


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


def _as_int(value: str | None) -> int | None:
    if value in {None, "", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _strategy_topology_label(strategy: dict[str, str] | None) -> str:
    if strategy is None:
        return "未冻结拓扑"
    parts: list[str] = []
    tp = _as_int(strategy.get("tp"))
    dp = _as_int(strategy.get("dp"))
    ep = _as_int(strategy.get("ep"))
    cards = _as_int(strategy.get("cards"))
    logical = _as_int(strategy.get("logical"))
    if tp:
        parts.append(f"TP{tp}")
    if dp and dp != 1:
        parts.append(f"DP{dp}")
    if ep and ep != 1:
        parts.append(f"EP{ep}")
    if cards:
        parts.append(f"{cards} cards")
    if logical:
        parts.append(f"{logical} logical NPUs")
    return " / ".join(parts) if parts else "未冻结拓扑"


def _artifact_label(artifact: dict[str, str] | None) -> str:
    if artifact is None:
        return "未冻结产物路径"
    kind = artifact.get("kind")
    if kind == "documented_native_deploy":
        return "documented native deploy"
    if kind == "documented_convert_then_deploy":
        return "documented convert-then-deploy"
    if kind == "inferred_convert_then_deploy":
        return "inferred convert-then-deploy"
    if kind == "unsupported_requires_choice":
        return "route choice required"
    return kind or "未冻结产物路径"


def _quantized_output_path(model_name: str, artifact: dict[str, str] | None) -> str:
    quant = artifact.get("quant") if artifact else None
    if quant == "quant_w8a8":
        return f"/model/{model_name}-W8A8"
    if quant == "quant_w4a8":
        return f"/model/{model_name}-W4A8"
    if quant == "quant_w4a4":
        return f"/model/{model_name}-W4A4"
    return f"/model/{model_name}-Quantized"


def _render_qwen3_launch_block(
    *,
    model_path: str,
    model_name: str,
    tensor_parallel: int,
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
            f"  --tensor-parallel-size {tensor_parallel} \\",
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


def _render_modelslim_conversion_block(
    *,
    model_path: str,
    model_name: str,
    output_path: str,
    logical_npus: int,
) -> str:
    lines = [
        "# Runbook only: choose the closest official ModelSlim example for the target model family.",
        "git clone https://gitcode.com/Ascend/msit.git -b br_release_MindStudio_8.3.0_20261231",
        "cd msit/msmodelslim",
        "bash install.sh",
        f"export ASCEND_RT_VISIBLE_DEVICES={visible_devices(logical_npus)}",
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False",
        f"export MODEL_PATH={model_path}",
        f"export SAVE_PATH={output_path}",
        "python3 /path/to/msmodelslim/example/<closest_example>.py \\",
        "  --model_path $MODEL_PATH \\",
        "  --save_path $SAVE_PATH \\",
        "  --trust_remote_code True",
    ]
    return "```bash\n" + "\n".join(lines) + "\n```"


def _render_strategy_script(strategy: dict[str, str]) -> str | None:
    preset_name = strategy.get("preset")
    if preset_name in {None, "", "none"}:
        return None
    preset = RENDER_PRESETS.get(preset_name)
    if preset is None:
        return None
    logical_npus = _as_int(strategy.get("logical"))
    tensor_parallel = _as_int(strategy.get("tp"))
    if logical_npus is None or tensor_parallel is None:
        return None
    return _render_qwen3_launch_block(
        model_path=preset["model_path"],
        model_name=preset["model_name"],
        tensor_parallel=tensor_parallel,
        logical_npus=logical_npus,
        is_quantized=preset["is_quantized"],
        conservative=strategy.get("kind") == "inferred_preserve_topology",
    )


def _selected_deployment_inputs(
    pack_response: dict[str, Any],
) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    return (
        selected_strategy_from_atoms(pack_response["atoms"]),
        selected_artifact_from_atoms(pack_response["atoms"]),
    )


def _deployment_notes(pack_response: dict[str, Any]) -> str | None:
    selected, selected_artifact = _selected_deployment_inputs(pack_response)
    if selected is None or selected_artifact is None:
        return None
    if selected.get("kind") == "unknown_or_reroute":
        return None
    if selected_artifact.get("kind") == "unsupported_requires_choice":
        return None
    script = _render_strategy_script(selected)
    if script is None:
        return None
    documented = documented_strategy_from_atoms(pack_response["atoms"])
    documented_artifact = documented_artifact_from_atoms(pack_response["atoms"])
    cards = _as_int(selected.get("cards"))
    topology_name = "single-card" if cards == 1 else _strategy_topology_label(selected)
    if selected_artifact.get("kind") in {"documented_convert_then_deploy", "inferred_convert_then_deploy"}:
        preset = RENDER_PRESETS.get(selected.get("preset") or "")
        logical_npus = _as_int(selected.get("logical"))
        if preset is None or logical_npus is None:
            return None
        conversion = _render_modelslim_conversion_block(
            model_path="/model/Qwen3-32B",
            model_name="Qwen3-32B",
            output_path=_quantized_output_path("Qwen3-32B", selected_artifact),
            logical_npus=logical_npus,
        )
        route_label = "documented" if selected_artifact.get("kind") == "documented_convert_then_deploy" else "inferred unvalidated"
        return (
            f"{route_label} conversion + deployment runbook preserving {topology_name} "
            f"({_strategy_topology_label(selected)}).\n\n"
            f"ModelSlim conversion runbook:\n\n{conversion}\n\n"
            f"Serve runbook for the converted artifact:\n\n{script}\n\n"
            f"documented comparison anchors: strategy={_strategy_topology_label(documented)}; "
            f"artifact={_artifact_label(documented_artifact)}."
        )
    if selected.get("kind") == "inferred_preserve_topology":
        return (
            f"inferred unvalidated deployment script preserving the requested topology "
            f"({topology_name}; {_strategy_topology_label(selected)}).\n\n{script}\n\n"
            f"documented best-performance baseline for comparison: {_strategy_topology_label(documented)}."
        )
    return f"documented deployment script for {_strategy_topology_label(selected)}:\n\n{script}"


def _deployment_reroute_card(
    selector_plan: dict[str, Any],
    pack_response: dict[str, Any],
    *,
    atomic_skill: str,
    root: Any | None = None,
) -> dict[str, Any]:
    root = root or repo_root()
    card = _base_card("atomic-result-card.reroute.json", selector_plan, atomic_skill, root=root)
    card = _apply_pack(card, pack_response)
    selected_artifact = selected_artifact_from_atoms(pack_response["atoms"])
    artifact_alternatives = alternative_artifacts_from_atoms(pack_response["atoms"])
    alternatives = alternative_strategies_from_atoms(pack_response["atoms"])
    if selected_artifact and selected_artifact.get("kind") == "unsupported_requires_choice":
        artifact_labels = ", ".join(_artifact_label(item) for item in artifact_alternatives) or "ModelSlim conversion / fp8-origin adaptation"
        card.update(
            {
                "task_family": "deployment_execution",
                "resolution_code": "reroute_family_boundary",
                "deliverable_fragment_summary": "当前 deployment 请求先要收口 artifact 路线：native FP8 直跑不受支持。",
                "next_action": {
                    "kind": "reroute_task",
                    "owner_stage": "intake",
                    "summary": "重新进入 Intake，转 design_analysis 收口 conversion 与 adaptation 路线",
                },
                "reroute": {
                    "target_family": "design_analysis",
                    "target_stage": "intake",
                    "reason": f"deployment family cannot emit a runbook until it chooses among {artifact_labels}.",
                    "carry_over_unknowns": pack_response["unknowns"] or ["需要先收口支持的量化产物路线"],
                },
                "produced_artifacts": [],
                "notes": None,
            }
        )
    else:
        alternative_labels = ", ".join(_strategy_topology_label(item) for item in alternatives) or "多个未冻结候选"
        card.update(
            {
                "task_family": "deployment_execution",
                "resolution_code": "reroute_family_boundary",
                "deliverable_fragment_summary": "当前 deployment 请求无法稳定收敛到单个 artifact；需要进入 design_analysis 收口策略。",
                "next_action": {
                    "kind": "reroute_task",
                    "owner_stage": "intake",
                    "summary": "重新进入 Intake，转 design_analysis 收口并行策略",
                },
                "reroute": {
                    "target_family": "design_analysis",
                    "target_stage": "intake",
                    "reason": f"deployment family cannot safely choose between {alternative_labels}.",
                    "carry_over_unknowns": pack_response["unknowns"] or ["需要在设计分析阶段收口并行策略"],
                },
                "produced_artifacts": [],
                "notes": None,
            }
        )
    validate_instance(card, "atomic-result-card.schema.json", root=root)
    return card


def feature_policy_resolver(
    selector_plan: dict[str, Any],
    pack_response: dict[str, Any],
    *,
    code_change_required: bool = False,
    root: Any | None = None,
) -> dict[str, Any]:
    root = root or repo_root()
    selected = selected_strategy_from_atoms(pack_response["atoms"])
    if code_change_required or (selected and selected.get("kind") == "unknown_or_reroute"):
        return _deployment_reroute_card(selector_plan, pack_response, atomic_skill="feature-policy-resolver", root=root)
    card = _base_card("atomic-result-card.complete.json", selector_plan, "feature-policy-resolver", root=root)
    card = _apply_pack(card, pack_response)
    validate_instance(card, "atomic-result-card.schema.json", root=root)
    return card


def deployment_config_synthesizer(selector_plan: dict[str, Any], pack_response: dict[str, Any], root: Any | None = None) -> dict[str, Any]:
    root = root or repo_root()
    selected = selected_strategy_from_atoms(pack_response["atoms"])
    selected_artifact = selected_artifact_from_atoms(pack_response["atoms"])
    if selected_artifact and selected_artifact.get("kind") == "unsupported_requires_choice":
        return _deployment_reroute_card(selector_plan, pack_response, atomic_skill="deployment-config-synthesizer", root=root)
    if selected and selected.get("kind") == "unknown_or_reroute":
        return _deployment_reroute_card(selector_plan, pack_response, atomic_skill="deployment-config-synthesizer", root=root)

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
    notes = _deployment_notes(pack_response)
    if notes:
        card["notes"] = notes
    validate_instance(card, "atomic-result-card.schema.json", root=root)
    return card


def deployment_artifact_packager(selector_plan: dict[str, Any], pack_response: dict[str, Any], root: Any | None = None) -> dict[str, Any]:
    root = root or repo_root()
    selected = selected_strategy_from_atoms(pack_response["atoms"])
    selected_artifact = selected_artifact_from_atoms(pack_response["atoms"])
    if selected_artifact and selected_artifact.get("kind") == "unsupported_requires_choice":
        return _deployment_reroute_card(selector_plan, pack_response, atomic_skill="deployment-artifact-packager", root=root)
    if selected and selected.get("kind") == "unknown_or_reroute":
        return _deployment_reroute_card(selector_plan, pack_response, atomic_skill="deployment-artifact-packager", root=root)

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
    if selected:
        if selected_artifact and selected_artifact.get("kind") in {"documented_convert_then_deploy", "inferred_convert_then_deploy"}:
            if selected.get("kind") == "inferred_preserve_topology":
                card["deliverable_fragment_summary"] = (
                    f"已按用户锁定拓扑 {_strategy_topology_label(selected)} 生成 conversion + serve 两阶段未验证 runbook。"
                )
            else:
                card["deliverable_fragment_summary"] = (
                    f"已按 {_strategy_topology_label(selected)} 收口 conversion + serve 两阶段 runbook。"
                )
        elif selected.get("kind") == "inferred_preserve_topology":
            card["deliverable_fragment_summary"] = (
                f"已按用户锁定拓扑 {_strategy_topology_label(selected)} 生成未验证脚本，并附带文档化对照基线。"
            )
        else:
            card["deliverable_fragment_summary"] = (
                f"已按 {_strategy_topology_label(selected)} 收口文档化 deployment artifact。"
            )
    notes = _deployment_notes(pack_response)
    if notes:
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
    selected = selected_strategy_from_atoms(pack_response["atoms"])
    selected_artifact = selected_artifact_from_atoms(pack_response["atoms"])
    if selected_artifact and selected_artifact.get("kind") == "unsupported_requires_choice":
        card["confidence"] = "low"
        card["notes"] = (
            "expected envelope remains conditional because native FP8 direct deployment is unsupported on A2/A3; "
            "choose ModelSlim conversion or fp8-origin adaptation before collapsing to a single route."
        )
    elif selected and selected.get("kind") == "unknown_or_reroute":
        card["confidence"] = "low"
    elif selected and selected.get("kind") == "inferred_preserve_topology":
        card["confidence"] = "medium"
    elif selected_artifact and selected_artifact.get("kind") == "inferred_convert_then_deploy":
        card["confidence"] = "medium"
    else:
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
