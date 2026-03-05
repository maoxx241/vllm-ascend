#!/usr/bin/env python3
"""Regression tests for deployment package rendering."""

from __future__ import annotations

import tempfile
from pathlib import Path

from build_global_param_kb import main as build_main
from render_deploy_package import render_package


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> int:
    build_main()

    with tempfile.TemporaryDirectory(prefix="deploy_pkg_test_") as td:
        base = Path(td)

        primary = render_package(
            output_dir=base / "primary",
            model_profile="qwen3-32b-w8a8",
            model_path_override=None,
            hardware_type="Atlas A2",
            npu_count=4,
            port_override=None,
            text="帮我开图并开启w8a8和权重预取，tp4部署",
            features_input=[],
        )

        start_path = Path(primary["generated_commands"]["start_script"])
        validate_path = Path(primary["generated_commands"]["validate_script"])
        rollback_path = Path(primary["generated_commands"]["rollback_script"])

        assert start_path.exists(), "start.sh should exist"
        assert validate_path.exists(), "validate.sh should exist"
        assert rollback_path.exists(), "rollback.sh should exist"

        start_text = _read(start_path)
        assert "--quantization ascend" in start_text
        assert "--compilation-config" in start_text
        assert "weight_prefetch_config" in start_text
        assert "--tensor-parallel-size 4" in start_text

        plan = primary["deployment_plan"]
        assert "deployment_plan" in primary
        assert "generated_commands" in primary
        assert "validation_steps" in primary
        assert "rollback_steps" in primary
        assert plan["model_profile"] == "qwen3-32b-w8a8"
        assert "compatibility" in plan
        assert plan["compatibility"]["blocked_features"] == []
        assert plan["compatibility"]["downgraded_features"] == []
        assert isinstance(plan["compatibility"]["reasonability_checks"], list)
        assert plan["model_knowledge"]["architecture"]["has_moe_layers"] is False
        assert isinstance(plan["evidence_block"], list) and plan["evidence_block"], "evidence_block should exist"
        assert isinstance(plan["conflict_alerts"], list), "conflict_alerts should exist"
        assert any(section["feature"] == "graph_mode" for section in plan["evidence_block"])
        for section in plan["evidence_block"]:
            for item in section["items"]:
                assert "confidence" in item
                assert "status" in item
                assert "definition_ref" in item

        backup = render_package(
            output_dir=base / "backup",
            model_profile="qwen3-next-80b-a3b-instruct-w8a8",
            model_path_override=None,
            hardware_type="Atlas A3",
            npu_count=8,
            port_override=18000,
            text="部署next模型，开ep和投机",
            features_input=[],
        )

        backup_start = Path(backup["generated_commands"]["start_script"])
        backup_text = _read(backup_start)
        assert "Qwen3-Next-80B-A3B-Instruct-W8A8" in backup_text
        assert "--port 18000" in backup_text
        assert "--enable-expert-parallel" in backup_text
        assert "--speculative-config" in backup_text

        blocked_case = render_package(
            output_dir=base / "blocked",
            model_profile="qwen3-32b-w8a8",
            model_path_override=None,
            hardware_type="Atlas A2",
            npu_count=4,
            port_override=None,
            text="给qwen3-32b-w8a8开int4和ep",
            features_input=[],
        )
        blocked_plan = blocked_case["deployment_plan"]
        blocked_items = blocked_plan["compatibility"]["blocked_features"]
        blocked_features = {item["feature"] for item in blocked_items}
        assert "int4_quantization" in blocked_features
        assert "expert_parallel" in blocked_features
        ep_item = next(item for item in blocked_items if item["feature"] == "expert_parallel")
        assert "no MoE layers" in ep_item["reason"]
        assert ep_item["source"] in {"model_knowledge_inference", "model_profile_constraint"}
        assert any("Blocked feature 'int4_quantization'" in risk for risk in blocked_plan["risks"])
        assert any("Blocked feature 'expert_parallel'" in risk for risk in blocked_plan["risks"])
        assert isinstance(blocked_plan["conflict_alerts"], list)

        blocked_start_text = _read(Path(blocked_case["generated_commands"]["start_script"]))
        assert "--enable-expert-parallel" not in blocked_start_text
        assert "int4" not in blocked_start_text.lower()

        ambiguous = render_package(
            output_dir=base / "ambiguous",
            model_profile="qwen3-32b-w8a8",
            model_path_override=None,
            hardware_type="Atlas A2",
            npu_count=4,
            port_override=None,
            text="开并行提吞吐",
            features_input=[],
        )
        risks = ambiguous["deployment_plan"]["risks"]
        assert any("clarification" in risk for risk in risks), "Ambiguous input should add clarification risk"

        cp_low_card = render_package(
            output_dir=base / "cp_low_card",
            model_profile="qwen3-32b-w8a8",
            model_path_override=None,
            hardware_type="Atlas A2",
            npu_count=4,
            port_override=None,
            text="开context parallel",
            features_input=[],
        )
        cp_risks = cp_low_card["deployment_plan"]["risks"]
        cp_downgraded = cp_low_card["deployment_plan"]["compatibility"]["downgraded_features"]
        assert any(item["feature"] == "context_parallel" for item in cp_downgraded), (
            "CP on low-card setup should be downgraded by model/hardware knowledge."
        )
        assert any("context_parallel" in risk for risk in cp_risks), (
            "CP on low-card setup should report risk."
        )

        all_features = render_package(
            output_dir=base / "all_features",
            model_profile="qwen3-next-80b-a3b-instruct-w8a8",
            model_path_override=None,
            hardware_type="Atlas A3",
            npu_count=8,
            port_override=19000,
            text="开图+量化+dp+ep+lora+投机+sleep+预取+prefix cache",
            features_input=[],
        )
        all_start_text = _read(Path(all_features["generated_commands"]["start_script"]))
        assert "--compilation-config" in all_start_text
        assert "--quantization ascend" in all_start_text
        assert "--data-parallel-size 2" in all_start_text
        assert "--enable-expert-parallel" in all_start_text
        assert "--enable-lora" in all_start_text
        assert "--enable-sleep-mode" in all_start_text
        assert "--speculative-config" in all_start_text
        assert "weight_prefetch_config" in all_start_text
        assert all_features["deployment_plan"]["evidence_block"], "All-features case should include evidence block"

    print("PASS: render package tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
