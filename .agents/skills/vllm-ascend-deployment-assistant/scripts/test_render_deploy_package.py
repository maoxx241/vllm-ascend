#!/usr/bin/env python3
"""Regression tests for deployment package rendering."""

from __future__ import annotations

import tempfile
from pathlib import Path

from render_deploy_package import render_package


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> int:
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

    print("PASS: render package tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
