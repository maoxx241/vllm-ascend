#!/usr/bin/env python3
"""Regression checks for skill scenario coverage artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_source_id(value: str) -> str:
    def _tokenize(token: str) -> str:
        chars = []
        for idx, ch in enumerate(token):
            if ch.isupper() and idx > 0 and (token[idx - 1].islower() or token[idx - 1].isdigit()):
                chars.append("_")
            chars.append(ch)
        normalized = "".join(chars).replace("-", "_").replace(".", "_").replace("/", "_")
        return "_".join(part for part in normalized.lower().split("_") if part)

    if ":" not in value:
        return _tokenize(value)
    prefix, suffix = value.split(":", 1)
    return f"{_tokenize(prefix)}:{_tokenize(suffix)}"


def main() -> int:
    root = Path(__file__).resolve().parents[5]
    shared_root = (
        root
        / "vllm-ascend"
        / ".agents"
        / "skills"
        / "_shared"
    )
    generated_root = shared_root / "knowledge-governance" / "generated"

    scenario_path = generated_root / "skill_scenario_coverage.json"
    manifest_path = generated_root / "imported_knowledge_manifest.json"
    task_skill_path = generated_root / "task_skill_index.json"

    scenario_payload = _load(scenario_path)
    manifest = _load(manifest_path)
    task_skill = _load(task_skill_path)

    source_ids = {_normalize_source_id(entry["source_id"]) for entry in manifest["entries"]}
    tasks = set(task_skill["task_types"])

    expected_task_types = {
        "deployment",
        "env_bootstrap",
        "debugging",
        "model_adaptation",
        "upstream_sync",
        "release_analysis",
        "op_development",
        "performance_analysis",
        "design_analysis",
    }
    assert set(scenario_payload["task_type_coverage"]) == expected_task_types
    assert tasks == expected_task_types

    expected_composers = {
        "model-adapter",
        "sync-coordinator",
        "debug-assistant",
        "release-assistant",
        "op-developer",
        "perf-assistant",
    }
    assert set(scenario_payload["composer_coverage"]) == expected_composers

    expected_atomic = {
        "env-bootstrap",
        "compatibility-checker",
        "repo-state-auditor",
        "log-analyzer",
        "crash-rooter",
        "perf-hunter",
        "graph-analyzer",
        "parallelism-planner",
        "scheduler-feature-designer",
        "attention-kv-designer",
        "custom-model-integrator",
        "precision-validator",
        "release-commit-analyzer",
        "release-notes-composer",
        "docs-compliance-checker",
        "test-matrix-planner",
        "ci-gatekeeper",
        "knowledge-index-maintainer",
    }
    assert set(scenario_payload["atomic_skill_coverage"]) == expected_atomic

    scenarios = scenario_payload["scenarios"]
    assert len(scenarios) == scenario_payload["scenario_count"]
    for scenario in scenarios:
        assert scenario["status"] == "covered", scenario
        assert scenario["task_type"] in expected_task_types
        for rel_path in scenario["required_docs"]:
            assert (shared_root / rel_path).exists(), rel_path
        for source_id in scenario["evidence_entry_ids"]:
            assert _normalize_source_id(source_id) in source_ids, source_id
        for skill, matched in scenario["matched_atomic_skills"].items():
            assert skill in expected_atomic
            assert matched, f"{scenario['id']} missing matched entries for {skill}"

    print("PASS: skill scenario coverage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
