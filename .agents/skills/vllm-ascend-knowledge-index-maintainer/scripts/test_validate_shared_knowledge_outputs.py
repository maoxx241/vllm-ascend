#!/usr/bin/env python3
"""Smoke tests for the validation/import pipeline outputs."""

from __future__ import annotations

import json
from pathlib import Path


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    root = Path(__file__).resolve().parents[5]
    shared_root = (
        root
        / "vllm-ascend"
        / ".agents"
        / "skills"
        / "_shared"
    )
    upstream_source_root = shared_root / "vllm-upstream" / "references" / "source" / "knowledge"
    ascend_source_root = shared_root / "vllm-ascend-core" / "references" / "source" / "knowledge"
    integration_source_root = shared_root / "integration-core" / "references" / "source" / "knowledge"
    governance_root = shared_root / "knowledge-governance"
    contracts_root = governance_root / "contracts"
    provenance_root = governance_root / "provenance"
    shared_generated = governance_root / "generated"

    required_paths = [
        contracts_root / "knowledge_point_schema.json",
        contracts_root / "knowledge_domain_registry.json",
        governance_root / "source" / "meta" / "version.json",
        provenance_root / "verification_manifest.json",
        provenance_root / "verification_batches.json",
        provenance_root / "verification_handoff.md",
        provenance_root / "execution_state.json",
        provenance_root / "import_manifest.json",
        provenance_root / "final_verification_report.json",
        provenance_root / "blocker_entries.json",
        provenance_root / "count_mismatch_diagnosis.json",
        provenance_root / "web_evidence_cache.json",
        shared_generated / "imported_knowledge_manifest.json",
        shared_generated / "imported_knowledge_search_index.json",
        shared_generated / "design_analysis_index.json",
        shared_generated / "task_skill_index.json",
        shared_generated / "skill_scenario_coverage.json",
        shared_generated / "domain_index.json",
        shared_generated / "imported_knowledge_report.json",
    ]
    for path in required_paths:
        assert path.exists(), f"Missing required artifact: {path}"

    manifest = _load(provenance_root / "verification_manifest.json")
    batches = _load(provenance_root / "verification_batches.json")
    execution_state = _load(provenance_root / "execution_state.json")
    import_manifest = _load(provenance_root / "import_manifest.json")
    final_report = _load(provenance_root / "final_verification_report.json")
    blocker_entries = _load(provenance_root / "blocker_entries.json")
    count_diagnosis = _load(provenance_root / "count_mismatch_diagnosis.json")
    shared_manifest = _load(shared_generated / "imported_knowledge_manifest.json")
    design_index = _load(shared_generated / "design_analysis_index.json")
    task_skill_index = _load(shared_generated / "task_skill_index.json")
    scenario_coverage = _load(shared_generated / "skill_scenario_coverage.json")
    domain_index = _load(shared_generated / "domain_index.json")

    entries = manifest["entries"]
    assert len(entries) == 196
    assert all("task_types" in row for row in entries)
    assert all("consumer_skills" in row for row in entries)
    assert all("domain_scope" in row for row in entries)
    assert all("knowledge_domain" in row for row in entries)
    assert all("implementation_repos" in row for row in entries)
    assert all("domain_reason" in row for row in entries)
    assert all("source_hash" in row for row in entries)
    assert any("design_analysis" in row["task_types"] for row in entries)
    assert all(row["status"] == "validated" for row in entries)
    assert {row["domain_scope"] for row in entries} <= {"vllm", "vllm-ascend", "both"}
    assert {row["knowledge_domain"] for row in entries} == {
        "vllm-upstream",
        "vllm-ascend-core",
        "integration-core",
    }
    assert sum(1 for _ in upstream_source_root.rglob("*.json")) == 23
    assert sum(1 for _ in ascend_source_root.rglob("*.json")) == 137
    assert sum(1 for _ in integration_source_root.rglob("*.json")) == 36

    assert batches["batches"], "Expected at least one batch"
    assert batches["current_batch_id"] is None
    assert batches["batches"][-1]["name"] == "relations_matrices_consistency"
    assert any(batch["design_analysis_entries"] for batch in batches["batches"] if batch["entry_count"] > 0)
    assert execution_state["phase"] == "completed"
    assert execution_state["domain_adjudication_completed"] is True
    assert execution_state["current_entry_cursor"] == 196
    assert execution_state["current_domain"] is None
    assert execution_state["pending_entry_ids"] == []
    assert set(execution_state["source_roots"]) == {
        "vllm-upstream",
        "vllm-ascend-core",
        "integration-core",
    }

    assert import_manifest["coverage"]["ratio"] == 1.0
    assert import_manifest["eligible_entry_count"] == shared_manifest["eligible_entry_count"]
    assert final_report["task_coverage"]["design_analysis"] > 0
    assert final_report["summary"]["validated"] == 196
    assert final_report["summary"]["validated_with_gap"] == 0
    assert final_report["summary"]["rewrite_required"] == len(blocker_entries) == 0
    assert count_diagnosis["issues"] == []
    assert final_report["count_validation"]["status"] == "pass"
    assert final_report["web_evidence_cache"]["url_count"] > 0
    assert final_report["web_evidence_cache"]["fetched_ok"] > 0
    assert final_report["scenario_coverage"]["status"] == "pass"

    assert design_index["entry_count"] > 0
    assert any(row["design_skills"] for row in design_index["entries"])
    assert "design_analysis" in task_skill_index["task_types"]
    assert task_skill_index["task_types"]["design_analysis"]["skills"]
    assert "docs-compliance-checker" in task_skill_index["task_types"]["release_analysis"]["skills"]
    assert "knowledge-index-maintainer" in task_skill_index["task_types"]["upstream_sync"]["skills"]
    assert "ci-gatekeeper" in task_skill_index["task_types"]["op_development"]["skills"]
    assert set(domain_index["domain_scope_index"]) == {"vllm", "vllm-ascend", "both"}
    assert set(domain_index["knowledge_domain_index"]) == {
        "vllm-upstream",
        "vllm-ascend-core",
        "integration-core",
    }
    assert all(source_ids for source_ids in domain_index["implementation_repo_index"].values())

    assert scenario_coverage["status"] == "pass"
    assert scenario_coverage["scenario_count"] >= 15
    assert all(payload["scenario_count"] >= 1 for payload in scenario_coverage["task_type_coverage"].values())
    assert all(payload["meets_minimum_two"] for payload in scenario_coverage["composer_coverage"].values())
    assert all(payload["covered"] for payload in scenario_coverage["atomic_skill_coverage"].values())
    assert all(row["status"] == "covered" for row in scenario_coverage["scenarios"])

    handoff = (provenance_root / "verification_handoff.md").read_text(encoding="utf-8")
    assert "Top Design-Analysis Gaps" in handoff
    assert "Normalization Notes" in handoff

    print("PASS: shared knowledge validation outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
