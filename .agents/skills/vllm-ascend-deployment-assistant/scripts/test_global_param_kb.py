#!/usr/bin/env python3
"""Comprehensive tests for high-confidence global parameter/env knowledge base."""

from __future__ import annotations

import json
from pathlib import Path

from build_global_param_kb import main as build_main

ALLOWED_STATUS = {"aligned", "upstream_delta", "needs_manual_review"}
HIGH_RISK = {
    "quantization",
    "int4_quantization",
    "graph_mode",
    "tensor_parallel",
    "data_parallel",
    "expert_parallel",
    "context_parallel",
    "prefill_decode_disaggregation",
    "lora",
    "speculative_decode",
    "sleep_mode",
    "weight_prefetch",
    "prefix_cache",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    build_main()

    repo_root = Path(__file__).resolve().parents[4]
    shared_root = repo_root / ".agents" / "skills" / "_shared"
    deploy_root = shared_root / "deployment-config" / "references" / "generated"

    kb_path = deploy_root / "global_parameter_kb.json"
    feature_summary_path = deploy_root / "global_feature_summary.json"
    combo_rules_path = deploy_root / "global_combo_rules.json"
    report_path = deploy_root / "global_validation_report.json"
    upstream_path = deploy_root / "global_upstream_snapshot.json"
    value_semantics_progress_path = deploy_root / "global_value_semantics_progress.json"
    legacy_pairings_path = deploy_root / "global_flag_pairings.json"
    legacy_scan_files_path = deploy_root / "global_scan_files.json"

    vllm_args_path = shared_root / "vllm-upstream" / "references" / "generated" / "vllm_args_inventory.json"
    vllm_env_path = shared_root / "vllm-upstream" / "references" / "generated" / "vllm_env_inventory.json"
    asc_args_path = shared_root / "vllm-ascend-core" / "references" / "generated" / "vllm_ascend_args_inventory.json"
    asc_env_path = shared_root / "vllm-ascend-core" / "references" / "generated" / "vllm_ascend_env_inventory.json"
    asc_args_freq_path = shared_root / "vllm-ascend-core" / "references" / "generated" / "vllm_ascend_args_frequency.json"
    ai_root = shared_root / "ai-foundation"
    ai_topic_index_path = ai_root / "indexes" / "topic-index.json"
    ai_alias_index_path = ai_root / "indexes" / "term-alias-index.json"
    ai_view_index_path = ai_root / "indexes" / "view-index.json"
    ai_rule_index_path = ai_root / "indexes" / "rule-index.json"
    ai_build_report_path = ai_root / "indexes" / "build-report.json"

    for path in [
        kb_path,
        feature_summary_path,
        combo_rules_path,
        report_path,
        upstream_path,
        value_semantics_progress_path,
        legacy_pairings_path,
        legacy_scan_files_path,
        vllm_args_path,
        vllm_env_path,
        asc_args_path,
        asc_env_path,
        asc_args_freq_path,
        ai_topic_index_path,
        ai_alias_index_path,
        ai_view_index_path,
        ai_rule_index_path,
        ai_build_report_path,
    ]:
        assert path.exists(), f"Missing generated artifact: {path}"

    kb = _load(kb_path)
    entries = kb["entries"]
    datasets = kb["datasets"]
    combo_rules = kb["combo_rules"]
    report = kb["validation_report"]
    summary = _load(feature_summary_path)
    upstream = _load(upstream_path)
    value_semantics_progress = _load(value_semantics_progress_path)
    legacy_pairings = _load(legacy_pairings_path)
    legacy_scan_files = _load(legacy_scan_files_path)
    asc_args_freq = _load(asc_args_freq_path)
    ai_topic_index = _load(ai_topic_index_path)
    ai_alias_index = _load(ai_alias_index_path)
    ai_view_index = _load(ai_view_index_path)
    ai_rule_index = _load(ai_rule_index_path)
    ai_build_report = _load(ai_build_report_path)

    assert kb["baseline"]["mode"] == "dual"
    assert len(entries) >= 350, f"Expected broad coverage, got {len(entries)}"

    # Schema + evidence checks
    allowed_sources = {"code", "docs_export", "tests_yaml", "multi_source"}
    for entry in entries:
        assert entry["source"] in allowed_sources, f"Unexpected source detected: {entry['id']}"
        assert isinstance(entry.get("source_tags"), list) and entry["source_tags"], (
            f"Missing source_tags: {entry['id']}"
        )
        if entry["source"] != "code":
            assert entry["scope"] == "vllm_ascend", f"Non-code source should stay in vllm_ascend scope: {entry['id']}"
            assert entry["kind"] == "env", f"Non-code source should only contribute env entries: {entry['id']}"
        assert entry["kind"] in {"arg", "env"}
        assert entry["scope"] in {"vllm", "vllm_ascend"}
        assert entry["status"] in ALLOWED_STATUS
        assert entry["definition_ref"], f"Missing definition_ref: {entry['id']}"
        assert isinstance(entry["read_ref"], list)
        assert isinstance(entry["effect_ref"], list)
        assert isinstance(entry["web_refs"], list)
        assert isinstance(entry["confidence"], float)
        assert 0.0 < entry["confidence"] <= 1.0
        assert isinstance(entry.get("value_semantics"), dict), f"Missing value_semantics: {entry['id']}"
        assert entry.get("value_semantics_completion") in {"todo", "done"}

    # High-risk entries should be mostly backed by local docs + official web evidence
    high_risk_rows = [e for e in entries if e["primary_feature"] in HIGH_RISK]
    assert high_risk_rows, "Expected high-risk entries"
    high_risk_dual_evidence = 0
    for row in high_risk_rows:
        if row["local_doc_refs"] and any(ref.get("tier") == "official" for ref in row["web_refs"]):
            high_risk_dual_evidence += 1
    assert high_risk_dual_evidence / len(high_risk_rows) >= 0.35, (
        f"High-risk dual-evidence ratio too low: {high_risk_dual_evidence}/{len(high_risk_rows)}"
    )

    # Core entries and blocked demo cases
    by_name = {e["name"]: e for e in entries}
    assert by_name["--quantization"]["primary_feature"] == "quantization"
    assert by_name["--compilation-config"]["primary_feature"] == "graph_mode"
    assert by_name["--enable-expert-parallel"]["primary_feature"] == "expert_parallel"
    assert by_name["VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL"]["primary_feature"] == "context_parallel"
    assert by_name["TASK_QUEUE_ENABLE"]["scope"] == "vllm_ascend"
    assert by_name["HCCL_OP_EXPANSION_MODE"]["scope"] == "vllm_ascend"
    assert by_name["TASK_QUEUE_ENABLE"]["source"] in {"tests_yaml", "docs_export", "multi_source", "code"}

    rule_ids = {row["rule_id"] for row in combo_rules}
    assert "hard_block.qwen3_32b_w8a8_int4" in rule_ids
    assert "hard_block.qwen3_32b_w8a8_ep" in rule_ids

    # Coverage and report interfaces
    assert report["coverage"]["ratio"] == 1.0
    assert report["coverage"]["expected_entries"] == report["coverage"]["actual_entries"]
    assert report["evidence_completeness"]["with_definition_ref"] == len(entries)
    assert report["high_risk_validated_count"] >= 20
    assert isinstance(report["unresolved_items"], list)
    assert "value_semantics_progress" in report
    assert report["value_semantics_progress"]["done"] >= 10
    assert report["value_semantics_progress"]["ratio"] > 0
    stats = report["source_tier_stats"]
    assert stats["official_ref_count"] >= len(entries)
    assert stats["external_ref_count"] > 0
    assert stats["entries_with_external_refs"] > 0
    assert stats["entries_with_official_refs"] == len(entries)

    # Legacy compatibility artifacts should still be generated
    assert isinstance(legacy_pairings, list) and legacy_pairings, "global_flag_pairings should be non-empty"
    assert all({"left", "right", "cooccurrence_files"} <= set(row.keys()) for row in legacy_pairings[:10])
    assert isinstance(legacy_scan_files, list) and legacy_scan_files, "global_scan_files should be non-empty"
    assert any(path.startswith("examples/") for path in legacy_scan_files)
    assert isinstance(asc_args_freq, dict) and asc_args_freq, "vllm_ascend_args_frequency should be non-empty"
    assert asc_args_freq.get("--quantization", 0) > 0
    assert value_semantics_progress["done"] == report["value_semantics_progress"]["done"]
    assert ai_build_report["coverage_from_global_kb"]["ratio"] == 1.0
    assert ai_build_report["coverage_from_global_kb"]["actual"] == len(entries)
    assert ai_build_report["model_profile_count"] >= 2
    assert ai_topic_index["total_topics"] >= len(entries)
    assert "graph_mode" in ai_alias_index.get("feature_aliases", {})
    assert any(row.get("query_intent") == "deploy" for row in ai_view_index.get("routes", []))
    assert any(row.get("rule_id") == "hard_block.qwen3_32b_w8a8_ep" for row in ai_rule_index.get("rules", []))

    # Dataset snapshot interface exists for Skill consumption
    for key in ["vllm_args", "vllm_envs", "vllm_ascend_args", "vllm_ascend_envs"]:
        assert key in datasets
        assert datasets[key], f"Empty dataset snapshot: {key}"

    # Upstream snapshot should carry url inventory
    assert "urls" in upstream and "vllm_env" in upstream["urls"]

    # Feature summary should cover key feature families
    for required in [
        "quantization",
        "graph_mode",
        "tensor_parallel",
        "data_parallel",
        "context_parallel",
        "security_auth",
        "memory_tuning",
    ]:
        assert required in summary, f"Missing feature bucket: {required}"

    print("PASS: global parameter knowledge base generation (high-confidence)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
