#!/usr/bin/env python3
"""Regression tests for topic-centered knowledge base artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from build_global_param_kb import main as build_main

REQUIRED_SECTIONS = [
    "## Core",
    "## Foundation",
    "## Deployment View",
    "## Development View",
    "## Details/Edge Cases",
]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    build_main()

    repo_root = Path(__file__).resolve().parents[4]
    ai_root = repo_root / ".agents" / "skills" / "_shared" / "ai-foundation"
    indexes_root = ai_root / "indexes"
    topics_root = ai_root / "topics"
    profiles_root = ai_root / "model-profiles"

    for path in [
        ai_root / "INDEX.md",
        indexes_root / "topic-index.json",
        indexes_root / "term-alias-index.json",
        indexes_root / "view-index.json",
        indexes_root / "rule-index.json",
        indexes_root / "evidence-index.json",
        indexes_root / "build-report.json",
        profiles_root / "qwen3-32b-w8a8.json",
        profiles_root / "qwen3-next-80b-a3b-instruct-w8a8.json",
        topics_root / "vllm.arg.async_scheduling.md",
        topics_root / "feature.graph_mode.md",
        topics_root / "model.qwen3-32b-w8a8.md",
    ]:
        assert path.exists(), f"Missing topic-centered artifact: {path}"

    topic_index = _load(indexes_root / "topic-index.json")
    alias_index = _load(indexes_root / "term-alias-index.json")
    view_index = _load(indexes_root / "view-index.json")
    rule_index = _load(indexes_root / "rule-index.json")
    report = _load(indexes_root / "build-report.json")
    qwen3_profile = _load(profiles_root / "qwen3-32b-w8a8.json")

    topics = topic_index["topics"]
    assert topics, "topic-index topics must be non-empty"
    topic_ids = [row["topic_id"] for row in topics]
    assert len(topic_ids) == len(set(topic_ids)), "topic_id must be unique"

    assert report["coverage_from_global_kb"]["ratio"] == 1.0
    assert report["coverage_from_global_kb"]["actual"] == report["coverage_from_global_kb"]["expected"]
    assert report["model_profile_count"] >= 2
    assert "resource_guidance" in qwen3_profile
    assert "feature_min_npu_count" not in qwen3_profile

    # One-topic-one-file structure with fixed sections.
    sample_topic = topics_root / "vllm.arg.async_scheduling.md"
    sample_content = sample_topic.read_text(encoding="utf-8")
    for section in REQUIRED_SECTIONS:
        assert section in sample_content, f"Missing section {section} in {sample_topic}"

    # Alias index should support oral Chinese and CLI/env terms.
    feature_aliases = alias_index.get("feature_aliases", {})
    assert "graph_mode" in feature_aliases
    assert "开图" in feature_aliases["graph_mode"]

    aliases = alias_index.get("aliases", [])
    alias_terms = {row.get("alias") for row in aliases if isinstance(row, dict)}
    assert "--block-size" in alias_terms
    assert "VLLM_DP_SIZE" in alias_terms

    # Dual index routes for deploy/develop/troubleshoot.
    routes = view_index.get("routes", [])
    route_triplets = {
        (row.get("query_intent"), row.get("topic_id"), row.get("target_section"))
        for row in routes
        if isinstance(row, dict)
    }
    assert ("deploy", "vllm.arg.async_scheduling", "Deployment View") in route_triplets
    assert ("develop", "vllm.arg.async_scheduling", "Development View") in route_triplets
    assert ("troubleshoot", "vllm.arg.async_scheduling", "Details/Edge Cases") in route_triplets

    # Rules must include demo hard blocks.
    rules = rule_index.get("rules", [])
    rule_ids = {row.get("rule_id") for row in rules if isinstance(row, dict)}
    assert "hard_block.qwen3_32b_w8a8_int4" in rule_ids
    assert "hard_block.qwen3_32b_w8a8_ep" in rule_ids

    model_topic = (topics_root / "model.qwen3-32b-w8a8.md").read_text(encoding="utf-8")
    assert "resource_guidance.recommended" in model_topic

    print("PASS: topic-centered knowledge base")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
