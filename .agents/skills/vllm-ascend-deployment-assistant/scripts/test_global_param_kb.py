#!/usr/bin/env python3
"""Comprehensive tests for global parameter/env knowledge base generation."""

from __future__ import annotations

import json
import re
from pathlib import Path

from build_global_param_kb import main as build_main


LOWER_FLAG_PATTERN = re.compile(r"^--[a-z0-9][a-z0-9\-]*$")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _contains_pair(pairings: list[dict], left: str, right: str) -> bool:
    for row in pairings:
        l = row.get("left")
        r = row.get("right")
        if {l, r} == {left, right}:
            return True
    return False


def main() -> int:
    build_main()

    repo_root = Path(__file__).resolve().parents[4]
    shared_root = repo_root / ".agents" / "skills" / "_shared"

    vllm_env_json = shared_root / "vllm-foundation" / "references" / "generated" / "vllm_env_inventory.json"
    vllm_args_json = shared_root / "vllm-foundation" / "references" / "generated" / "vllm_args_inventory.json"
    asc_env_json = shared_root / "vllm-ascend-core" / "references" / "generated" / "vllm_ascend_env_inventory.json"
    asc_args_json = shared_root / "vllm-ascend-core" / "references" / "generated" / "vllm_ascend_args_inventory.json"

    deploy_gen_root = shared_root / "deployment-config" / "references" / "generated"
    global_kb_json = deploy_gen_root / "global_parameter_kb.json"
    global_summary_json = deploy_gen_root / "global_feature_summary.json"
    global_pairings_json = deploy_gen_root / "global_flag_pairings.json"

    combo_doc = shared_root / "deployment-config" / "references" / "global-parameter-combination-guide.md"
    feature_map_doc = shared_root / "deployment-config" / "references" / "global-parameter-feature-map.md"

    required_paths = [
        vllm_env_json,
        vllm_args_json,
        asc_env_json,
        asc_args_json,
        global_kb_json,
        global_summary_json,
        global_pairings_json,
        combo_doc,
        feature_map_doc,
    ]
    for path in required_paths:
        assert path.exists(), f"Missing generated artifact: {path}"

    vllm_env = _load_json(vllm_env_json)
    vllm_args = _load_json(vllm_args_json)
    asc_env = _load_json(asc_env_json)
    asc_args = _load_json(asc_args_json)

    # Coverage guardrails.
    assert len(vllm_env) >= 180, f"Unexpectedly low vLLM env count: {len(vllm_env)}"
    assert len(vllm_args) >= 200, f"Unexpectedly low vLLM arg count: {len(vllm_args)}"
    assert len(asc_env) >= 20, f"Unexpectedly low vLLM-Ascend env count: {len(asc_env)}"
    assert len(asc_args) >= 120, f"Unexpectedly low vLLM-Ascend arg count: {len(asc_args)}"

    assert "VLLM_TARGET_DEVICE" in vllm_env, "Expected core vLLM env var"
    assert "VLLM_ASCEND_ENABLE_NZ" in asc_env, "Expected core vLLM-Ascend env var"
    assert "--model" in vllm_args, "Expected common serve argument --model"
    assert "--tensor-parallel-size" in vllm_args, "Expected TP argument in vLLM args"
    assert "--quantization" in asc_args, "Expected observed ascend argument --quantization"

    # Cleanliness guardrail: observed args should be normalized lowercase flags.
    invalid_flags = [flag for flag in asc_args if not LOWER_FLAG_PATTERN.match(flag)]
    assert not invalid_flags, f"Found noisy/non-CLI flags in observed args: {invalid_flags[:10]}"

    global_kb = _load_json(global_kb_json)
    datasets = global_kb["datasets"]
    blocked_cases = global_kb["blocked_cases"]
    feature_summary = _load_json(global_summary_json)
    pairings = _load_json(global_pairings_json)

    # Semantic mapping checks.
    assert datasets["vllm_args"]["--quantization"]["primary_feature"] == "quantization"
    assert datasets["vllm_args"]["--compilation-config"]["primary_feature"] == "graph_mode"
    assert datasets["vllm_args"]["--enable-expert-parallel"]["primary_feature"] == "expert_parallel"
    assert datasets["vllm_args"]["--tensor-parallel-size"]["primary_feature"] == "tensor_parallel"
    assert datasets["vllm_ascend_envs"]["VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL"]["primary_feature"] == "context_parallel"

    # Every entry should carry usage and combination hints for weak-model determinism.
    for scope_name, scope_data in datasets.items():
        for name, row in scope_data.items():
            assert row.get("usage_hint"), f"Missing usage_hint: {scope_name}:{name}"
            assert row.get("feature_tags"), f"Missing feature_tags: {scope_name}:{name}"
            assert row.get("primary_feature"), f"Missing primary_feature: {scope_name}:{name}"
            assert isinstance(row.get("combination_candidates"), list), (
                f"Missing combination_candidates list: {scope_name}:{name}"
            )

    # Pairing/co-occurrence evidence should be present.
    assert len(pairings) >= 50, "Expected rich pairing evidence"
    assert any(
        row.get("left") == "--tensor-parallel-size" or row.get("right") == "--tensor-parallel-size"
        for row in pairings
    ), "Expected TP involvement in pairings"
    assert _contains_pair(pairings, "--max-model-len", "--max-num-batched-tokens") or any(
        row.get("left") == "--quantization" or row.get("right") == "--quantization" for row in pairings
    ), "Expected either throughput or quantization pairing evidence"

    # Feature summary must cover core features.
    for core_feature in [
        "quantization",
        "graph_mode",
        "tensor_parallel",
        "data_parallel",
        "context_parallel",
        "throughput_tuning",
        "memory_tuning",
    ]:
        assert core_feature in feature_summary, f"Missing feature summary for: {core_feature}"

    # Explicit blocked cases for demonstration error paths.
    blocked_lookup = {(row["profile"], row["blocked_feature"]): row for row in blocked_cases}
    assert ("qwen3-32b-w8a8", "int4_quantization") in blocked_lookup
    assert ("qwen3-32b-w8a8", "expert_parallel") in blocked_lookup

    combo_text = combo_doc.read_text(encoding="utf-8")
    assert "qwen3-32b-w8a8 + int4_quantization" in combo_text
    assert "qwen3-32b-w8a8 + expert_parallel" in combo_text
    assert "Co-occurrence evidence" in combo_text

    feature_map_text = feature_map_doc.read_text(encoding="utf-8")
    assert "vLLM Serve Args -> Semantics" in feature_map_text
    assert "vLLM-Ascend Env Vars -> Semantics" in feature_map_text

    print("PASS: global parameter knowledge base generation (comprehensive)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
