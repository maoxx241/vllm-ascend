#!/usr/bin/env python3
"""Coverage-oriented test matrix for feature normalization and compatibility."""

from __future__ import annotations

import tempfile
from pathlib import Path

from build_global_param_kb import main as build_main
from normalize_terms import normalize_input
from render_deploy_package import SUPPORTED_FEATURES, render_package


FEATURE_PHRASES = {
    "quantization": "开量化w8a8",
    "int4_quantization": "开int4或者w4a4",
    "graph_mode": "开图模式",
    "tensor_parallel": "开tp并行",
    "data_parallel": "开dp并行",
    "expert_parallel": "开ep并行",
    "prefill_decode_disaggregation": "做pd分离部署",
    "prefix_cache": "开启前缀缓存",
    "context_parallel": "开启context parallel",
    "lora": "挂lora",
    "speculative_decode": "开投机解码",
    "sleep_mode": "开sleep mode",
    "weight_prefetch": "开权重预取",
}


def main() -> int:
    build_main()

    missing = []
    for feature, phrase in FEATURE_PHRASES.items():
        parsed = normalize_input(phrase)
        if feature not in parsed["features"]:
            missing.append((feature, phrase, parsed["features"]))

    if missing:
        print("Feature normalization misses:")
        for row in missing:
            print(row)
        raise SystemExit(1)

    with tempfile.TemporaryDirectory(prefix="deploy_pkg_cov_") as td:
        out = Path(td)
        for profile in ("qwen3-32b-w8a8", "qwen3-next-80b-a3b-instruct-w8a8"):
            for feature in sorted(SUPPORTED_FEATURES):
                case = render_package(
                    output_dir=out / profile / feature,
                    model_profile=profile,
                    model_path_override=None,
                    hardware_type="Atlas A2/A3",
                    npu_count=8,
                    port_override=28000,
                    text=None,
                    features_input=[feature],
                )
                plan = case["deployment_plan"]
                assert "compatibility" in plan
                assert "model_knowledge" in plan
                blocked = {item["feature"] for item in plan["compatibility"]["blocked_features"]}
                allowed = set(plan["compatibility"]["allowed_features"])
                advisory = {item["feature"] for item in plan["compatibility"]["advisory_features"]}
                assert plan["compatibility"]["downgraded_features"] == []
                assert feature in blocked or feature in allowed, (
                    f"Feature {feature} must be either allowed or blocked for profile {profile}."
                )
                if feature in advisory:
                    assert feature in allowed
                assert isinstance(plan["evidence_block"], list) and plan["evidence_block"], (
                    f"Feature {feature} should include evidence block for profile {profile}."
                )

        qwen3_dense_block = render_package(
            output_dir=out / "dense_block",
            model_profile="qwen3-32b-w8a8",
            model_path_override=None,
            hardware_type="Atlas A2",
            npu_count=4,
            port_override=28111,
            text="给qwen3-32b-w8a8开int4和ep",
            features_input=[],
        )
        blocked = {item["feature"] for item in qwen3_dense_block["deployment_plan"]["compatibility"]["blocked_features"]}
        assert "int4_quantization" in blocked
        assert "expert_parallel" in blocked

    print("PASS: feature coverage matrix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
