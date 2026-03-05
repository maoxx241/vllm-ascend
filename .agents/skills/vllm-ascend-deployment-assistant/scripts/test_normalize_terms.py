#!/usr/bin/env python3
"""Regression tests for natural-language term normalization."""

from __future__ import annotations

from normalize_terms import normalize_input


def main() -> int:
    cases = [
        ("帮我开w8a8部署qwen3", {"quantization"}),
        ("先开图模式再启动服务", {"graph_mode"}),
        ("我要tp4部署", {"tensor_parallel"}),
        ("这个模型加dp并行", {"data_parallel"}),
        ("moe模型开ep", {"expert_parallel"}),
        ("我要做pd分离", {"prefill_decode_disaggregation"}),
        ("保留前缀缓存", {"prefix_cache"}),
        ("长上下文要cp并行", {"context_parallel"}),
        ("把lora挂上", {"lora"}),
        ("开投机解码", {"speculative_decode"}),
        ("服务空闲时休眠", {"sleep_mode"}),
        ("开启权重预取提吞吐", {"weight_prefetch"}),
        ("部署时开图加w8a8", {"graph_mode", "quantization"}),
        ("tp和dp都开", {"tensor_parallel", "data_parallel"}),
        ("给MoE开专家并行和投机", {"expert_parallel", "speculative_decode"}),
        ("我要context parallel", {"context_parallel"}),
        ("enable graph mode", {"graph_mode"}),
        ("use quantization and lora", {"quantization", "lora"}),
        ("prefill decode disaggregation for this service", {"prefill_decode_disaggregation"}),
        ("请部署并开启sleep mode", {"sleep_mode"}),
        ("开图并开启prefix cache", {"graph_mode", "prefix_cache"}),
        ("我要在部署里开weight prefetch", {"weight_prefetch"}),
    ]

    passed = 0
    failed_rows = []

    for idx, (text, expected) in enumerate(cases, 1):
        result = normalize_input(text)
        actual = set(result["features"])
        ok = expected.issubset(actual)
        if ok:
            passed += 1
        else:
            failed_rows.append((idx, text, sorted(expected), sorted(actual)))

    accuracy = passed / len(cases)
    print(f"Normalization accuracy: {passed}/{len(cases)} = {accuracy:.2%}")

    ambiguous = normalize_input("开并行提吞吐")
    assert "parallel_strategy" in ambiguous["missing_slots"], (
        "Expected ambiguous parallel phrase to require parallel_strategy clarification."
    )
    assert ambiguous["clarification_question"], "Expected clarification question for ambiguous phrase."

    unknown = normalize_input("给我来个黑科技")
    assert "feature" in unknown["missing_slots"], "Unknown phrase should request feature clarification."
    assert unknown["clarification_question"], "Unknown phrase should include one clarification question."

    if accuracy < 0.90:
        print("Failed rows:")
        for row in failed_rows:
            print(row)
        raise SystemExit(1)

    print("PASS: normalization tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
