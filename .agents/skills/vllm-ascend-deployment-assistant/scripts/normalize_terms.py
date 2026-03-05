#!/usr/bin/env python3
"""Normalize natural-language deployment requests into canonical features."""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple

FEATURES: List[Dict[str, List[str] | str]] = [
    {
        "canonical_feature": "quantization",
        "zh_aliases": ["量化", "开量化", "int8量化", "w8a8"],
        "en_aliases": ["quantization", "int8", "w8a8"],
        "slang_aliases": ["压模型", "压权重"],
    },
    {
        "canonical_feature": "graph_mode",
        "zh_aliases": ["图模式", "开图", "全图", "图加速"],
        "en_aliases": ["graph mode", "cudagraph", "full decode"],
        "slang_aliases": ["抓图"],
    },
    {
        "canonical_feature": "tensor_parallel",
        "zh_aliases": ["张量并行", "tp并行", "切tp"],
        "en_aliases": ["tensor parallel", "tp", "tp="],
        "slang_aliases": ["横切并行"],
    },
    {
        "canonical_feature": "data_parallel",
        "zh_aliases": ["数据并行", "dp并行", "切dp"],
        "en_aliases": ["data parallel", "dp", "dp="],
        "slang_aliases": ["副本并行"],
    },
    {
        "canonical_feature": "expert_parallel",
        "zh_aliases": ["专家并行", "ep并行"],
        "en_aliases": ["expert parallel", "ep", "ep="],
        "slang_aliases": ["moe并行", "moe"],
    },
    {
        "canonical_feature": "prefill_decode_disaggregation",
        "zh_aliases": ["预填充解码分离", "pd分离", "prefill-decode分离"],
        "en_aliases": ["prefill decode disaggregation", "pd disaggregation"],
        "slang_aliases": ["p节点d节点", "pd部署"],
    },
    {
        "canonical_feature": "prefix_cache",
        "zh_aliases": ["前缀缓存", "开缓存"],
        "en_aliases": ["prefix cache", "automatic prefix caching"],
        "slang_aliases": ["复用前缀"],
    },
    {
        "canonical_feature": "context_parallel",
        "zh_aliases": ["上下文并行", "长上下文并行", "cp并行"],
        "en_aliases": ["context parallel", "cp", "cp="],
        "slang_aliases": ["长序列并行"],
    },
    {
        "canonical_feature": "lora",
        "zh_aliases": ["lora", "lora适配", "挂lora"],
        "en_aliases": ["lora", "lora adapter"],
        "slang_aliases": ["外挂lora"],
    },
    {
        "canonical_feature": "speculative_decode",
        "zh_aliases": ["投机解码", "草稿解码", "spec decode"],
        "en_aliases": ["speculative decoding", "mtp"],
        "slang_aliases": ["猜词加速", "投机"],
    },
    {
        "canonical_feature": "sleep_mode",
        "zh_aliases": ["休眠模式", "空闲休眠"],
        "en_aliases": ["sleep mode"],
        "slang_aliases": ["省电模式"],
    },
    {
        "canonical_feature": "weight_prefetch",
        "zh_aliases": ["权重预取", "预取权重"],
        "en_aliases": ["weight prefetch"],
        "slang_aliases": ["提前拉权重"],
    },
]

PARALLEL_FEATURES = {
    "tensor_parallel",
    "data_parallel",
    "expert_parallel",
    "context_parallel",
}


def _contains(text: str, text_lower: str, alias: str) -> bool:
    alias_norm = alias.strip()
    if not alias_norm:
        return False
    alias_lower = alias_norm.lower()
    if any("a" <= ch <= "z" for ch in alias_lower):
        return alias_lower in text_lower
    return alias_norm in text


def _detect_intent(text: str, text_lower: str, default_intent: str) -> str:
    env_kw = ["环境", "安装", "bootstrap", "setup env", "初始化"]
    deploy_kw = ["部署", "上线", "serve", "启动服务", "launch"]
    if any(_contains(text, text_lower, kw) for kw in env_kw):
        return "env_bootstrap"
    if any(_contains(text, text_lower, kw) for kw in deploy_kw):
        return "deploy_model"
    return default_intent


def _detect_features(text: str, text_lower: str) -> Tuple[List[str], Dict[str, str]]:
    detected: List[str] = []
    matched_alias: Dict[str, str] = {}
    for entry in FEATURES:
        canonical = str(entry["canonical_feature"])
        aliases = (
            list(entry["zh_aliases"])
            + list(entry["en_aliases"])
            + list(entry["slang_aliases"])
        )
        for alias in aliases:
            if _contains(text, text_lower, alias):
                detected.append(canonical)
                matched_alias[canonical] = alias
                break
    return detected, matched_alias


def _suggest_candidates(text: str, text_lower: str) -> List[str]:
    if _contains(text, text_lower, "吞吐") or _contains(text, text_lower, "性能"):
        return ["graph_mode", "tensor_parallel", "weight_prefetch"]
    if _contains(text, text_lower, "并行") or _contains(text, text_lower, "parallel"):
        return ["tensor_parallel", "data_parallel", "expert_parallel"]
    return ["quantization", "graph_mode", "tensor_parallel"]


def normalize_input(text: str, default_intent: str = "deploy_model") -> Dict[str, object]:
    raw = text.strip()
    lowered = raw.lower()

    intent = _detect_intent(raw, lowered, default_intent)
    features, _ = _detect_features(raw, lowered)

    missing_slots: List[str] = []
    clarification_question = ""

    parallel_mentioned = _contains(raw, lowered, "并行") or _contains(raw, lowered, "parallel")
    has_parallel_feature = any(feature in PARALLEL_FEATURES for feature in features)

    if parallel_mentioned and not has_parallel_feature:
        missing_slots.append("parallel_strategy")
        candidates = _suggest_candidates(raw, lowered)
        clarification_question = (
            "你说的并行更偏向 "
            f"{candidates[0]}、{candidates[1]} 还是 {candidates[2]}？"
        )

    if not features:
        if "feature" not in missing_slots:
            missing_slots.append("feature")
        candidates = _suggest_candidates(raw, lowered)
        clarification_question = (
            "我理解你可能在说 "
            f"{candidates[0]}、{candidates[1]}、{candidates[2]}，请确认优先开启哪一个？"
        )

    confidence = 0.25
    if features:
        confidence = min(0.98, 0.62 + 0.12 * len(features))
        if missing_slots:
            confidence = max(0.45, confidence - 0.25)

    result = {
        "intent": intent,
        "features": features,
        "confidence": round(confidence, 2),
        "missing_slots": missing_slots,
        "clarification_question": clarification_question,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="Natural language user request")
    parser.add_argument(
        "--default-intent",
        default="deploy_model",
        help="Fallback intent if no strong intent keyword is found",
    )
    args = parser.parse_args()

    result = normalize_input(args.text, default_intent=args.default_intent)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
