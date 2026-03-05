#!/usr/bin/env python3
"""Build topic-centered knowledge base with dual indexes for weak-model-safe retrieval."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SECTION_CORE = "Core"
SECTION_FOUNDATION = "Foundation"
SECTION_DEPLOY = "Deployment View"
SECTION_DEV = "Development View"
SECTION_DETAILS = "Details/Edge Cases"

FEATURE_ALIAS_CONFIG: dict[str, dict[str, list[str] | str]] = {
    "quantization": {
        "topic_id": "feature.quantization",
        "zh_aliases": ["量化", "开量化", "int8量化", "w8a8"],
        "en_aliases": ["quantization", "int8", "w8a8"],
        "slang_aliases": ["压模型", "压权重"],
    },
    "int4_quantization": {
        "topic_id": "feature.int4_quantization",
        "zh_aliases": ["int4量化", "w4a4", "4bit量化"],
        "en_aliases": ["int4", "w4a4", "int4 quantization", "4bit"],
        "slang_aliases": ["开int4", "开4bit"],
    },
    "graph_mode": {
        "topic_id": "feature.graph_mode",
        "zh_aliases": ["图模式", "开图", "全图", "图加速"],
        "en_aliases": ["graph mode", "cudagraph", "full decode"],
        "slang_aliases": ["抓图"],
    },
    "tensor_parallel": {
        "topic_id": "feature.tensor_parallel",
        "zh_aliases": ["张量并行", "tp并行", "切tp"],
        "en_aliases": ["tensor parallel", "tp", "tp="],
        "slang_aliases": ["横切并行"],
    },
    "data_parallel": {
        "topic_id": "feature.data_parallel",
        "zh_aliases": ["数据并行", "dp并行", "切dp"],
        "en_aliases": ["data parallel", "dp", "dp="],
        "slang_aliases": ["副本并行"],
    },
    "expert_parallel": {
        "topic_id": "feature.expert_parallel",
        "zh_aliases": ["专家并行", "ep并行"],
        "en_aliases": ["expert parallel", "ep", "ep="],
        "slang_aliases": ["moe并行", "moe"],
    },
    "prefill_decode_disaggregation": {
        "topic_id": "feature.prefill_decode_disaggregation",
        "zh_aliases": ["预填充解码分离", "pd分离", "prefill-decode分离"],
        "en_aliases": ["prefill decode disaggregation", "pd disaggregation"],
        "slang_aliases": ["p节点d节点", "pd部署"],
    },
    "prefix_cache": {
        "topic_id": "feature.prefix_cache",
        "zh_aliases": ["前缀缓存", "开缓存"],
        "en_aliases": ["prefix cache", "automatic prefix caching"],
        "slang_aliases": ["复用前缀"],
    },
    "context_parallel": {
        "topic_id": "feature.context_parallel",
        "zh_aliases": ["上下文并行", "长上下文并行", "cp并行"],
        "en_aliases": ["context parallel", "cp", "cp="],
        "slang_aliases": ["长序列并行"],
    },
    "lora": {
        "topic_id": "feature.lora",
        "zh_aliases": ["lora", "lora适配", "挂lora"],
        "en_aliases": ["lora", "lora adapter"],
        "slang_aliases": ["外挂lora"],
    },
    "speculative_decode": {
        "topic_id": "feature.speculative_decode",
        "zh_aliases": ["投机解码", "草稿解码", "spec decode"],
        "en_aliases": ["speculative decoding", "mtp"],
        "slang_aliases": ["猜词加速", "投机"],
    },
    "sleep_mode": {
        "topic_id": "feature.sleep_mode",
        "zh_aliases": ["休眠模式", "空闲休眠"],
        "en_aliases": ["sleep mode", "sleep"],
        "slang_aliases": ["省电模式"],
    },
    "weight_prefetch": {
        "topic_id": "feature.weight_prefetch",
        "zh_aliases": ["权重预取", "预取权重", "预取"],
        "en_aliases": ["weight prefetch"],
        "slang_aliases": ["提前拉权重"],
    },
}

FEATURE_FOUNDATION_FACTS: dict[str, dict[str, Any]] = {
    "quantization": {
        "core": "量化通过低比特权重/激活表示降低显存和带宽开销，常以精度换吞吐。",
        "deployment": "先确认模型工件支持，再配置 quantization + dtype + 并行组合。",
        "development": "核验量化后端分支、算子覆盖率、降级路径和精度监控。",
    },
    "int4_quantization": {
        "core": "INT4/W4A4 需要模型工件、内核和平台三方同时支持。",
        "deployment": "未验证工件必须 hard block，避免线上误启动。",
        "development": "增加 profile 规则，明确哪些模型可用 INT4。",
    },
    "graph_mode": {
        "core": "图模式通过稳定执行图降低调度抖动，提升吞吐稳定性。",
        "deployment": "先小流量验证，再放大并发。",
        "development": "关注图捕获边界、动态 shape 分支、fallback 到 eager 的触发条件。",
    },
    "expert_parallel": {
        "core": "EP 面向 MoE 专家路由，Dense 模型没有专家层时不成立。",
        "deployment": "先判定模型是否 MoE，再决定是否开启 EP。",
        "development": "在模型画像中固化 has_moe_layers 与专家数量。",
    },
}

MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "qwen3-32b-w8a8": {
        "topic_id": "model.qwen3-32b-w8a8",
        "canonical_term": "qwen3-32b-w8a8",
        "architecture": {
            "family": "qwen3_dense",
            "is_moe": False,
            "has_moe_layers": False,
            "num_experts": 0,
        },
        "quantization_profile": {
            "fixed_weight_format": "w8a8",
            "supported_variants": ["w8a8"],
        },
        "feature_min_npu_count": {
            "data_parallel": 8,
            "context_parallel": 8,
        },
        "evidence_refs": [
            ".agents/skills/_shared/vllm-ascend-core/concepts/model-feature-compatibility-matrix.md",
            "docs/source/tutorials/models/Qwen3-Dense.md",
        ],
    },
    "qwen3-next-80b-a3b-instruct-w8a8": {
        "topic_id": "model.qwen3-next-80b-a3b-instruct-w8a8",
        "canonical_term": "qwen3-next-80b-a3b-instruct-w8a8",
        "architecture": {
            "family": "qwen3_next",
            "is_moe": True,
            "has_moe_layers": True,
            "num_experts": 80,
        },
        "quantization_profile": {
            "fixed_weight_format": "w8a8",
            "supported_variants": ["w8a8"],
        },
        "feature_min_npu_count": {
            "data_parallel": 8,
            "context_parallel": 8,
        },
        "evidence_refs": [
            ".agents/skills/_shared/vllm-ascend-core/concepts/model-feature-compatibility-matrix.md",
            "docs/source/tutorials/models/Qwen3-Next.md",
        ],
    },
}

MODEL_FEATURE_RULES: list[dict[str, Any]] = [
    {
        "rule_id": "hard_block.qwen3_32b_w8a8_int4",
        "profile": "qwen3-32b-w8a8",
        "conditions": ["int4_quantization"],
        "level": "hard_block",
        "reason": "qwen3-32b-w8a8 profile 为固定 W8A8 工件，不支持 INT4/W4A4。",
        "evidence_refs": [
            ".agents/skills/_shared/vllm-ascend-core/concepts/model-feature-compatibility-matrix.md",
            "docs/source/tutorials/models/Qwen3-Dense.md",
        ],
        "fallback_actions": ["保持 W8A8 或切换到已验证 INT4 模型 profile"],
    },
    {
        "rule_id": "hard_block.qwen3_32b_w8a8_ep",
        "profile": "qwen3-32b-w8a8",
        "conditions": ["expert_parallel"],
        "level": "hard_block",
        "reason": "qwen3-32b-w8a8 是 Dense 模型，无 MoE 层，不适用 EP。",
        "evidence_refs": [
            ".agents/skills/_shared/vllm-ascend-core/concepts/model-feature-compatibility-matrix.md",
            "docs/source/tutorials/models/Qwen3-Dense.md",
        ],
        "fallback_actions": ["改用 TP/DP/图模式/weight_prefetch 进行吞吐优化"],
    },
]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _to_topic_filename(topic_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", topic_id) + ".md"


def _flatten(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        token = value.strip()
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _guess_entry_aliases(entry: dict[str, Any]) -> list[str]:
    name = str(entry.get("name", "")).strip()
    if not name:
        return []

    aliases = [name]
    raw = name.strip("-")
    aliases.append(raw)
    aliases.append(raw.replace("-", "_"))
    aliases.append(raw.replace("-", " "))

    if name.startswith("--"):
        aliases.append(name.replace("--", ""))
        aliases.append(name.replace("--", "").replace("-", ""))

    if name.isupper():
        lowered = name.lower()
        aliases.extend([lowered, lowered.replace("_", "-"), lowered.replace("_", " ")])

    feature = str(entry.get("primary_feature", "")).strip()
    if feature:
        aliases.extend([feature, feature.replace("_", " "), feature.replace("_", "-")])

    return _flatten(aliases)


def _section_list(title: str, rows: list[str]) -> str:
    if not rows:
        return f"## {title}\n\n- N/A\n"
    body = "\n".join(f"- {row}" for row in rows)
    return f"## {title}\n\n{body}\n"


def _write_param_topic(topic_path: Path, entry: dict[str, Any]) -> None:
    topic_id = str(entry["id"])
    canonical = str(entry["name"])
    aliases = _guess_entry_aliases(entry)
    evidence = entry.get("definition_ref", [])
    read_refs = entry.get("read_ref", [])
    effect_refs = entry.get("effect_ref", [])
    value_semantics = entry.get("value_semantics", {}) if isinstance(entry.get("value_semantics"), dict) else {}

    core_rows = [
        f"topic_id: `{topic_id}`",
        f"canonical_term: `{canonical}`",
        f"kind/scope: `{entry.get('kind')}` / `{entry.get('scope')}`",
        f"stage: `{entry.get('stage')}`",
        f"primary_feature: `{entry.get('primary_feature')}`",
        f"status/confidence: `{entry.get('status')}` / `{entry.get('confidence')}`",
        f"semantics: {entry.get('semantics') or 'N/A'}",
        f"aliases: {', '.join(f'`{alias}`' for alias in aliases[:16])}",
    ]

    foundation_feature = str(entry.get("primary_feature", ""))
    foundation_cfg = FEATURE_FOUNDATION_FACTS.get(foundation_feature, {})
    foundation_rows = [
        foundation_cfg.get("core", "该条目属于部署/推理配置知识，基础语义以代码证据为主。"),
        f"推荐结合 feature: `{foundation_feature or 'general_runtime'}` 查看稳定原理。",
    ]

    accepted_values = value_semantics.get("accepted_values", [])
    if isinstance(accepted_values, list):
        accepted_preview = ", ".join(str(item) for item in accepted_values[:12])
    else:
        accepted_preview = str(accepted_values)

    deploy_rows = [
        f"default_behavior: {value_semantics.get('default_behavior', entry.get('default', 'N/A'))}",
        f"value_shape: `{value_semantics.get('value_shape', 'unknown')}`",
        f"accepted_values: {accepted_preview or 'N/A'}",
        f"constraints: {'; '.join(value_semantics.get('constraints', [])[:4]) or 'N/A'}",
        f"combo_effects: {'; '.join(value_semantics.get('combo_effects', [])[:4]) or 'N/A'}",
    ]

    dev_rows = [
        f"definition_ref: {', '.join(evidence[:3]) or 'N/A'}",
        f"read_ref: {', '.join(read_refs[:3]) or 'N/A'}",
        f"effect_ref: {', '.join(effect_refs[:3]) or 'N/A'}",
        f"web_refs: {len(entry.get('web_refs', []))}",
    ]

    detail_rows = [
        f"failure_modes: {'; '.join(entry.get('failure_modes', [])[:6]) or 'N/A'}",
        f"value_failure_signals: {'; '.join(value_semantics.get('failure_signals', [])[:6]) or 'N/A'}",
        f"recommendation: {entry.get('recommendation') or 'N/A'}",
        f"updated_at: {entry.get('updated_at', 'N/A')}",
    ]

    content = [
        "---",
        f"topic_id: {topic_id}",
        f"canonical_term: {canonical}",
        f"topic_kind: parameter",
        "---",
        "",
        f"# {canonical}",
        "",
        _section_list(SECTION_CORE, core_rows).strip(),
        "",
        _section_list(SECTION_FOUNDATION, foundation_rows).strip(),
        "",
        _section_list(SECTION_DEPLOY, deploy_rows).strip(),
        "",
        _section_list(SECTION_DEV, dev_rows).strip(),
        "",
        _section_list(SECTION_DETAILS, detail_rows).strip(),
        "",
    ]
    topic_path.write_text("\n".join(content), encoding="utf-8")


def _write_feature_topic(topic_path: Path, feature: str, cfg: dict[str, list[str] | str]) -> None:
    topic_id = str(cfg["topic_id"])
    aliases = _flatten(
        [feature, feature.replace("_", " "), feature.replace("_", "-")]
        + list(cfg.get("zh_aliases", []))
        + list(cfg.get("en_aliases", []))
        + list(cfg.get("slang_aliases", []))
    )
    foundation = FEATURE_FOUNDATION_FACTS.get(feature, {})

    core_rows = [
        f"topic_id: `{topic_id}`",
        f"canonical_term: `{feature}`",
        f"aliases: {', '.join(f'`{alias}`' for alias in aliases[:24])}",
    ]
    foundation_rows = [
        foundation.get("core", "该特性属于部署/推理配置能力。"),
    ]
    deploy_rows = [
        foundation.get("deployment", "部署时应先检查模型/硬件前置条件，再开启。"),
    ]
    dev_rows = [
        foundation.get("development", "开发时应核验定义-读取-生效的完整证据链。"),
    ]
    detail_rows = [
        "与参数 topic 通过 `primary_feature` 关联，所有值语义在参数 topic 中展开。",
    ]

    content = [
        "---",
        f"topic_id: {topic_id}",
        f"canonical_term: {feature}",
        "topic_kind: feature",
        "---",
        "",
        f"# Feature: {feature}",
        "",
        _section_list(SECTION_CORE, core_rows).strip(),
        "",
        _section_list(SECTION_FOUNDATION, foundation_rows).strip(),
        "",
        _section_list(SECTION_DEPLOY, deploy_rows).strip(),
        "",
        _section_list(SECTION_DEV, dev_rows).strip(),
        "",
        _section_list(SECTION_DETAILS, detail_rows).strip(),
        "",
    ]
    topic_path.write_text("\n".join(content), encoding="utf-8")


def _write_model_topic(topic_path: Path, model_id: str, profile: dict[str, Any]) -> None:
    architecture = profile.get("architecture", {})
    quant_profile = profile.get("quantization_profile", {})

    core_rows = [
        f"topic_id: `{profile.get('topic_id')}`",
        f"canonical_term: `{model_id}`",
        f"has_moe_layers: `{architecture.get('has_moe_layers')}`",
        f"num_experts: `{architecture.get('num_experts')}`",
        f"fixed_weight_format: `{quant_profile.get('fixed_weight_format')}`",
    ]
    foundation_rows = [
        "模型画像用于配置可行性推导，不参与业务逻辑改写。",
        "Dense 模型不适用 EP；MoE 模型可进一步评估 EP。",
    ]
    deploy_rows = [
        "部署前先做 profile 校验：量化工件支持、并行能力边界、最小卡数。",
        "不满足条件时返回 hard_block/warning，并附 fallback。",
    ]
    dev_rows = [
        f"evidence_refs: {', '.join(profile.get('evidence_refs', [])) or 'N/A'}",
        f"feature_min_npu_count: {profile.get('feature_min_npu_count', {})}",
    ]
    detail_rows = [
        f"supported_variants: {quant_profile.get('supported_variants', [])}",
        f"architecture_family: {architecture.get('family', 'unknown')}",
    ]

    content = [
        "---",
        f"topic_id: {profile.get('topic_id')}",
        f"canonical_term: {model_id}",
        "topic_kind: model_profile",
        "---",
        "",
        f"# Model Profile: {model_id}",
        "",
        _section_list(SECTION_CORE, core_rows).strip(),
        "",
        _section_list(SECTION_FOUNDATION, foundation_rows).strip(),
        "",
        _section_list(SECTION_DEPLOY, deploy_rows).strip(),
        "",
        _section_list(SECTION_DEV, dev_rows).strip(),
        "",
        _section_list(SECTION_DETAILS, detail_rows).strip(),
        "",
    ]
    topic_path.write_text("\n".join(content), encoding="utf-8")


def _build_indexes(
    topics: list[dict[str, Any]],
    feature_topics: list[dict[str, Any]],
    model_topics: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    topic_rows = topics + feature_topics + model_topics

    topic_index = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_topics": len(topic_rows),
        "topics": topic_rows,
    }

    alias_rows: list[dict[str, Any]] = []
    feature_aliases: dict[str, list[str]] = {}

    for row in topic_rows:
        topic_id = row["topic_id"]
        canonical = row["canonical_term"]
        aliases = row.get("aliases", [])
        for alias in aliases:
            alias_rows.append(
                {
                    "alias": alias,
                    "canonical_term": canonical,
                    "topic_id": topic_id,
                }
            )

        if row.get("topic_kind") == "feature":
            feature_aliases[canonical] = aliases

    term_alias_index = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_aliases": feature_aliases,
        "aliases": alias_rows,
    }

    view_routes: list[dict[str, str]] = []
    for row in topic_rows:
        topic_id = row["topic_id"]
        view_routes.extend(
            [
                {"query_intent": "deploy", "topic_id": topic_id, "target_section": SECTION_DEPLOY},
                {"query_intent": "develop", "topic_id": topic_id, "target_section": SECTION_DEV},
                {"query_intent": "troubleshoot", "topic_id": topic_id, "target_section": SECTION_DETAILS},
            ]
        )

    view_index = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "routes": view_routes,
    }

    evidence_index = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence": [
            {
                "topic_id": row["topic_id"],
                "definition_ref": row.get("definition_ref", []),
                "read_ref": row.get("read_ref", []),
                "effect_ref": row.get("effect_ref", []),
                "web_refs": row.get("web_refs", []),
            }
            for row in topics
        ],
    }

    return topic_index, term_alias_index, view_index, evidence_index


def build_topic_centered_kb(ascend_root: Path) -> dict[str, Any]:
    shared_root = ascend_root / ".agents" / "skills" / "_shared"
    ai_root = shared_root / "ai-foundation"
    topics_root = ai_root / "topics"
    indexes_root = ai_root / "indexes"
    models_root = ai_root / "model-profiles"
    rules_root = ai_root / "rules"

    for path in [topics_root, indexes_root, models_root, rules_root]:
        path.mkdir(parents=True, exist_ok=True)

    kb_path = shared_root / "deployment-config" / "references" / "generated" / "global_parameter_kb.json"
    if not kb_path.exists():
        raise FileNotFoundError(f"Missing input KB: {kb_path}")

    kb = json.loads(kb_path.read_text(encoding="utf-8"))
    entries = kb.get("entries", [])
    combo_rules = kb.get("combo_rules", [])

    topic_rows: list[dict[str, Any]] = []
    for entry in entries:
        topic_id = str(entry["id"])
        topic_filename = _to_topic_filename(topic_id)
        topic_relpath = f"topics/{topic_filename}"
        _write_param_topic(topics_root / topic_filename, entry)

        topic_rows.append(
            {
                "topic_id": topic_id,
                "canonical_term": entry.get("name"),
                "topic_kind": "parameter",
                "scope": entry.get("scope"),
                "kind": entry.get("kind"),
                "primary_feature": entry.get("primary_feature"),
                "topic_path": topic_relpath,
                "aliases": _guess_entry_aliases(entry),
                "definition_ref": entry.get("definition_ref", []),
                "read_ref": entry.get("read_ref", []),
                "effect_ref": entry.get("effect_ref", []),
                "web_refs": entry.get("web_refs", []),
            }
        )

    feature_rows: list[dict[str, Any]] = []
    for feature, cfg in FEATURE_ALIAS_CONFIG.items():
        topic_id = str(cfg["topic_id"])
        aliases = _flatten(
            [feature, feature.replace("_", " "), feature.replace("_", "-")]
            + list(cfg.get("zh_aliases", []))
            + list(cfg.get("en_aliases", []))
            + list(cfg.get("slang_aliases", []))
        )
        topic_filename = _to_topic_filename(topic_id)
        topic_relpath = f"topics/{topic_filename}"
        _write_feature_topic(topics_root / topic_filename, feature, cfg)
        feature_rows.append(
            {
                "topic_id": topic_id,
                "canonical_term": feature,
                "topic_kind": "feature",
                "scope": "cross_domain",
                "kind": "feature",
                "primary_feature": feature,
                "topic_path": topic_relpath,
                "aliases": aliases,
            }
        )

    model_rows: list[dict[str, Any]] = []
    for model_id, profile in MODEL_PROFILES.items():
        topic_id = str(profile["topic_id"])
        topic_filename = _to_topic_filename(topic_id)
        topic_relpath = f"topics/{topic_filename}"
        _write_model_topic(topics_root / topic_filename, model_id, profile)
        _write_json(models_root / f"{model_id}.json", profile)
        model_rows.append(
            {
                "topic_id": topic_id,
                "canonical_term": model_id,
                "topic_kind": "model_profile",
                "scope": "cross_domain",
                "kind": "model_profile",
                "primary_feature": "model_profile",
                "topic_path": topic_relpath,
                "aliases": [
                    model_id,
                    model_id.replace("-", " "),
                    model_id.upper(),
                ],
            }
        )

    topic_index, term_alias_index, view_index, evidence_index = _build_indexes(
        topics=topic_rows,
        feature_topics=feature_rows,
        model_topics=model_rows,
    )

    rule_index = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rules": combo_rules + MODEL_FEATURE_RULES,
    }

    _write_json(indexes_root / "topic-index.json", topic_index)
    _write_json(indexes_root / "term-alias-index.json", term_alias_index)
    _write_json(indexes_root / "view-index.json", view_index)
    _write_json(indexes_root / "evidence-index.json", evidence_index)
    _write_json(indexes_root / "rule-index.json", rule_index)
    _write_json(rules_root / "model-feature-rules.json", MODEL_FEATURE_RULES)

    index_md = ai_root / "INDEX.md"
    index_md.write_text(
        "\n".join(
            [
                "# AI Foundation Knowledge (Topic-Centered)",
                "",
                "Single-source topic files with dual indexes for deployment/development retrieval.",
                "",
                "## Layout",
                "",
                "- `topics/`: one topic per file (`Core`, `Foundation`, `Deployment View`, `Development View`, `Details/Edge Cases`).",
                "- `indexes/topic-index.json`: canonical topic metadata index.",
                "- `indexes/term-alias-index.json`: alias -> canonical term/topic mapping.",
                "- `indexes/view-index.json`: intent -> section routing index.",
                "- `indexes/rule-index.json`: combo and model-compat rules.",
                "- `model-profiles/`: model capability profiles.",
                "",
                "## Guardrails",
                "",
                "- `Core` is single source of truth for facts.",
                "- Deployment/Development sections must not rewrite core facts.",
                "- Evidence and conflict statuses come from generated KB entries.",
                "",
                "Back to [shared index](../INDEX.md).",
                "",
            ]
        ),
        encoding="utf-8",
    )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "topic_count": len(topic_rows),
        "feature_topic_count": len(feature_rows),
        "model_profile_count": len(model_rows),
        "total_topics": len(topic_rows) + len(feature_rows) + len(model_rows),
        "rule_count": len(rule_index["rules"]),
        "coverage_from_global_kb": {
            "expected": len(entries),
            "actual": len(topic_rows),
            "ratio": 1.0 if entries else 0.0,
        },
    }
    _write_json(indexes_root / "build-report.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None, help="Path to vllm-ascend repo root")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    ascend_root = Path(args.repo_root).resolve() if args.repo_root else script_dir.parents[3]
    result = build_topic_centered_kb(ascend_root=ascend_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
