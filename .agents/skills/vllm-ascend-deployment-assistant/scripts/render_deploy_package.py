#!/usr/bin/env python3
"""Render deterministic deployment package from canonical features."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any, Dict, List, Tuple

from normalize_terms import normalize_input

STATUS_ALIGNED = "aligned"
KB_MIN_CONFIDENCE = 0.65

PROFILES: Dict[str, Dict[str, object]] = {
    "qwen3-32b-w8a8": {
        "model": "vllm-ascend/Qwen3-32B-W8A8",
        "served_model_name": "qwen3",
        "tensor_parallel_size": 4,
        "max_model_len": 5500,
        "max_num_batched_tokens": 40960,
        "gpu_memory_utilization": 0.9,
        "default_port": 8113,
    },
    "qwen3-next-80b-a3b-instruct-w8a8": {
        "model": "vllm-ascend/Qwen3-Next-80B-A3B-Instruct-W8A8",
        "served_model_name": "qwen3-next",
        "tensor_parallel_size": 4,
        "max_model_len": 32768,
        "max_num_batched_tokens": 8192,
        "gpu_memory_utilization": 0.65,
        "default_port": 8000,
    },
}

SUPPORTED_FEATURES = {
    "quantization",
    "int4_quantization",
    "graph_mode",
    "tensor_parallel",
    "data_parallel",
    "expert_parallel",
    "prefill_decode_disaggregation",
    "prefix_cache",
    "context_parallel",
    "lora",
    "speculative_decode",
    "sleep_mode",
    "weight_prefetch",
}

MODEL_KNOWLEDGE: Dict[str, Dict[str, Any]] = {
    "qwen3-32b-w8a8": {
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
        "feature_constraints": {
            "int4_quantization": {
                "level": "hard_block",
                "reason": "Profile is fixed to W8A8 weights; no validated INT4/W4A4 artifact in this profile.",
            },
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
        "feature_constraints": {
            "int4_quantization": {
                "level": "hard_block",
                "reason": "No validated INT4 deployment path in this demo package.",
            },
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

FEATURE_TO_KB_ITEMS: Dict[str, List[str]] = {
    "quantization": ["--quantization", "--model", "--dtype"],
    "int4_quantization": ["--quantization", "--model"],
    "graph_mode": ["--compilation-config", "--enforce-eager"],
    "tensor_parallel": ["--tensor-parallel-size"],
    "data_parallel": ["--data-parallel-size", "--data-parallel-address", "--data-parallel-rpc-port"],
    "expert_parallel": ["--enable-expert-parallel"],
    "prefill_decode_disaggregation": ["--kv-transfer-config", "--ec-transfer-config"],
    "prefix_cache": ["--enable-prefix-caching", "--prefix-caching-hash-algo"],
    "context_parallel": [
        "--prefill-context-parallel-size",
        "--decode-context-parallel-size",
        "VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL",
    ],
    "lora": ["--enable-lora", "--lora-modules"],
    "speculative_decode": ["--speculative-config"],
    "sleep_mode": ["--enable-sleep-mode", "VLLM_SLEEP_WHEN_IDLE"],
    "weight_prefetch": ["--additional-config", "VLLM_ASCEND_ENABLE_PREFETCH_MLP"],
}


def _dedupe(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _parse_feature_list(raw: str | None) -> List[str]:
    if not raw:
        return []
    values = [part.strip() for part in raw.split(",") if part.strip()]
    return [value for value in values if value in SUPPORTED_FEATURES]


def _write_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _load_global_kb_entries() -> Dict[str, List[Dict[str, Any]]]:
    kb_path = (
        Path(__file__).resolve().parents[2]
        / "_shared"
        / "deployment-config"
        / "references"
        / "generated"
        / "global_parameter_kb.json"
    )
    if not kb_path.exists():
        return {}

    try:
        payload = json.loads(kb_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    entries = payload.get("entries", [])
    index: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        name = entry.get("name")
        if not isinstance(name, str):
            continue
        index.setdefault(name, []).append(entry)
    return index


def _build_evidence_block(
    features: List[str],
    kb_index: Dict[str, List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    evidence_block: List[Dict[str, Any]] = []
    conflict_alerts: List[str] = []
    seen_alerts = set()

    # Build reverse index by feature for fallback when name mapping misses.
    feature_index: Dict[str, List[Dict[str, Any]]] = {}
    for rows in kb_index.values():
        for row in rows:
            primary = row.get("primary_feature")
            if isinstance(primary, str):
                feature_index.setdefault(primary, []).append(row)

    for feature in features:
        candidates = FEATURE_TO_KB_ITEMS.get(feature, [])
        picked: List[Dict[str, Any]] = []

        for name in candidates:
            rows = kb_index.get(name, [])
            if not rows:
                continue
            rows = sorted(rows, key=lambda item: float(item.get("confidence", 0)), reverse=True)
            picked.append(rows[0])

        if not picked:
            fallback = sorted(
                feature_index.get(feature, []),
                key=lambda item: float(item.get("confidence", 0)),
                reverse=True,
            )
            picked = fallback[:3]

        items: List[Dict[str, Any]] = []
        for row in picked[:4]:
            status = str(row.get("status", "needs_manual_review"))
            confidence = float(row.get("confidence", 0))
            web_refs = row.get("web_refs", [])
            if isinstance(web_refs, list):
                web_refs = web_refs[:2]
            else:
                web_refs = []

            item = {
                "name": row.get("name"),
                "status": status,
                "confidence": confidence,
                "definition_ref": row.get("definition_ref", [])[:2],
                "web_refs": web_refs,
            }
            items.append(item)

            if status != STATUS_ALIGNED or confidence < KB_MIN_CONFIDENCE:
                alert = (
                    f"[{feature}] {row.get('name')} evidence is {status}, confidence={confidence:.2f}. "
                    "Keep recommendation but treat as risk."
                )
                if alert not in seen_alerts:
                    seen_alerts.add(alert)
                    conflict_alerts.append(alert)

        evidence_block.append({"feature": feature, "items": items})

    return evidence_block, conflict_alerts


def _apply_model_knowledge_compatibility(
    model_profile: str,
    requested_features: List[str],
    npu_count: int,
) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    knowledge = MODEL_KNOWLEDGE.get(model_profile, {})
    architecture = knowledge.get("architecture", {})
    quant_profile = knowledge.get("quantization_profile", {})
    feature_constraints = knowledge.get("feature_constraints", {})
    min_npu_map = knowledge.get("feature_min_npu_count", {})
    evidence_refs = knowledge.get("evidence_refs", [])

    has_moe_layers = bool(architecture.get("has_moe_layers", False))
    supported_variants = set(quant_profile.get("supported_variants", []))

    allowed: List[str] = []
    blocked: List[Dict[str, Any]] = []
    downgraded: List[Dict[str, Any]] = []
    reasonability_checks: List[Dict[str, Any]] = []

    for feature in requested_features:
        # Generic inference from architecture knowledge.
        if feature == "expert_parallel" and not has_moe_layers:
            reason = (
                f"Blocked by model knowledge: '{model_profile}' has no MoE layers "
                f"(has_moe_layers={has_moe_layers}), so expert_parallel is not applicable."
            )
            blocked.append(
                {
                    "feature": feature,
                    "reason": reason,
                    "level": "hard_block",
                    "source": "model_knowledge_inference",
                    "evidence_refs": evidence_refs,
                }
            )
            reasonability_checks.append(
                {
                    "feature": feature,
                    "result": "blocked",
                    "why": reason,
                    "inferred_from": "architecture.has_moe_layers",
                }
            )
            continue

        # Generic inference from quantization artifact knowledge.
        if feature == "int4_quantization" and "int4" not in supported_variants and "w4a4" not in supported_variants:
            fixed = str(quant_profile.get("fixed_weight_format", "unknown"))
            reason = (
                f"Blocked by model knowledge: '{model_profile}' quantization is fixed to {fixed}, "
                "INT4/W4A4 is not available in this profile."
            )
            blocked.append(
                {
                    "feature": feature,
                    "reason": reason,
                    "level": "hard_block",
                    "source": "model_knowledge_inference",
                    "evidence_refs": evidence_refs,
                }
            )
            reasonability_checks.append(
                {
                    "feature": feature,
                    "result": "blocked",
                    "why": reason,
                    "inferred_from": "quantization_profile.supported_variants",
                }
            )
            continue

        # Explicit profile constraints.
        if feature in feature_constraints:
            rule = feature_constraints[feature]
            reason = str(rule.get("reason", "Blocked by profile constraint."))
            level = str(rule.get("level", "hard_block"))
            blocked.append(
                {
                    "feature": feature,
                    "reason": reason,
                    "level": level,
                    "source": "model_profile_constraint",
                    "evidence_refs": evidence_refs,
                }
            )
            reasonability_checks.append(
                {
                    "feature": feature,
                    "result": "blocked",
                    "why": reason,
                    "inferred_from": "feature_constraints",
                }
            )
            continue

        # Hardware reasonability check.
        min_npu = int(min_npu_map.get(feature, 0) or 0)
        if min_npu > 0 and npu_count < min_npu:
            reason = (
                f"Requested feature '{feature}' is not reasonable for current hardware "
                f"(npu_count={npu_count} < required={min_npu})."
            )
            downgraded.append(
                {
                    "feature": feature,
                    "reason": reason,
                    "fallback": "Skip this feature and keep single-node safe defaults.",
                    "source": "hardware_reasonability_inference",
                    "evidence_refs": evidence_refs,
                }
            )
            reasonability_checks.append(
                {
                    "feature": feature,
                    "result": "downgraded",
                    "why": reason,
                    "inferred_from": "feature_min_npu_count",
                }
            )
            continue

        allowed.append(feature)
        reasonability_checks.append(
            {
                "feature": feature,
                "result": "allowed",
                "why": "No model/hardware conflict detected by model knowledge rules.",
                "inferred_from": "model_knowledge",
            }
        )

    return allowed, blocked, downgraded, reasonability_checks


def _build_command_parts(
    model_path: str,
    served_model_name: str,
    tp_size: int,
    port: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    gpu_memory_utilization: float,
    features: List[str],
) -> Dict[str, object]:
    cmd_parts: List[str] = [
        "vllm serve",
        shlex.quote(model_path),
        "--host 0.0.0.0",
        f"--port {port}",
        f"--served-model-name {shlex.quote(served_model_name)}",
        "--trust-remote-code",
        "--distributed-executor-backend mp",
        f"--tensor-parallel-size {tp_size}",
        f"--max-model-len {max_model_len}",
        f"--max-num-batched-tokens {max_num_batched_tokens}",
        "--block-size 128",
        f"--gpu-memory-utilization {gpu_memory_utilization}",
    ]

    env_exports = {
        "TASK_QUEUE_ENABLE": "1",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
    }

    risks: List[str] = []
    additional_config: Dict[str, object] = {}
    compilation_config: Dict[str, object] = {}

    if "quantization" in features:
        cmd_parts.append("--quantization ascend")

    if "graph_mode" in features:
        compilation_config["cudagraph_mode"] = "FULL_DECODE_ONLY"

    if "data_parallel" in features:
        cmd_parts.extend([
            "--data-parallel-size 2",
            "--data-parallel-size-local 2",
            "--data-parallel-address 127.0.0.1",
            "--data-parallel-rpc-port 13389",
        ])

    if "expert_parallel" in features:
        cmd_parts.append("--enable-expert-parallel")

    if "prefill_decode_disaggregation" in features:
        risks.append(
            "prefill_decode_disaggregation usually needs multi-node connector setup; demo package keeps single-node safe defaults."
        )

    if "context_parallel" in features:
        additional_config["context_parallel_size"] = 2

    if "lora" in features:
        cmd_parts.append("--enable-lora")

    if "speculative_decode" in features:
        spec_config = {"method": "mtp", "num_speculative_tokens": 1}
        cmd_parts.append(f"--speculative-config {shlex.quote(json.dumps(spec_config, separators=(",", ":")))}")

    if "sleep_mode" in features:
        cmd_parts.append("--enable-sleep-mode")

    if "weight_prefetch" in features:
        additional_config["weight_prefetch_config"] = {"enabled": True}

    if compilation_config:
        cmd_parts.append(
            "--compilation-config "
            + shlex.quote(json.dumps(compilation_config, separators=(",", ":")))
        )

    if additional_config:
        cmd_parts.append(
            "--additional-config "
            + shlex.quote(json.dumps(additional_config, separators=(",", ":")))
        )

    return {
        "cmd_parts": cmd_parts,
        "env_exports": env_exports,
        "risks": risks,
    }


def render_package(
    output_dir: Path,
    model_profile: str,
    model_path_override: str | None,
    hardware_type: str,
    npu_count: int,
    port_override: int | None,
    text: str | None,
    features_input: List[str],
) -> Dict[str, object]:
    if model_profile not in PROFILES:
        raise ValueError(f"Unsupported model profile: {model_profile}")

    profile = PROFILES[model_profile]
    normalization = normalize_input(text or "") if text else {
        "intent": "deploy_model",
        "features": [],
        "confidence": 1.0,
        "missing_slots": [],
        "clarification_question": "",
    }

    normalized_features = list(normalization["features"])
    requested_features = _dedupe(features_input + normalized_features)
    applied_features, blocked_features, downgraded_features, reasonability_checks = (
        _apply_model_knowledge_compatibility(
            model_profile=model_profile,
            requested_features=requested_features,
            npu_count=npu_count,
        )
    )
    model_knowledge = MODEL_KNOWLEDGE.get(model_profile, {})

    kb_index = _load_global_kb_entries()
    evidence_block, conflict_alerts = _build_evidence_block(requested_features, kb_index)

    model_path = model_path_override or str(profile["model"])
    served_model_name = str(profile["served_model_name"])
    tp_size = int(profile["tensor_parallel_size"])
    max_model_len = int(profile["max_model_len"])
    max_num_batched_tokens = int(profile["max_num_batched_tokens"])
    gpu_memory_utilization = float(profile["gpu_memory_utilization"])
    port = int(port_override or profile["default_port"])

    built = _build_command_parts(
        model_path=model_path,
        served_model_name=served_model_name,
        tp_size=tp_size,
        port=port,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        features=applied_features,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    start_script = output_dir / "start.sh"
    validate_script = output_dir / "validate.sh"
    rollback_script = output_dir / "rollback.sh"

    cmd_preview = " \\\n  ".join(built["cmd_parts"])
    env_lines = "\n".join(
        f'export {key}="{value}"' for key, value in built["env_exports"].items()
    )

    _write_file(
        start_script,
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
                env_lines,
                "",
                f"{cmd_preview} > \"${{SCRIPT_DIR}}/server.log\" 2>&1 &",
                'echo $! > "${SCRIPT_DIR}/server.pid"',
                'echo "Started vLLM server. PID=$(cat \"${SCRIPT_DIR}/server.pid\")"',
                'echo "Log file: ${SCRIPT_DIR}/server.log"',
            ]
        )
        + "\n",
    )

    validate_payload = {
        "model": served_model_name,
        "messages": [{"role": "user", "content": "请用一句话介绍vLLM-Ascend。"}],
        "max_completion_tokens": 32,
        "temperature": 0,
    }

    _write_file(
        validate_script,
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"PORT=${{1:-{port}}}",
                'echo "[check] /v1/models"',
                'curl -sf "http://127.0.0.1:${PORT}/v1/models" | tee /tmp/vllm_models.json >/dev/null',
                'echo "[check] /v1/chat/completions"',
                "curl -sf \"http://127.0.0.1:${PORT}/v1/chat/completions\" \\",
                "  -H 'Content-Type: application/json' \\",
                f"  -d '{json.dumps(validate_payload, ensure_ascii=False)}' | tee /tmp/vllm_chat_resp.json >/dev/null",
                'echo "Validation passed."',
            ]
        )
        + "\n",
    )

    _write_file(
        rollback_script,
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
                'if [ -f "${SCRIPT_DIR}/server.pid" ]; then',
                '  PID="$(cat "${SCRIPT_DIR}/server.pid")"',
                '  if kill -0 "${PID}" >/dev/null 2>&1; then',
                '    kill -2 "${PID}"',
                '    echo "Stopped PID ${PID}"',
                "  else",
                '    echo "PID ${PID} is not running"',
                "  fi",
                "else",
                '  echo "No server.pid found"',
                "fi",
            ]
        )
        + "\n",
    )

    risks = list(built["risks"])
    for blocked in blocked_features:
        risks.append(
            f"Blocked feature '{blocked['feature']}': {blocked['reason']}"
        )
    for downgraded in downgraded_features:
        risks.append(
            f"Downgraded feature '{downgraded['feature']}': {downgraded['reason']} "
            f"Fallback: {downgraded['fallback']}"
        )
    if normalization["missing_slots"]:
        risks.append(
            "Input is ambiguous; ask one clarification before production execution: "
            + str(normalization["clarification_question"])
        )
    risks.extend(conflict_alerts)

    result = {
        "deployment_plan": {
            "intent": normalization["intent"],
            "model_profile": model_profile,
            "model_path": model_path,
            "hardware_type": hardware_type,
            "npu_count": npu_count,
            "port": port,
            "canonical_features": applied_features,
            "canonical_features_requested": requested_features,
            "canonical_features_applied": applied_features,
            "model_knowledge": model_knowledge,
            "compatibility": {
                "allowed_features": applied_features,
                "blocked_features": blocked_features,
                "downgraded_features": downgraded_features,
                "reasonability_checks": reasonability_checks,
            },
            "evidence_block": evidence_block,
            "conflict_alerts": conflict_alerts,
            "normalization_confidence": normalization["confidence"],
            "missing_slots": normalization["missing_slots"],
            "clarification_question": normalization["clarification_question"],
            "risks": risks,
        },
        "generated_commands": {
            "start_script": str(start_script.resolve()),
            "validate_script": str(validate_script.resolve()),
            "rollback_script": str(rollback_script.resolve()),
            "start_command_preview": cmd_preview,
        },
        "validation_steps": [
            "Run start.sh and wait for server ready logs.",
            "Run validate.sh and verify HTTP 200 for /v1/models.",
            "Verify /v1/chat/completions returns non-empty text.",
        ],
        "rollback_steps": [
            "Run rollback.sh to stop current server.",
            "Adjust feature set or port and rerender package.",
            "Restart with new start.sh and rerun validate.sh.",
        ],
        "evidence_block": evidence_block,
        "conflict_alerts": conflict_alerts,
    }

    plan_path = output_dir / "deployment_plan.json"
    plan_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-profile", default="qwen3-32b-w8a8", choices=sorted(PROFILES.keys()))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--hardware-type", default="Atlas A2/A3")
    parser.add_argument("--npu-count", type=int, default=4)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--text", default=None, help="Natural language deployment request")
    parser.add_argument(
        "--features",
        default=None,
        help="Comma-separated canonical feature list (optional override)",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    feature_input = _parse_feature_list(args.features)
    result = render_package(
        output_dir=output_dir,
        model_profile=args.model_profile,
        model_path_override=args.model_path,
        hardware_type=args.hardware_type,
        npu_count=args.npu_count,
        port_override=args.port,
        text=args.text,
        features_input=feature_input,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
