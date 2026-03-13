from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from .contracts import now_utc
from .paths import repo_root

ShardRows = dict[str, list[tuple[Any, ...]]]
TABLE_KEYS = ("sources", "entities", "facts", "edges", "symbol_index", "validations")


def empty_shard_rows() -> ShardRows:
    return {key: [] for key in TABLE_KEYS}


def merge_shard_rows(*shards: ShardRows) -> ShardRows:
    merged = empty_shard_rows()
    for shard in shards:
        for key in TABLE_KEYS:
            merged[key].extend(shard.get(key, []))
    for key in TABLE_KEYS:
        merged[key] = sorted(merged[key], key=lambda row: tuple(str(part) for part in row))
    return merged


def _paired_vllm_root(root: Path) -> Path | None:
    candidate = root.parent / "vllm"
    if candidate.exists():
        return candidate
    return None


def extract_repo_semantics(root: Path | None = None, resolve_result: dict[str, Any] | None = None) -> ShardRows:
    root = root or repo_root()
    resolve_result = resolve_result or {}
    repo_sha = resolve_result.get("repo_sha", "unknown")
    paired_vllm_ref = resolve_result.get("paired_vllm_ref", "unknown")
    runtime_tuple = resolve_result.get("runtime_tuple", {})

    shard = empty_shard_rows()
    shard["sources"].extend(
        [
            (
                "source-repo-prefill",
                "repo_file",
                "vllm_ascend/ascend_forward_context.py",
                None,
                repo_sha,
                paired_vllm_ref,
                "repo_semantics",
                None,
                json.dumps({"summary": "MoE communication policy for A2/A3 prefill/decode routing"}, ensure_ascii=False),
            ),
            (
                "source-repo-dynamic-batch",
                "repo_file",
                "vllm_ascend/core/scheduler_dynamic_batch.py",
                None,
                repo_sha,
                paired_vllm_ref,
                "repo_semantics",
                None,
                json.dumps({"summary": "Dynamic batching and chunked prefill scheduler behavior"}, ensure_ascii=False),
            ),
            (
                "source-repo-config",
                "repo_file",
                "vllm_ascend/ascend_config.py",
                None,
                repo_sha,
                paired_vllm_ref,
                "repo_semantics",
                None,
                json.dumps({"summary": "Ascend runtime config and prefill/decode parallel ratios"}, ensure_ascii=False),
            ),
            (
                "source-repo-cpu-binding",
                "repo_file",
                "vllm_ascend/cpu_binding.py",
                None,
                repo_sha,
                paired_vllm_ref,
                "repo_semantics",
                None,
                json.dumps({"summary": "A3 logical NPU numbering and physical-card topology mapping"}, ensure_ascii=False),
            ),
            (
                "source-doc-qwen3-dense",
                "repo_doc",
                "docs/source/tutorials/models/Qwen3-Dense.md",
                None,
                repo_sha,
                paired_vllm_ref,
                "repo_semantics",
                None,
                json.dumps({"summary": "Qwen3-32B-W8A8 A3 deployment baseline and launch guidance"}, ensure_ascii=False),
            ),
            (
                "source-doc-deepseek-v3_2",
                "repo_doc",
                "docs/source/tutorials/models/DeepSeek-V3.2.md",
                None,
                repo_sha,
                paired_vllm_ref,
                "repo_semantics",
                None,
                json.dumps({"summary": "DeepSeek-V3.2 A3 single-node deployment and performance reference"}, ensure_ascii=False),
            ),
        ]
    )
    shard["entities"].extend(
        [
            ("entity-model-qwen3-next", "model", "qwen3-next", json.dumps([]), json.dumps(["deployment"]), json.dumps({})),
            (
                "entity-model-qwen3-next-32b",
                "model",
                "qwen3-next-32b",
                json.dumps([]),
                json.dumps(["performance"]),
                json.dumps({}),
            ),
            (
                "entity-model-qwen3-32b",
                "model",
                "qwen3-32b",
                json.dumps(["Qwen3-32B"]),
                json.dumps(["deployment", "performance"]),
                json.dumps({}),
            ),
            (
                "entity-model-qwen3-32b-w8a8",
                "model",
                "qwen3-32b-w8a8",
                json.dumps(["Qwen3-32B-W8A8"]),
                json.dumps(["deployment", "performance"]),
                json.dumps({}),
            ),
            (
                "entity-model-deepseek-v3",
                "model",
                "deepseek-v3",
                json.dumps(["DeepSeek-V3", "DeepSeek-V3.2-W8A8"]),
                json.dumps(["performance"]),
                json.dumps({}),
            ),
            ("entity-hw-a2", "hardware", "A2", json.dumps(["910B4"]), json.dumps(["ascend"]), json.dumps({})),
            ("entity-hw-a3", "hardware", "A3", json.dumps(["910_9391"]), json.dumps(["ascend"]), json.dumps({})),
            ("entity-feature-prefill", "feature", "prefill", json.dumps([]), json.dumps(["runtime"]), json.dumps({})),
            ("entity-feature-decode", "feature", "decode", json.dumps([]), json.dumps(["runtime"]), json.dumps({})),
            (
                "entity-feature-dynamic-batching",
                "feature",
                "dynamic_batching",
                json.dumps(["dynamic batch"]),
                json.dumps(["validation"]),
                json.dumps({}),
            ),
            (
                "entity-feature-allgather-ep",
                "feature",
                "allgather_ep",
                json.dumps(["allgather ep"]),
                json.dumps(["deployment"]),
                json.dumps({}),
            ),
            ("entity-policy-prefill", "policy", "repo.policy.prefill", json.dumps([]), json.dumps(["repo"]), json.dumps({})),
            ("entity-config-tp4-bf16-ctx8k", "config", "tp4_bf16_ctx8k", json.dumps([]), json.dumps(["performance"]), json.dumps({})),
        ]
    )
    shard["facts"].extend(
        [
            (
                "fact-prefill-a2",
                "entity-policy-prefill",
                "supports_hw",
                "entity-hw-a2",
                "A2 prefill/decode path is bounded by MC2 capacity and may fall back to all-gather.",
                0.92,
                None,
                None,
                json.dumps({"models": ["qwen3-next"], "features": ["prefill", "decode"]}, ensure_ascii=False),
                "source-repo-prefill",
                "repo_semantics",
                json.dumps({}),
            ),
            (
                "fact-allgather-ep",
                "entity-feature-allgather-ep",
                "deployment_usage",
                None,
                "Without expert parallel or world_size=1, the runtime falls back to all-gather communication.",
                0.88,
                None,
                None,
                json.dumps({"hw": ["A2"]}, ensure_ascii=False),
                "source-repo-prefill",
                "repo_semantics",
                json.dumps({}),
            ),
            (
                "fact-dynamic-batch-budget",
                "entity-feature-dynamic-batching",
                "scheduler_policy",
                None,
                "Dynamic batching refines token budget and depends on chunked prefill scheduling.",
                0.84,
                None,
                None,
                json.dumps({"hw": [runtime_tuple.get("soc", "A2")]}, ensure_ascii=False),
                "source-repo-dynamic-batch",
                "repo_semantics",
                json.dumps({}),
            ),
            (
                "fact-a3-card-die-topology",
                "entity-hw-a3",
                "topology_mapping",
                None,
                "On A3, logical npu_id is computed as card_id*2 + chip_id, so one physical card exposes 2 logical NPUs (dies).",
                0.97,
                None,
                None,
                json.dumps({"hw": ["A3"], "physical_card_to_logical_npus": 2}, ensure_ascii=False),
                "source-repo-cpu-binding",
                "repo_semantics",
                json.dumps({}),
            ),
            (
                "fact-qwen3-32b-a3-deployment",
                "entity-model-qwen3-32b",
                "deployment_baseline",
                "entity-hw-a3",
                "Qwen3-32B on A3 is documented around a TP4 / 2-card / 4-logical-NPU baseline; requests for other A3 card counts should first convert 1 card to 2 logical NPUs.",
                0.94,
                None,
                None,
                json.dumps({"hw": ["A3"], "configs": ["tp4", "bf16"], "physical_cards": 2, "logical_npus": 4}, ensure_ascii=False),
                "source-doc-qwen3-dense",
                "repo_semantics",
                json.dumps({}),
            ),
            (
                "fact-qwen3-32b-w8a8-a3-deployment",
                "entity-model-qwen3-32b-w8a8",
                "deployment_baseline",
                "entity-hw-a3",
                "Qwen3-32B-W8A8 on A3 is documented around a TP4 / 2-card / 4-logical-NPU baseline; requests for other A3 card counts should first convert 1 card to 2 logical NPUs.",
                0.95,
                None,
                None,
                json.dumps({"hw": ["A3"], "configs": ["tp4", "w8a8"], "physical_cards": 2, "logical_npus": 4}, ensure_ascii=False),
                "source-doc-qwen3-dense",
                "repo_semantics",
                json.dumps({}),
            ),
            (
                "fact-deepseek-v3-a3-envelope",
                "entity-model-deepseek-v3",
                "performance_baseline",
                "entity-hw-a3",
                "DeepSeek-V3 on A3 should be reasoned from A3 single-node W8A8 references and topology-sensitive assumptions, not presented as a measured single point.",
                0.87,
                None,
                None,
                json.dumps({"hw": ["A3"], "features": ["prefill", "decode"]}, ensure_ascii=False),
                "source-doc-deepseek-v3_2",
                "repo_semantics",
                json.dumps({}),
            ),
        ]
    )
    shard["edges"].extend(
        [
            (
                "edge-policy-prefill-a2",
                "entity-policy-prefill",
                "targets",
                "entity-hw-a2",
                1.0,
                "source-repo-prefill",
                json.dumps({}),
            ),
            (
                "edge-dynamic-batching-prefill",
                "entity-feature-dynamic-batching",
                "touches",
                "entity-feature-prefill",
                0.8,
                "source-repo-dynamic-batch",
                json.dumps({}),
            ),
            (
                "edge-qwen3-32b-a3",
                "entity-model-qwen3-32b",
                "targets",
                "entity-hw-a3",
                1.0,
                "source-doc-qwen3-dense",
                json.dumps({}),
            ),
            (
                "edge-qwen3-32b-w8a8-a3",
                "entity-model-qwen3-32b-w8a8",
                "targets",
                "entity-hw-a3",
                1.0,
                "source-doc-qwen3-dense",
                json.dumps({}),
            ),
            (
                "edge-deepseek-v3-a3",
                "entity-model-deepseek-v3",
                "targets",
                "entity-hw-a3",
                1.0,
                "source-doc-deepseek-v3_2",
                json.dumps({}),
            ),
        ]
    )
    shard["symbol_index"].extend(
        [
            (
                "symbol-select-moe-comm-method",
                "select_moe_comm_method",
                "function",
                "vllm_ascend/ascend_forward_context.py",
                "select_moe_comm_method(num_tokens, vllm_config, is_draft_model=False)",
                "vllm_ascend.ascend_forward_context",
                "vllm_ascend/ascend_forward_context.py",
                paired_vllm_ref,
                json.dumps({"surfaces": ["prefill", "decode", "allgather_ep"]}, ensure_ascii=False),
            ),
            (
                "symbol-dynamic-batch-scheduler",
                "SchedulerDynamicBatch",
                "class",
                "vllm_ascend/core/scheduler_dynamic_batch.py",
                None,
                "vllm_ascend.core.scheduler_dynamic_batch",
                "vllm_ascend/core/scheduler_dynamic_batch.py",
                paired_vllm_ref,
                json.dumps({"surfaces": ["dynamic_batching", "prefill"]}, ensure_ascii=False),
            ),
        ]
    )
    return shard


def extract_vllm_semantics(root: Path | None = None, resolve_result: dict[str, Any] | None = None) -> ShardRows:
    root = root or repo_root()
    resolve_result = resolve_result or {}
    repo_sha = resolve_result.get("repo_sha", "unknown")
    paired_vllm_ref = resolve_result.get("paired_vllm_ref", "unknown")
    vllm_root = _paired_vllm_root(root)
    if vllm_root is None:
        return empty_shard_rows()

    arg_utils_path = vllm_root / "vllm" / "engine" / "arg_utils.py"
    readme_path = vllm_root / "README.md"
    arg_utils_text = arg_utils_path.read_text(encoding="utf-8") if arg_utils_path.exists() else ""
    readme_text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

    shard = empty_shard_rows()
    shard["sources"].extend(
        [
            (
                "source-vllm-engine-arg-utils",
                "repo_file",
                "vllm/vllm/engine/arg_utils.py",
                None,
                repo_sha,
                paired_vllm_ref,
                "vllm_semantics",
                None,
                json.dumps({"summary": "Upstream EngineArgs and CLI-exposed engine configuration semantics"}, ensure_ascii=False),
            ),
            (
                "source-vllm-readme",
                "repo_file",
                "vllm/README.md",
                None,
                repo_sha,
                paired_vllm_ref,
                "vllm_semantics",
                None,
                json.dumps({"summary": "Upstream public usage surface and release-facing configuration context"}, ensure_ascii=False),
            ),
        ]
    )
    shard["entities"].extend(
        [
            (
                "entity-vllm-config-tensor-parallel-size",
                "config",
                "vllm.config.tensor_parallel_size",
                json.dumps(["tp", "tensor-parallel-size"]),
                json.dumps(["upstream", "parallelism"]),
                json.dumps({}),
            ),
            (
                "entity-vllm-config-max-model-len",
                "config",
                "vllm.config.max_model_len",
                json.dumps(["context length", "max-model-len"]),
                json.dumps(["upstream", "scheduler"]),
                json.dumps({}),
            ),
            (
                "entity-vllm-engine-config",
                "feature",
                "vllm.engine.config",
                json.dumps(["EngineArgs", "create_engine_config"]),
                json.dumps(["upstream", "engine"]),
                json.dumps({}),
            ),
        ]
    )
    if "tensor_parallel_size" in arg_utils_text:
        shard["facts"].append(
            (
                "fact-vllm-tensor-parallel-semantics",
                "entity-vllm-config-tensor-parallel-size",
                "upstream_semantics",
                None,
                "Upstream EngineArgs exposes tensor_parallel_size and uses it when creating engine configuration.",
                0.93,
                None,
                None,
                json.dumps({"surface": "parallelism", "repo": "vllm"}, ensure_ascii=False),
                "source-vllm-engine-arg-utils",
                "vllm_semantics",
                json.dumps({"paired_vllm_ref": paired_vllm_ref}, ensure_ascii=False),
            )
        )
    if "max_model_len" in arg_utils_text:
        shard["facts"].append(
            (
                "fact-vllm-max-model-len-semantics",
                "entity-vllm-config-max-model-len",
                "upstream_semantics",
                None,
                "Upstream EngineArgs carries max_model_len into engine configuration and scheduler-related defaults.",
                0.9,
                None,
                None,
                json.dumps({"surface": "context", "repo": "vllm"}, ensure_ascii=False),
                "source-vllm-engine-arg-utils",
                "vllm_semantics",
                json.dumps({"paired_vllm_ref": paired_vllm_ref}, ensure_ascii=False),
            )
        )
    if "OpenAI" in readme_text or "LLM" in readme_text:
        shard["facts"].append(
            (
                "fact-vllm-readme-usage-surface",
                "entity-vllm-engine-config",
                "usage_surface",
                None,
                "Upstream public usage docs confirm that serving configuration is expressed through EngineArgs-backed CLI/runtime options.",
                0.78,
                None,
                None,
                json.dumps({"surface": "serving", "repo": "vllm"}, ensure_ascii=False),
                "source-vllm-readme",
                "vllm_semantics",
                json.dumps({"paired_vllm_ref": paired_vllm_ref}, ensure_ascii=False),
            )
        )
    return shard


def extract_vllm_symbols(root: Path | None = None, resolve_result: dict[str, Any] | None = None) -> ShardRows:
    root = root or repo_root()
    resolve_result = resolve_result or {}
    paired_vllm_ref = resolve_result.get("paired_vllm_ref", "unknown")
    vllm_root = _paired_vllm_root(root)
    if vllm_root is None:
        return empty_shard_rows()

    arg_utils_path = vllm_root / "vllm" / "engine" / "arg_utils.py"
    if not arg_utils_path.exists():
        return empty_shard_rows()
    arg_utils_text = arg_utils_path.read_text(encoding="utf-8")

    class_sig = None
    method_sig = None
    class_match = re.search(r"^class EngineArgs:", arg_utils_text, re.M)
    method_match = re.search(
        r"^\s+def create_engine_config\(\s*$[\s\S]*?^\s+\)\s*->\s*VllmConfig:",
        arg_utils_text,
        re.M,
    )
    if class_match:
        class_sig = "class EngineArgs"
    if method_match:
        method_sig = "def create_engine_config(self, usage_context: UsageContext | None = None, headless: bool = False) -> VllmConfig"

    shard = empty_shard_rows()
    shard["symbol_index"].extend(
        [
            (
                "symbol-vllm-engineargs",
                "EngineArgs",
                "class",
                "vllm/vllm/engine/arg_utils.py",
                class_sig,
                "vllm.engine.arg_utils",
                "vllm/engine/arg_utils.py",
                paired_vllm_ref,
                json.dumps({"surfaces": ["engine_config", "cli"]}, ensure_ascii=False),
            ),
            (
                "symbol-vllm-engineargs-create-engine-config",
                "EngineArgs.create_engine_config",
                "method",
                "vllm/vllm/engine/arg_utils.py",
                method_sig,
                "vllm.engine.arg_utils",
                "vllm/engine/arg_utils.py",
                paired_vllm_ref,
                json.dumps({"surfaces": ["engine_config", "builder"]}, ensure_ascii=False),
            ),
        ]
    )
    return shard


def extract_vllm_release_delta(root: Path | None = None, resolve_result: dict[str, Any] | None = None) -> ShardRows:
    root = root or repo_root()
    resolve_result = resolve_result or {}
    repo_sha = resolve_result.get("repo_sha", "unknown")
    paired_vllm_ref = resolve_result.get("paired_vllm_ref", "unknown")
    vllm_root = _paired_vllm_root(root)
    if vllm_root is None:
        return empty_shard_rows()

    release_path = vllm_root / "RELEASE.md"
    if not release_path.exists():
        return empty_shard_rows()
    release_text = release_path.read_text(encoding="utf-8")
    from_version = "unknown"
    since_match = re.search(r"Since\s+(v\d+\.\d+\.\d+)", release_text)
    if since_match:
        from_version = since_match.group(1)

    shard = empty_shard_rows()
    shard["sources"].append(
        (
            "source-vllm-release-md",
            "repo_file",
            "vllm/RELEASE.md",
            None,
            repo_sha,
            paired_vllm_ref,
            "vllm_release_delta",
            None,
            json.dumps({"summary": "Upstream release process and validation window documentation"}, ensure_ascii=False),
        )
    )
    shard["entities"].append(
        (
            "entity-vllm-release-delta",
            "feature",
            "vllm.release.delta",
            json.dumps(["release delta", "upstream sync"]),
            json.dumps(["upstream", "release"]),
            json.dumps({}),
        )
    )
    shard["facts"].append(
        (
            "fact-vllm-release-delta-current",
            "entity-vllm-release-delta",
            "release_delta",
            None,
            "Upstream release notes describe cadence, release branches, and validation windows that bound sync impact analysis.",
            0.82,
            None,
            None,
            json.dumps({"surface": "release_process"}, ensure_ascii=False),
            "source-vllm-release-md",
            "vllm_release_delta",
            json.dumps(
                {
                    "from_version": from_version,
                    "to_version": paired_vllm_ref[:12] if paired_vllm_ref != "unknown" else "unknown",
                    "impact_tags": ["release_policy", "validation_window", "engine_config"],
                },
                ensure_ascii=False,
            ),
        )
    )
    return shard


def extract_repo_custom_ops(root: Path | None = None, resolve_result: dict[str, Any] | None = None) -> ShardRows:
    root = root or repo_root()
    resolve_result = resolve_result or {}
    repo_sha = resolve_result.get("repo_sha", "unknown")
    paired_vllm_ref = resolve_result.get("paired_vllm_ref", "unknown")

    shard = empty_shard_rows()
    shard["sources"].extend(
        [
            (
                "source-custom-ops-register",
                "repo_file",
                "vllm_ascend/ops/register_custom_ops.py",
                None,
                repo_sha,
                paired_vllm_ref,
                "repo_custom_ops",
                None,
                json.dumps({"summary": "torch_npu custom op registration and fake impl surfaces"}, ensure_ascii=False),
            ),
            (
                "source-custom-ops-aclgraph",
                "repo_file",
                "vllm_ascend/compilation/acl_graph.py",
                None,
                repo_sha,
                paired_vllm_ref,
                "repo_custom_ops",
                None,
                json.dumps({"summary": "ACL graph capture/cudagraph-adjacent runtime hooks"}, ensure_ascii=False),
            ),
        ]
    )
    shard["entities"].extend(
        [
            (
                "entity-custom-op-overlay",
                "feature",
                "repo.custom_ops.overlay",
                json.dumps(["torch_npu custom ops"]),
                json.dumps(["ops"]),
                json.dumps({}),
            ),
            (
                "entity-aclgraph-runtime",
                "feature",
                "repo.aclgraph.runtime",
                json.dumps(["acl graph"]),
                json.dumps(["graph"]),
                json.dumps({}),
            ),
        ]
    )
    shard["facts"].extend(
        [
            (
                "fact-custom-op-overlay",
                "entity-custom-op-overlay",
                "build_surface",
                None,
                "Custom op overlay registers torch_npu-backed kernels and fake impls for capture-safe execution.",
                0.85,
                None,
                None,
                json.dumps({"surfaces": ["torch_npu", "custom_op", "graph_capture"]}, ensure_ascii=False),
                "source-custom-ops-register",
                "repo_custom_ops",
                json.dumps({"updated_at": now_utc()}, ensure_ascii=False),
            ),
            (
                "fact-aclgraph-runtime",
                "entity-aclgraph-runtime",
                "capture_surface",
                None,
                "ACL graph runtime hooks constrain graph capture behavior and related hardening paths.",
                0.8,
                None,
                None,
                json.dumps({"surfaces": ["aclgraph", "capture"]}, ensure_ascii=False),
                "source-custom-ops-aclgraph",
                "repo_custom_ops",
                json.dumps({}),
            ),
        ]
    )
    shard["edges"].append(
        (
            "edge-custom-op-aclgraph",
            "entity-custom-op-overlay",
            "depends_on",
            "entity-aclgraph-runtime",
            0.6,
            "source-custom-ops-aclgraph",
            json.dumps({}),
        )
    )
    shard["symbol_index"].append(
        (
            "symbol-custom-op-register-file",
            "direct_register_custom_op",
            "callsite_group",
            "vllm_ascend/ops/register_custom_ops.py",
            None,
            "vllm_ascend.ops.register_custom_ops",
            "vllm_ascend/ops/register_custom_ops.py",
            paired_vllm_ref,
            json.dumps({"surfaces": ["custom_op", "torch_npu"]}, ensure_ascii=False),
        )
    )
    return shard


def extract_minimal_validation(root: Path | None = None, resolve_result: dict[str, Any] | None = None) -> ShardRows:
    root = root or repo_root()
    resolve_result = resolve_result or {}
    repo_sha = resolve_result.get("repo_sha", "unknown")
    paired_vllm_ref = resolve_result.get("paired_vllm_ref", "unknown")
    runtime_tuple = resolve_result.get("runtime_tuple", {})

    shard = empty_shard_rows()
    shard["sources"].extend(
        [
            (
                "source-val-async",
                "repo_test",
                "tests/e2e/singlecard/test_async_scheduling.py",
                None,
                repo_sha,
                paired_vllm_ref,
                "validation",
                None,
                json.dumps({"summary": "single-card async scheduling smoke/regression"}, ensure_ascii=False),
            ),
            (
                "source-val-dynamic-batch",
                "repo_test",
                "tests/ut/core/test_scheduler_dynamic_batch.py",
                None,
                repo_sha,
                paired_vllm_ref,
                "validation",
                None,
                json.dumps({"summary": "dynamic batch unit tests"}, ensure_ascii=False),
            ),
            (
                "source-val-qwen3-32b-a3",
                "repo_test",
                "tests/e2e/nightly/single_node/models/configs/Qwen3-32B.yaml",
                None,
                repo_sha,
                paired_vllm_ref,
                "validation",
                None,
                json.dumps({"summary": "Qwen3-32B BF16 A3 2-card / 4-logical-NPU TP4 single-node/nightly baseline"}, ensure_ascii=False),
            ),
            (
                "source-val-qwen3-32b-int8-a3",
                "repo_test",
                "tests/e2e/nightly/single_node/models/configs/Qwen3-32B-Int8.yaml",
                None,
                repo_sha,
                paired_vllm_ref,
                "validation",
                None,
                json.dumps({"summary": "Qwen3-32B-W8A8 A3 2-card / 4-logical-NPU TP4 single-node/nightly baseline"}, ensure_ascii=False),
            ),
            (
                "source-val-deepseek-v3-a3",
                "repo_test",
                "tests/e2e/nightly/single_node/models/configs/DeepSeek-V3.2-W8A8.yaml",
                None,
                repo_sha,
                paired_vllm_ref,
                "validation",
                None,
                json.dumps({"summary": "DeepSeek-V3 A3 single-node W8A8 performance reference"}, ensure_ascii=False),
            ),
        ]
    )
    shard["validations"].extend(
        [
            (
                "validation:baseline:qwen3-next:a2",
                "entity-model-qwen3-next",
                "model",
                "compatible_baseline",
                "pass",
                json.dumps({"hw": "A2", "cann": runtime_tuple.get("cann", "unknown")}, ensure_ascii=False),
                json.dumps(["tests/e2e/singlecard/test_async_scheduling.py"], ensure_ascii=False),
                "compatible baseline for deployment policy lookups",
                "source-val-async",
                json.dumps({}),
            ),
            (
                "validation:baseline:qwen3-next-32b:a2:tp4",
                "entity-model-qwen3-next-32b",
                "model",
                "compatible_baseline",
                "pass",
                json.dumps({"hw": "A2", "config": "tp4_bf16_ctx8k"}, ensure_ascii=False),
                json.dumps(["tests/e2e/singlecard/test_async_scheduling.py"], ensure_ascii=False),
                "comparable baseline for expected TTFT and throughput envelope",
                "source-val-async",
                json.dumps({}),
            ),
            (
                "validation:matrix:prefill",
                "entity-feature-prefill",
                "feature",
                "matrix",
                "pass",
                json.dumps({"hw": "A2"}, ensure_ascii=False),
                json.dumps(
                    [
                        "tests/ut/core/test_scheduler_dynamic_batch.py",
                        "tests/e2e/singlecard/test_async_scheduling.py",
                    ],
                    ensure_ascii=False,
                ),
                "prefill and scheduler validation assets available",
                "source-val-dynamic-batch",
                json.dumps({}),
            ),
            (
                "validation:baseline:qwen3-32b:a3:tp4",
                "entity-model-qwen3-32b",
                "model",
                "documented_baseline",
                "pass",
                json.dumps({"hw": "A3", "config": "tp4", "topology": "2_cards_4_logical_npus", "physical_cards": 2, "logical_npus": 4}, ensure_ascii=False),
                json.dumps(
                    [
                        "docs/source/tutorials/models/Qwen3-Dense.md",
                        "tests/e2e/nightly/single_node/models/configs/Qwen3-32B.yaml",
                    ],
                    ensure_ascii=False,
                ),
                "documented A3 2-card / 4-logical-NPU deployment baseline for Qwen3-32B",
                "source-val-qwen3-32b-a3",
                json.dumps({}),
            ),
            (
                "validation:baseline:qwen3-32b-w8a8:a3:tp4",
                "entity-model-qwen3-32b-w8a8",
                "model",
                "documented_baseline",
                "pass",
                json.dumps({"hw": "A3", "config": "tp4", "topology": "2_cards_4_logical_npus", "physical_cards": 2, "logical_npus": 4}, ensure_ascii=False),
                json.dumps(
                    [
                        "docs/source/tutorials/models/Qwen3-Dense.md",
                        "tests/e2e/nightly/single_node/models/configs/Qwen3-32B-Int8.yaml",
                    ],
                    ensure_ascii=False,
                ),
                "documented A3 2-card / 4-logical-NPU deployment baseline for Qwen3-32B-W8A8",
                "source-val-qwen3-32b-int8-a3",
                json.dumps({}),
            ),
            (
                "validation:baseline:deepseek-v3:a3:single-node",
                "entity-model-deepseek-v3",
                "model",
                "documented_baseline",
                "pass",
                json.dumps({"hw": "A3", "config": "single_node_w8a8"}, ensure_ascii=False),
                json.dumps(
                    [
                        "docs/source/tutorials/models/DeepSeek-V3.2.md",
                        "tests/e2e/nightly/single_node/models/configs/DeepSeek-V3.2-W8A8.yaml",
                    ],
                    ensure_ascii=False,
                ),
                "documented A3 single-node expectation anchor for DeepSeek-V3 family",
                "source-val-deepseek-v3-a3",
                json.dumps({}),
            ),
        ]
    )
    return shard


def extract_runtime_caps(resolve_result: dict[str, Any]) -> ShardRows:
    runtime_tuple = resolve_result["runtime_tuple"]
    repo_sha = resolve_result["repo_sha"]
    paired_vllm_ref = resolve_result["paired_vllm_ref"]
    shard = empty_shard_rows()
    shard["sources"].append(
        (
            "source-runtime-tuple",
            "runtime",
            None,
            None,
            repo_sha,
            paired_vllm_ref,
            "hw_runtime_caps",
            None,
            json.dumps(runtime_tuple, ensure_ascii=False),
        )
    )
    shard["facts"].append(
        (
            "fact-runtime-cann",
            "entity-hw-a3" if runtime_tuple.get("soc") == "A3" else "entity-hw-a2",
            "runtime_constraint",
            None,
            f"Current validated runtime tuple uses CANN {runtime_tuple.get('cann', 'unknown')} on {runtime_tuple.get('soc', 'unknown')}.",
            1.0,
            None,
            None,
            json.dumps(runtime_tuple, ensure_ascii=False),
            "source-runtime-tuple",
            "hw_runtime_caps",
            json.dumps({}),
        )
    )
    return shard


def extract_hw_soc_detail(resolve_result: dict[str, Any]) -> ShardRows:
    runtime_tuple = resolve_result["runtime_tuple"]
    repo_sha = resolve_result["repo_sha"]
    paired_vllm_ref = resolve_result["paired_vllm_ref"]
    soc = runtime_tuple.get("soc", "unknown")
    if soc == "unknown":
        return empty_shard_rows()
    shard = empty_shard_rows()
    shard["sources"].append(
        (
            "source-hw-soc-detail",
            "runtime",
            None,
            None,
            repo_sha,
            paired_vllm_ref,
            "hw_soc_detail",
            None,
            json.dumps({"soc": soc}, ensure_ascii=False),
        )
    )
    if soc == "A3":
        shard["facts"].append(
            (
                "fact-a3-card-die-topology",
                "entity-hw-a3",
                "soc_profile",
                None,
                "A3 topology uses logical npu_id = card_id*2 + chip_id, so one physical card maps to 2 logical NPUs (dies).",
                1.0,
                None,
                None,
                json.dumps({"soc": soc, "physical_card_to_logical_npus": 2}, ensure_ascii=False),
                "source-hw-soc-detail",
                "hw_soc_detail",
                json.dumps({}),
            )
        )
    shard["facts"].append(
        (
            f"fact-hw-soc-{soc.lower()}",
            "entity-hw-a3" if soc == "A3" else "entity-hw-a2",
            "soc_profile",
            None,
            f"{soc} runtime tuple is available and can be used as a formal hardware selector for capsule ranking.",
            1.0,
            None,
            None,
            json.dumps({"soc": soc}, ensure_ascii=False),
            "source-hw-soc-detail",
            "hw_soc_detail",
            json.dumps({}),
        )
    )
    return shard


def extract_cann_op_constraints(resolve_result: dict[str, Any]) -> ShardRows:
    runtime_tuple = resolve_result["runtime_tuple"]
    repo_sha = resolve_result["repo_sha"]
    paired_vllm_ref = resolve_result["paired_vllm_ref"]
    cann = runtime_tuple.get("cann", "unknown")
    if cann == "unknown":
        return empty_shard_rows()
    shard = empty_shard_rows()
    shard["sources"].append(
        (
            "source-cann-constraints",
            "runtime",
            None,
            None,
            repo_sha,
            paired_vllm_ref,
            "cann_op_constraints",
            None,
            json.dumps({"cann": cann}, ensure_ascii=False),
        )
    )
    shard["facts"].append(
        (
            "fact-cann-version-constraint",
            "entity-hw-a3" if runtime_tuple.get("soc") == "A3" else "entity-hw-a2",
            "runtime_constraint",
            None,
            f"CANN {cann} constraints are available for ranking deployment and expectation capsules.",
            0.92,
            None,
            None,
            json.dumps({"cann": cann}, ensure_ascii=False),
            "source-cann-constraints",
            "cann_op_constraints",
            json.dumps({}),
        )
    )
    return shard


def extract_torch_npu_bindings(resolve_result: dict[str, Any]) -> ShardRows:
    runtime_tuple = resolve_result["runtime_tuple"]
    repo_sha = resolve_result["repo_sha"]
    paired_vllm_ref = resolve_result["paired_vllm_ref"]
    torch_npu = runtime_tuple.get("torch_npu", "unknown")
    if torch_npu == "unknown":
        return empty_shard_rows()
    shard = empty_shard_rows()
    shard["sources"].append(
        (
            "source-torch-npu-bindings",
            "runtime",
            "vllm_ascend/ops/register_custom_ops.py",
            None,
            repo_sha,
            paired_vllm_ref,
            "torch_npu_bindings",
            None,
            json.dumps({"torch_npu": torch_npu}, ensure_ascii=False),
        )
    )
    shard["facts"].append(
        (
            "fact-torch-npu-bindings",
            "entity-custom-op-overlay",
            "binding_surface",
            None,
            f"torch_npu {torch_npu} bindings are available for custom-op and runtime capability analysis.",
            0.9,
            None,
            None,
            json.dumps({"torch_npu": torch_npu}, ensure_ascii=False),
            "source-torch-npu-bindings",
            "torch_npu_bindings",
            json.dumps({}),
        )
    )
    return shard


__all__ = [
    "ShardRows",
    "empty_shard_rows",
    "extract_cann_op_constraints",
    "extract_hw_soc_detail",
    "extract_minimal_validation",
    "extract_repo_custom_ops",
    "extract_repo_semantics",
    "extract_runtime_caps",
    "extract_torch_npu_bindings",
    "extract_vllm_release_delta",
    "extract_vllm_semantics",
    "extract_vllm_symbols",
    "merge_shard_rows",
]
