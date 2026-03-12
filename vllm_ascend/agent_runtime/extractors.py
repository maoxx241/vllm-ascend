from __future__ import annotations

import json
from pathlib import Path
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
            ("entity-hw-a2", "hardware", "A2", json.dumps(["910B4"]), json.dumps(["ascend"]), json.dumps({})),
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
            "entity-hw-a2",
            "runtime_constraint",
            None,
            "Current validated runtime tuple uses CANN 8.5.0 on A2/910B4.",
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


__all__ = [
    "ShardRows",
    "empty_shard_rows",
    "extract_minimal_validation",
    "extract_repo_custom_ops",
    "extract_repo_semantics",
    "extract_runtime_caps",
    "merge_shard_rows",
]
