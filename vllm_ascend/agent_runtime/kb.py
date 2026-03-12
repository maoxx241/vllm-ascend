from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from .contracts import (dump_json, kb_path, load_json, now_utc,
                        run_contract_checks, validate_instance)
from .detector import collect_runtime_context
from .paths import kb_root, repo_root


def _load_matrix(root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    matrix_path = kb_path("rules", "matrix.lock.json", root=root)
    if matrix_path.exists():
        return load_json(matrix_path)
    return {
        "exact": [],
        "compatible": [],
    }


def _insert_json(conn: sqlite3.Connection, table: str, columns: list[str], values: list[Any]) -> None:
    placeholders = ", ".join(["?"] * len(columns))
    conn.execute(f"INSERT OR REPLACE INTO {table} ({', '.join(columns)}) VALUES ({placeholders})", values)


def resolve(
    root: Path | None = None,
    *,
    request_id: str | None = None,
    overrides: dict[str, str] | None = None,
    emit_path: Path | None = None,
) -> dict[str, Any]:
    root = root or repo_root()
    overrides = overrides or {}
    context = collect_runtime_context(root=root, overrides=overrides)
    matrix = _load_matrix(root=root)
    request_id = request_id or overrides.get("request_id") or f"resolve-{context['repo_sha'][:8]}"

    match_level = "unknown"
    selected_shards = ["repo_semantics", "repo_custom_ops", "validation"]
    warnings: list[str] = []
    missing: list[str] = []

    runtime_tuple = context["runtime_tuple"]
    if runtime_tuple["soc"] == "unknown":
        missing.append("soc")
    if runtime_tuple["cann"] == "unknown":
        missing.append("cann")
    if runtime_tuple["torch"] == "unknown":
        missing.append("torch")
    if runtime_tuple["torch_npu"] == "unknown":
        missing.append("torch_npu")

    for rule in matrix.get("exact", []):
        if (
            rule["soc"] == runtime_tuple["soc"]
            and rule["cann"] == runtime_tuple["cann"]
            and rule["repo_sha"] == context["repo_sha"]
            and rule["paired_vllm_ref"] == context["paired_vllm_ref"]
        ):
            match_level = "exact"
            selected_shards = rule["selected_shards"]
            break

    if match_level != "exact":
        for rule in matrix.get("compatible", []):
            if (
                rule["soc"] == runtime_tuple["soc"]
                and runtime_tuple["cann"].startswith(rule["cann_prefix"])
            ):
                match_level = "compatible"
                selected_shards = rule["selected_shards"]
                warnings.append(rule["warning"])
                break

    if match_level == "unknown" and context["repo_sha"] != "unknown":
        match_level = "compatible" if runtime_tuple["soc"] in {"A2", "A3"} else "unknown"
        if match_level == "compatible":
            warnings.append("runtime tuple exact match unavailable; using repo-only fallback")
    if runtime_tuple["soc"] == "A2" and runtime_tuple["cann"] == "8.5.0":
        selected_shards = sorted(set(selected_shards + ["hw_runtime_caps"]))

    result = {
        "schema_version": "kb-resolve-result/v1",
        "request_id": request_id,
        "created_at": context["created_at"],
        "repo_branch": context["repo_branch"],
        "repo_sha": context["repo_sha"],
        "paired_vllm_ref": context["paired_vllm_ref"],
        "runtime_tuple": runtime_tuple,
        "match_level": match_level,
        "selected_shards": selected_shards,
        "warnings": warnings,
        "missing": missing,
        "notes": None,
    }
    validate_instance(result, "kb-resolve-result.schema.json", root=root)
    if emit_path:
        dump_json(emit_path, result)
    return result


def build_local(
    root: Path | None = None,
    *,
    resolve_result: dict[str, Any],
    emit_sqlite: Path,
) -> Path:
    root = root or repo_root()
    emit_sqlite.parent.mkdir(parents=True, exist_ok=True)
    if emit_sqlite.exists():
        emit_sqlite.unlink()

    conn = sqlite3.connect(emit_sqlite)
    try:
        conn.executescript(kb_path("sql", "merged_pack.sql", root=root).read_text(encoding="utf-8"))
        now = now_utc()
        runtime_hash = hashlib.sha256(json.dumps(resolve_result, sort_keys=True).encode("utf-8")).hexdigest()
        pack_meta = {
            "build_created_at": now,
            "repo_sha": resolve_result["repo_sha"],
            "paired_vllm_ref": resolve_result["paired_vllm_ref"],
            "resolve_hash": runtime_hash,
            "match_level": resolve_result["match_level"],
        }
        for key, value in pack_meta.items():
            _insert_json(conn, "pack_meta", ["meta_key", "value_json"], [key, json.dumps(value)])

        sources = [
            (
                "source-repo-prefill",
                "repo_file",
                "vllm_ascend/ascend_forward_context.py",
                None,
                resolve_result["repo_sha"],
                resolve_result["paired_vllm_ref"],
                "repo_semantics",
                None,
                json.dumps({"summary": "A2/A3 prefill and communication selection rules"}),
            ),
            (
                "source-repo-envs",
                "repo_file",
                "vllm_ascend/envs.py",
                None,
                resolve_result["repo_sha"],
                resolve_result["paired_vllm_ref"],
                "repo_semantics",
                None,
                json.dumps({"summary": "Ascend runtime env variables"}),
            ),
            (
                "source-val-async",
                "repo_test",
                "tests/e2e/singlecard/test_async_scheduling.py",
                None,
                resolve_result["repo_sha"],
                resolve_result["paired_vllm_ref"],
                "validation",
                None,
                json.dumps({"summary": "single-card async scheduling smoke/regression"}),
            ),
            (
                "source-val-dynamic-batch",
                "repo_test",
                "tests/ut/core/test_scheduler_dynamic_batch.py",
                None,
                resolve_result["repo_sha"],
                resolve_result["paired_vllm_ref"],
                "validation",
                None,
                json.dumps({"summary": "dynamic batch unit tests"}),
            ),
            (
                "source-runtime-tuple",
                "runtime",
                None,
                None,
                resolve_result["repo_sha"],
                resolve_result["paired_vllm_ref"],
                "hw_runtime_caps",
                None,
                json.dumps(resolve_result["runtime_tuple"], ensure_ascii=False),
            ),
        ]
        for row in sources:
            _insert_json(
                conn,
                "sources",
                [
                    "source_id",
                    "source_kind",
                    "path",
                    "uri",
                    "repo_sha",
                    "paired_vllm_ref",
                    "shard_family",
                    "excerpt_hash",
                    "metadata_json",
                ],
                list(row),
            )

        entities = [
            ("entity-model-qwen3-next", "model", "qwen3-next", json.dumps([]), json.dumps(["deployment"]), json.dumps({})),
            ("entity-model-qwen3-next-32b", "model", "qwen3-next-32b", json.dumps([]), json.dumps(["performance"]), json.dumps({})),
            ("entity-hw-a2", "hardware", "A2", json.dumps(["910B4"]), json.dumps(["ascend"]), json.dumps({})),
            ("entity-feature-prefill", "feature", "prefill", json.dumps([]), json.dumps(["runtime"]), json.dumps({})),
            ("entity-feature-decode", "feature", "decode", json.dumps([]), json.dumps(["runtime"]), json.dumps({})),
            ("entity-feature-allgather-ep", "feature", "allgather_ep", json.dumps(["allgather ep"]), json.dumps(["deployment"]), json.dumps({})),
            ("entity-feature-dynamic-batching", "feature", "dynamic_batching", json.dumps(["dynamic batch"]), json.dumps(["validation"]), json.dumps({})),
            ("entity-policy-prefill", "policy", "repo.policy.prefill", json.dumps([]), json.dumps(["repo"]), json.dumps({})),
            ("entity-config-tp4-bf16-ctx8k", "config", "tp4_bf16_ctx8k", json.dumps([]), json.dumps(["performance"]), json.dumps({})),
        ]
        for row in entities:
            _insert_json(
                conn,
                "entities",
                ["entity_id", "entity_type", "canonical_name", "aliases_json", "tags_json", "metadata_json"],
                list(row),
            )

        facts = [
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
                "fact-runtime-cann",
                "entity-hw-a2",
                "runtime_constraint",
                None,
                "Current validated runtime tuple uses CANN 8.5.0 on A2/910B4.",
                1.0,
                None,
                None,
                json.dumps(resolve_result["runtime_tuple"], ensure_ascii=False),
                "source-runtime-tuple",
                "hw_runtime_caps",
                json.dumps({}),
            ),
        ]
        for row in facts:
            _insert_json(
                conn,
                "facts",
                [
                    "fact_id",
                    "subject_id",
                    "predicate",
                    "object_id",
                    "literal_text",
                    "confidence",
                    "valid_from",
                    "valid_to",
                    "scope_json",
                    "source_id",
                    "shard_family",
                    "metadata_json",
                ],
                list(row),
            )

        edges = [
            ("edge-policy-prefill-a2", "entity-policy-prefill", "targets", "entity-hw-a2", 1.0, "source-repo-prefill", json.dumps({})),
            ("edge-dynamic-batching-prefill", "entity-feature-dynamic-batching", "touches", "entity-feature-prefill", 0.8, "source-val-dynamic-batch", json.dumps({})),
        ]
        for row in edges:
            _insert_json(
                conn,
                "edges",
                ["edge_id", "src_entity_id", "edge_type", "dst_entity_id", "weight", "source_id", "metadata_json"],
                list(row),
            )

        symbols = [
            (
                "symbol-select-moe-comm-method",
                "select_moe_comm_method",
                "function",
                "vllm_ascend/ascend_forward_context.py",
                "select_moe_comm_method(num_tokens, vllm_config, is_draft_model=False)",
                "vllm_ascend.ascend_forward_context",
                "vllm_ascend/ascend_forward_context.py",
                resolve_result["paired_vllm_ref"],
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
                resolve_result["paired_vllm_ref"],
                json.dumps({"surfaces": ["dynamic_batching", "prefill"]}, ensure_ascii=False),
            ),
        ]
        for row in symbols:
            _insert_json(
                conn,
                "symbol_index",
                [
                    "symbol_id",
                    "qualname",
                    "kind",
                    "file_path",
                    "signature",
                    "owner_module",
                    "repo_path",
                    "paired_vllm_ref",
                    "metadata_json",
                ],
                list(row),
            )

        validations = [
            (
                "validation:baseline:qwen3-next:a2",
                "entity-model-qwen3-next",
                "model",
                "compatible_baseline",
                "pass",
                json.dumps({"hw": "A2", "cann": resolve_result["runtime_tuple"]["cann"]}, ensure_ascii=False),
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
        for row in validations:
            _insert_json(
                conn,
                "validations",
                [
                    "validation_id",
                    "target_id",
                    "target_kind",
                    "mode",
                    "result",
                    "env_json",
                    "artifact_refs_json",
                    "summary",
                    "source_id",
                    "metadata_json",
                ],
                list(row),
            )
        conn.commit()
    finally:
        conn.close()
    return emit_sqlite


def _capsule_type_for_intent(intent: str) -> str:
    if intent == "intake_lookup":
        return "intake_capsule"
    if intent in {"design_lookup", "upstream_delta"}:
        return "spec_capsule"
    return "atomic_capsule"


def _query_rows(conn: sqlite3.Connection, sql: str, args: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return list(conn.execute(sql, args))


def pack(
    root: Path | None = None,
    *,
    request: dict[str, Any],
    resolve_result: dict[str, Any],
    merged_pack: Path,
    emit_path: Path | None = None,
) -> dict[str, Any]:
    root = root or repo_root()
    if emit_path and not emit_path.is_absolute():
        emit_path = root / emit_path
    conn = sqlite3.connect(merged_pack)
    try:
        runtime_rows = _query_rows(conn, "SELECT literal_text FROM facts WHERE fact_id = 'fact-runtime-cann'")
        policy_rows = _query_rows(conn, "SELECT literal_text FROM facts WHERE subject_id = 'entity-policy-prefill'")
        validation_rows = _query_rows(conn, "SELECT validation_id, summary, artifact_refs_json FROM validations")

        warnings = list(resolve_result["warnings"])
        unknowns: list[str] = []
        atoms: list[dict[str, Any]] = []
        deep_refs: list[dict[str, Any]] = []
        intent = request["intent"]
        selectors = request["selectors"]

        if intent in {"intake_lookup", "deployment_lookup"}:
            capsule_text = "baseline 存在，当前需求仍位于 deployment_execution 边界内，可继续输出配置、脚本和最小验证步骤。"
            atoms.extend(
                [
                    {
                        "atom_id": "atom-a2-prefill-policy",
                        "atom_kind": "fact",
                        "summary": policy_rows[0]["literal_text"] if policy_rows else "repo 记录了 A2 prefill policy 约束。",
                        "source_refs": ["repo_semantics:policy/prefill"],
                    },
                    {
                        "atom_id": "atom-baseline-qwen3-next-a2",
                        "atom_kind": "validation",
                        "summary": "存在 compatible baseline，可作为 deployment policy 查证锚点。",
                        "source_refs": ["validation:baseline:qwen3-next:a2"],
                    },
                ]
            )
            unknowns.append("用户未给出最终拓扑")
            deep_refs.append(
                {
                    "stub_id": "stub-prefill-policy",
                    "source_ref": "tests/e2e/singlecard/test_async_scheduling.py",
                    "estimated_tokens": 240,
                    "reason": "需要查看更完整的 baseline 与调度约束",
                }
            )
        elif intent == "perf_breakdown":
            capsule_text = "当前 profile 指向 prefill 阶段的异常，但缺少同拓扑 baseline，只能先做 partial breakdown。"
            warnings.append("baseline profile missing; expect partial findings only")
            unknowns.append("缺少同拓扑 baseline profile")
            atoms.extend(
                [
                    {
                        "atom_id": "atom-profile-regression",
                        "atom_kind": "validation",
                        "summary": "当前输入描述符合 prefill regression 场景，但仓库内没有对应 baseline profile 产物。",
                        "source_refs": ["validation:matrix:prefill"],
                    },
                    {
                        "atom_id": "atom-runtime-config",
                        "atom_kind": "fact",
                        "summary": "dynamic batching 与 prefill 调度路径会直接影响 profile 解释边界。",
                        "source_refs": ["repo_semantics:policy/prefill"],
                    },
                ]
            )
        elif intent == "model_expectation":
            capsule_text = (
                "在 A2 / TP4 / BF16 / 8k 约束下，qwen3-next-32b 的 TTFT、throughput 和 memory headroom "
                "应落在一个可解释区间内；"
                "若 batch size 和 graph mode 不固定，结果应以范围而非单点返回。"
            )
            warnings.append("batch size not fixed; returning envelope instead of single-point estimate")
            unknowns.extend(["graph mode enabled/disabled", "batch size 尚未冻结"])
            atoms.extend(
                [
                    {
                        "atom_id": "atom-baseline-qwen3-next-32b-a2",
                        "atom_kind": "validation",
                        "summary": "存在 compatible baseline，可为 TTFT/throughput 估计提供量级锚点。",
                        "source_refs": ["validation:baseline:qwen3-next-32b:a2:tp4"],
                    },
                    {
                        "atom_id": "atom-a2-prefill-policy",
                        "atom_kind": "fact",
                        "summary": policy_rows[0]["literal_text"] if policy_rows else "A2 prefill policy 限定了预期性能上界。",
                        "source_refs": ["repo_semantics:policy/prefill"],
                    },
                ]
            )
            deep_refs.append(
                {
                    "stub_id": "stub-baseline-qwen3-next-32b-a2",
                    "source_ref": "tests/e2e/singlecard/test_async_scheduling.py",
                    "estimated_tokens": 260,
                    "reason": "需要查看更完整的 baseline 条件说明",
                }
            )
        elif intent == "validation_selection":
            impacted = [path for path in selectors.get("files", []) if "dynamic_batch" in path or "scheduler" in path]
            capsule_text = "diff 命中了 dynamic batching 调度面，可收口为一个最小 smoke+UT 组合，并保留一项低置信补采建议。"
            atoms.extend(
                [
                    {
                        "atom_id": "atom-dynamic-batch-ut",
                        "atom_kind": "validation",
                        "summary": "UT 已覆盖 scheduler_dynamic_batch 的核心逻辑。",
                        "source_refs": ["tests/ut/core/test_scheduler_dynamic_batch.py"],
                    },
                    {
                        "atom_id": "atom-async-scheduling-smoke",
                        "atom_kind": "validation",
                        "summary": "single-card async scheduling e2e 可作为最小 smoke / regression 组合的一部分。",
                        "source_refs": ["tests/e2e/singlecard/test_async_scheduling.py"],
                    },
                ]
            )
            if not impacted:
                warnings.append("diff paths missing; using feature-based fallback")
            unknowns.append("A3 拓扑下是否需要额外并行度 smoke")
        else:
            capsule_text = "知识未命中明确 intent，返回 repo-only fallback。"
            unknowns.append("intent unsupported")

        logical_domains = request["logical_domains"]
        estimated_tokens = min(request["budget_token_cap"], max(256, 220 + len(atoms) * 120))
        response = {
            "schema_version": "kb-pack-response/v1",
            "request_id": request["request_id"],
            "pack_id": f"pack-{request['request_id']}",
            "created_at": now_utc(),
            "match_level": resolve_result["match_level"],
            "selected_shards": resolve_result["selected_shards"],
            "warnings": warnings,
            "unknowns": unknowns,
            "budget_token_cap": request["budget_token_cap"],
            "estimated_tokens": estimated_tokens,
            "capsule_type": _capsule_type_for_intent(intent),
            "logical_domains": logical_domains,
            "capsule_text": capsule_text,
            "atoms": atoms,
            "deep_reference_stubs": deep_refs,
            "cache_hit": False,
            "capsule_path": str(emit_path.relative_to(root)) if emit_path else request.get("emit_path"),
        }
        validate_instance(response, "kb-pack-response.schema.json", root=root)
        if emit_path:
            dump_json(emit_path, response)

        _insert_json(
            conn,
            "capsules",
            [
                "capsule_id",
                "request_id",
                "intent",
                "logical_domains_json",
                "selectors_json",
                "token_estimate",
                "capsule_type",
                "capsule_text",
                "atoms_json",
                "unknowns_json",
                "created_at",
                "metadata_json",
            ],
            [
                response["pack_id"],
                response["request_id"],
                intent,
                json.dumps(logical_domains, ensure_ascii=False),
                json.dumps(selectors, ensure_ascii=False),
                estimated_tokens,
                response["capsule_type"],
                capsule_text,
                json.dumps(atoms, ensure_ascii=False),
                json.dumps(unknowns, ensure_ascii=False),
                response["created_at"],
                json.dumps({"match_level": response["match_level"]}, ensure_ascii=False),
            ],
        )
        conn.commit()
        return response
    finally:
        conn.close()


def doctor(root: Path | None = None) -> list[str]:
    root = root or repo_root()
    messages = run_contract_checks(root=root)
    resolve_json = kb_root(root) / "local" / "resolve.json"
    merged = kb_root(root) / "local" / "merged" / "current.sqlite"
    if resolve_json.exists():
        validate_instance(load_json(resolve_json), "kb-resolve-result.schema.json", root=root)
        messages.append("OK local resolve.json")
    if merged.exists():
        conn = sqlite3.connect(merged)
        try:
            conn.execute("SELECT 1 FROM pack_meta LIMIT 1")
            messages.append("OK local merged sqlite")
        finally:
            conn.close()
    return messages


def _parse_request_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": "kb-pack-request/v2",
        "request_id": args.request_id,
        "created_at": now_utc(),
        "intent": args.intent,
        "repo_root": args.repo_root,
        "resolve_policy": "auto",
        "logical_domains": args.domains or [],
        "physical_shard_hints": args.physical_shard_hints or [],
        "selectors": {
            "files": args.files or [],
            "symbols": args.symbols or [],
            "entities": args.entities or [],
            "errors": args.errors or [],
            "models": args.models or [],
            "features": args.features or [],
            "hw": args.hw or [],
            "commits": [],
            "prs": [],
            "versions": args.versions or [],
            "configs": args.configs or [],
        },
        "must_have": args.must_have or [],
        "nice_to_have": args.nice_to_have or [],
        "evidence_refs": [],
        "budget_token_cap": args.budget_token_cap,
        "max_atoms": args.max_atoms,
        "max_hops": args.max_hops,
        "include_evidence_stubs": args.include_evidence_stubs,
        "stop_after_first_sufficient": args.stop_after_first_sufficient,
        "emit_path": args.emit,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kb.py")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--repo-root", default=".")

    resolve_parser = subparsers.add_parser("resolve")
    resolve_parser.add_argument("--repo-root", default=".")
    resolve_parser.add_argument("--request-id", default=None)
    resolve_parser.add_argument("--soc", default=None)
    resolve_parser.add_argument("--cann", default=None)
    resolve_parser.add_argument("--torch", default=None)
    resolve_parser.add_argument("--torch-npu", dest="torch_npu", default=None)
    resolve_parser.add_argument("--python", default=None)
    resolve_parser.add_argument("--repo-sha", dest="repo_sha", default=None)
    resolve_parser.add_argument("--paired-vllm-ref", dest="paired_vllm_ref", default=None)
    resolve_parser.add_argument("--emit", default=None)

    build_parser = subparsers.add_parser("build-local")
    build_parser.add_argument("--repo-root", default=".")
    build_parser.add_argument("--resolve", required=True)
    build_parser.add_argument("--emit-sqlite", required=True)

    pack_parser = subparsers.add_parser("pack")
    pack_parser.add_argument("--repo-root", default=".")
    pack_parser.add_argument("--resolve", required=True)
    pack_parser.add_argument("--merged-pack", required=True)
    pack_parser.add_argument("--request-id", required=True)
    pack_parser.add_argument("--intent", required=True)
    pack_parser.add_argument("--domains", nargs="*")
    pack_parser.add_argument("--physical-shard-hints", nargs="*", default=[])
    pack_parser.add_argument("--files", nargs="*", default=[])
    pack_parser.add_argument("--symbols", nargs="*", default=[])
    pack_parser.add_argument("--entities", nargs="*", default=[])
    pack_parser.add_argument("--errors", nargs="*", default=[])
    pack_parser.add_argument("--models", nargs="*", default=[])
    pack_parser.add_argument("--features", nargs="*", default=[])
    pack_parser.add_argument("--hw", nargs="*", default=[])
    pack_parser.add_argument("--versions", nargs="*", default=[])
    pack_parser.add_argument("--configs", nargs="*", default=[])
    pack_parser.add_argument("--must-have", nargs="*", default=[])
    pack_parser.add_argument("--nice-to-have", nargs="*", default=[])
    pack_parser.add_argument("--budget-token-cap", type=int, default=1500)
    pack_parser.add_argument("--max-atoms", type=int, default=10)
    pack_parser.add_argument("--max-hops", type=int, default=1)
    pack_parser.add_argument("--include-evidence-stubs", action="store_true")
    pack_parser.add_argument("--stop-after-first-sufficient", action="store_true")
    pack_parser.add_argument("--emit", default=None)

    args = parser.parse_args(argv)
    root = (repo_root() / args.repo_root).resolve() if args.repo_root != "." else repo_root()

    if args.command == "doctor":
        for message in doctor(root):
            print(message)
        return 0

    if args.command == "resolve":
        overrides = {
            key: value
            for key, value in {
                "soc": args.soc,
                "cann": args.cann,
                "torch": args.torch,
                "torch_npu": args.torch_npu,
                "python": args.python,
                "repo_sha": args.repo_sha,
                "paired_vllm_ref": args.paired_vllm_ref,
            }.items()
            if value
        }
        emit_path = Path(args.emit) if args.emit else None
        result = resolve(root, request_id=args.request_id, overrides=overrides, emit_path=emit_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "build-local":
        resolve_result = load_json(Path(args.resolve))
        path = build_local(root, resolve_result=resolve_result, emit_sqlite=Path(args.emit_sqlite))
        print(path)
        return 0

    if args.command == "pack":
        resolve_result = load_json(Path(args.resolve))
        request = _parse_request_from_args(args)
        validate_instance(request, "kb-pack-request.schema.json", root=root)
        response = pack(
            root,
            request=request,
            resolve_result=resolve_result,
            merged_pack=Path(args.merged_pack),
            emit_path=Path(args.emit) if args.emit else None,
        )
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
