from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

from .contracts import (ContractError, dump_json, kb_path, load_json, now_utc,
                        run_contract_checks, validate_instance)
from .detector import collect_runtime_context
from .extractors import (extract_cann_op_constraints,
                         extract_hw_soc_detail, extract_minimal_validation,
                         extract_repo_custom_ops, extract_repo_semantics,
                         extract_runtime_caps, extract_torch_npu_bindings,
                         extract_vllm_release_delta, extract_vllm_semantics,
                         extract_vllm_symbols, merge_shard_rows)
from .paths import kb_root, repo_root
from .shadow_adapter import build_shadow_diagnostics
from .strategy import (baselines_from_rows, build_artifact_atom,
                       build_strategy_atom, select_artifact_path,
                       select_deployment_strategy,
                       selector_context_from_selectors,
                       topology_multiplier_from_rows)

TABLE_COLUMNS: dict[str, list[str]] = {
    "sources": [
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
    "entities": ["entity_id", "entity_type", "canonical_name", "aliases_json", "tags_json", "metadata_json"],
    "facts": [
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
    "edges": ["edge_id", "src_entity_id", "edge_type", "dst_entity_id", "weight", "source_id", "metadata_json"],
    "symbol_index": [
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
    "validations": [
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
}


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


def _git_head(root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _rule_matches(expected: str | None, actual: str, *, self_value: str | None = None) -> bool:
    if expected is None:
        return actual != "unknown"
    if expected == "@self":
        return self_value is not None and actual == self_value
    return actual == expected


def _prefix_matches(expected_prefix: str | None, actual: str) -> bool:
    if expected_prefix is None:
        return actual != "unknown"
    return actual != "unknown" and actual.startswith(expected_prefix)


def _exact_rule_matches(rule: dict[str, Any], context: dict[str, Any], *, self_repo_sha: str) -> bool:
    runtime_tuple = context["runtime_tuple"]
    return all(
        [
            _rule_matches(rule.get("soc"), runtime_tuple["soc"]),
            _rule_matches(rule.get("cann"), runtime_tuple["cann"]),
            _rule_matches(rule.get("repo_sha"), context["repo_sha"], self_value=self_repo_sha),
            _rule_matches(rule.get("paired_vllm_ref"), context["paired_vllm_ref"]),
            _prefix_matches(rule.get("python_prefix"), runtime_tuple["python"]),
            _prefix_matches(rule.get("torch_prefix"), runtime_tuple["torch"]),
            _prefix_matches(rule.get("torch_npu_prefix"), runtime_tuple["torch_npu"]),
        ]
    )


def _rows_for_selected_shards(root: Path, resolve_result: dict[str, Any]) -> dict[str, list[tuple[Any, ...]]]:
    selected = set(resolve_result["selected_shards"])
    shards = []
    if "repo_semantics" in selected:
        shards.append(extract_repo_semantics(root=root, resolve_result=resolve_result))
    if "repo_custom_ops" in selected:
        shards.append(extract_repo_custom_ops(root=root, resolve_result=resolve_result))
    if "validation" in selected:
        shards.append(extract_minimal_validation(root=root, resolve_result=resolve_result))
    if "hw_runtime_caps" in selected:
        shards.append(extract_runtime_caps(resolve_result))
    if "vllm_semantics" in selected:
        shards.append(extract_vllm_semantics(root=root, resolve_result=resolve_result))
    if "vllm_symbols" in selected:
        shards.append(extract_vllm_symbols(root=root, resolve_result=resolve_result))
    if "vllm_release_delta" in selected:
        shards.append(extract_vllm_release_delta(root=root, resolve_result=resolve_result))
    if "hw_soc_detail" in selected:
        shards.append(extract_hw_soc_detail(resolve_result))
    if "cann_op_constraints" in selected:
        shards.append(extract_cann_op_constraints(resolve_result))
    if "torch_npu_bindings" in selected:
        shards.append(extract_torch_npu_bindings(resolve_result))
    return merge_shard_rows(*shards)


def _pair_shards_available(root: Path, context: dict[str, Any]) -> bool:
    return context["paired_vllm_ref"] != "unknown" and (root.parent / "vllm").exists()


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
    self_repo_sha = _git_head(root) or context["repo_sha"]

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
        if _exact_rule_matches(rule, context, self_repo_sha=self_repo_sha):
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
        match_level = "compatible" if runtime_tuple["soc"] in {"A2", "A3"} and runtime_tuple["cann"] != "unknown" else "unknown"
        if match_level == "compatible":
            warnings.append("runtime tuple exact match unavailable; using repo-only fallback")
    if _pair_shards_available(root, context):
        selected_shards = sorted(set(selected_shards + ["vllm_semantics", "vllm_symbols", "vllm_release_delta"]))
    if all(runtime_tuple.get(key) != "unknown" for key in ["soc", "cann", "torch_npu"]):
        selected_shards = sorted(
            set(selected_shards + ["hw_soc_detail", "hw_runtime_caps", "cann_op_constraints", "torch_npu_bindings"])
        )
    if match_level == "exact" and runtime_tuple["soc"] == "A2" and runtime_tuple["cann"] == "8.5.0":
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
        now = resolve_result["created_at"]
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
        extracted_rows = _rows_for_selected_shards(root, resolve_result)
        for table_name, columns in TABLE_COLUMNS.items():
            for row in extracted_rows[table_name]:
                _insert_json(conn, table_name, columns, list(row))
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


def _estimate_tokens(atoms: list[dict[str, Any]], deep_refs: list[dict[str, Any]]) -> int:
    return max(220, 220 + len(atoms) * 120 + len(deep_refs) * 80)


def _apply_budget(
    *,
    budget_token_cap: int,
    atoms: list[dict[str, Any]],
    deep_refs: list[dict[str, Any]],
    warnings: list[str],
    unknowns: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    trimmed_atoms = list(atoms)
    trimmed_refs = list(deep_refs)
    while trimmed_refs and _estimate_tokens(trimmed_atoms, trimmed_refs) > budget_token_cap:
        trimmed_refs.pop()
    while len(trimmed_atoms) > 1 and _estimate_tokens(trimmed_atoms, trimmed_refs) > budget_token_cap:
        trimmed_atoms.pop()
    estimated = _estimate_tokens(trimmed_atoms, trimmed_refs)
    if estimated > budget_token_cap:
        warnings.append("budget cap too small; returning explicit miss")
        unknowns.append("budget cap prevented a sufficient capsule")
        return [], [], min(budget_token_cap, 220)
    if len(trimmed_atoms) != len(atoms) or len(trimmed_refs) != len(deep_refs):
        warnings.append("budget cap forced a smaller capsule")
    return trimmed_atoms, trimmed_refs, estimated


def _entity_name_map(conn: sqlite3.Connection) -> dict[str, str]:
    rows = _query_rows(conn, "SELECT entity_id, canonical_name FROM entities")
    return {row["entity_id"]: row["canonical_name"] for row in rows}


def _strategy_validation_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return _query_rows(
        conn,
        "SELECT validation_id, target_id, mode, result, env_json, artifact_refs_json, summary, source_id, metadata_json "
        "FROM validations WHERE result = 'pass' ORDER BY validation_id",
    )


def _strategy_fact_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return _query_rows(
        conn,
        "SELECT fact_id, subject_id, predicate, object_id, literal_text, source_id, scope_json, metadata_json "
        "FROM facts WHERE predicate IN "
        "('topology_mapping', 'deployment_baseline', 'performance_baseline', 'model_traits', "
        "'artifact_path', 'tool_recipe', 'runtime_constraint', 'communication_profile') "
        "ORDER BY fact_id",
    )


def _has_rich_strategy_data(matches: list[Any]) -> bool:
    return any(
        baseline.render_preset
        or baseline.physical_cards is not None
        or baseline.logical_npus is not None
        or baseline.tensor_parallel is not None
        or baseline.model_traits
        for baseline in matches
    )


def _model_label(model_base: str | None, traits: tuple[str, ...]) -> str:
    if model_base == "qwen3-32b":
        return "Qwen3-32B-W8A8" if "quant_w8a8" in traits else "Qwen3-32B"
    if model_base == "deepseek-v3":
        return "DeepSeek-V3"
    return model_base or "当前模型"


def _matching_strategy_facts(
    fact_rows: list[sqlite3.Row],
    entity_names: dict[str, str],
    *,
    context: Any,
    predicate: str,
) -> list[sqlite3.Row]:
    matches: list[sqlite3.Row] = []
    for row in fact_rows:
        if row["predicate"] != predicate:
            continue
        metadata = json.loads(row["metadata_json"] or "{}")
        scope = json.loads(row["scope_json"] or "{}")
        subject_name = entity_names.get(row["subject_id"])
        model_base = metadata.get("model_base")
        hw_values = metadata.get("hw") or metadata.get("soc") or scope.get("hw")
        hw_matches = False
        if isinstance(hw_values, list):
            hw_matches = context.hw in hw_values
        elif isinstance(hw_values, str):
            hw_matches = hw_values == context.hw
        elif hw_values is None:
            hw_matches = True
        if predicate in {"topology_mapping", "communication_profile"}:
            if hw_matches:
                matches.append(row)
            continue
        normalized_subject = subject_name.removesuffix("-w8a8") if isinstance(subject_name, str) else subject_name
        model_matches = model_base == context.model_base or normalized_subject == context.model_base
        if predicate == "runtime_constraint":
            if hw_matches and (model_matches or model_base is None):
                matches.append(row)
            continue
        if model_matches and hw_matches:
            matches.append(row)
    return matches


def _strategy_deep_refs(
    *,
    request_id: str,
    refs: Iterable[str],
    reason: str,
) -> list[dict[str, Any]]:
    stubs: list[dict[str, Any]] = []
    for index, ref in enumerate(dict.fromkeys(refs)):
        stubs.append(
            {
                "stub_id": f"stub-{request_id}-{index}",
                "source_ref": ref,
                "estimated_tokens": 220 if ref.endswith(".yaml") else 260,
                "reason": reason,
            }
        )
        if len(stubs) >= 2:
            break
    return stubs


def _strategy_topology_label(candidate: Any) -> str:
    parts: list[str] = []
    if getattr(candidate, "tensor_parallel", None):
        parts.append(f"TP{candidate.tensor_parallel}")
    if getattr(candidate, "data_parallel", None) and candidate.data_parallel not in {None, 1}:
        parts.append(f"DP{candidate.data_parallel}")
    if getattr(candidate, "expert_parallel", None) and candidate.expert_parallel not in {None, 1}:
        parts.append(f"EP{candidate.expert_parallel}")
    if getattr(candidate, "physical_cards", None):
        parts.append(f"{candidate.physical_cards} cards")
    if getattr(candidate, "logical_npus", None):
        parts.append(f"{candidate.logical_npus} logical NPUs")
    return " / ".join(parts) if parts else "未冻结拓扑"


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
        policy_rows = _query_rows(conn, "SELECT literal_text FROM facts WHERE subject_id = 'entity-policy-prefill'")
        vllm_semantic_rows = _query_rows(
            conn,
            "SELECT fact_id, literal_text, source_id FROM facts WHERE shard_family = 'vllm_semantics' ORDER BY fact_id",
        )
        vllm_delta_rows = _query_rows(
            conn,
            "SELECT fact_id, literal_text, metadata_json, source_id FROM facts WHERE shard_family = 'vllm_release_delta' ORDER BY fact_id",
        )
        vllm_symbol_rows = _query_rows(
            conn,
            "SELECT qualname, file_path FROM symbol_index WHERE repo_path LIKE 'vllm/%' ORDER BY qualname",
        )
        entity_names = _entity_name_map(conn)
        strategy_validation_rows = _strategy_validation_rows(conn)
        strategy_fact_rows = _strategy_fact_rows(conn)
        substrate_rows = _query_rows(
            conn,
            "SELECT fact_id, literal_text, shard_family FROM facts WHERE shard_family IN ('hw_soc_detail', 'hw_runtime_caps', 'cann_op_constraints', 'torch_npu_bindings') ORDER BY fact_id",
        )
        selectors = request["selectors"]
        selector_context = selector_context_from_selectors(selectors, resolve_result["runtime_tuple"].get("soc", "unknown"))
        primary_model = selector_context.model_base or (selectors.get("models", [None])[0] if selectors.get("models") else None)
        primary_hw = selector_context.hw or resolve_result["runtime_tuple"].get("soc", "unknown")
        requested_card_count = selector_context.physical_cards
        requested_logical_npus = selector_context.logical_npus
        strategy_baselines = baselines_from_rows(strategy_validation_rows, entity_names)
        matching_strategy_baselines = [
            baseline
            for baseline in strategy_baselines
            if baseline.model_base == selector_context.model_base and baseline.hw == selector_context.hw
        ]
        topology_fact_rows = _matching_strategy_facts(
            strategy_fact_rows,
            entity_names,
            context=selector_context,
            predicate="topology_mapping",
        )
        trait_fact_rows = _matching_strategy_facts(
            strategy_fact_rows,
            entity_names,
            context=selector_context,
            predicate="model_traits",
        )
        perf_fact_rows = _matching_strategy_facts(
            strategy_fact_rows,
            entity_names,
            context=selector_context,
            predicate="performance_baseline",
        )
        artifact_fact_rows = _matching_strategy_facts(
            strategy_fact_rows,
            entity_names,
            context=selector_context,
            predicate="artifact_path",
        )
        tool_recipe_rows = _matching_strategy_facts(
            strategy_fact_rows,
            entity_names,
            context=selector_context,
            predicate="tool_recipe",
        )
        runtime_constraint_rows = _matching_strategy_facts(
            strategy_fact_rows,
            entity_names,
            context=selector_context,
            predicate="runtime_constraint",
        )
        comm_profile_rows = _matching_strategy_facts(
            strategy_fact_rows,
            entity_names,
            context=selector_context,
            predicate="communication_profile",
        )
        topology_multiplier = topology_multiplier_from_rows(topology_fact_rows, hw=selector_context.hw)
        strategy_selection = None
        if _has_rich_strategy_data(matching_strategy_baselines):
            strategy_selection = select_deployment_strategy(
                selector_context,
                tuple(matching_strategy_baselines),
                topology_multiplier=topology_multiplier,
            )
        artifact_selection = select_artifact_path(
            selector_context,
            tuple(matching_strategy_baselines),
            artifact_fact_rows=artifact_fact_rows,
            tool_recipe_rows=tool_recipe_rows,
            runtime_constraint_rows=runtime_constraint_rows,
        )

        warnings = list(resolve_result["warnings"])
        unknowns: list[str] = []
        atoms: list[dict[str, Any]] = []
        deep_refs: list[dict[str, Any]] = []
        intent = request["intent"]
        evidence_refs = request["evidence_refs"]
        shadow_diagnostics = None
        if strategy_selection is not None or artifact_selection is not None:
            shadow_diagnostics = build_shadow_diagnostics(
                root=root,
                request_id=request["request_id"],
                context=selector_context,
                strategy_selection=strategy_selection,
                artifact_selection=artifact_selection,
            )

        def append_shadow_diagnostics() -> None:
            if shadow_diagnostics is None:
                return
            warnings.extend(shadow_diagnostics["warnings"])
            unknowns.extend(shadow_diagnostics["unknowns"])
            # Shadow diagnostics are part of the handoff acceptance surface.
            # Keep them ahead of lower-priority atoms so budget trimming does
            # not silently erase the shadow-only wiring signal.
            atoms[0:0] = shadow_diagnostics["atoms"]

        if intent in {"intake_lookup", "deployment_lookup"}:
            if strategy_selection is not None or artifact_selection is not None:
                selected = strategy_selection.selected if strategy_selection is not None else None
                documented = strategy_selection.documented if strategy_selection is not None else None
                selected_artifact = artifact_selection.selected if artifact_selection is not None else None
                documented_artifact = artifact_selection.documented if artifact_selection is not None else None
                model_base = (
                    selected.model_base
                    if selected is not None
                    else selected_artifact.model_base
                    if selected_artifact is not None
                    else primary_model
                )
                model_traits = (
                    selected.model_traits
                    if selected is not None
                    else selected_artifact.model_traits
                    if selected_artifact is not None
                    else tuple()
                )
                model_label = _model_label(model_base, model_traits)
                if strategy_selection is not None:
                    warnings.extend(strategy_selection.warnings)
                    unknowns.extend(strategy_selection.unknowns)
                    atoms.append(build_strategy_atom("selected", selected))
                    if documented is not None:
                        atoms.append(build_strategy_atom("documented", documented))
                        atoms.append(
                            {
                                "atom_id": f"atom-{selected.model_base or 'model'}-documented-validation",
                                "atom_kind": "validation",
                                "summary": documented.summary,
                                "source_refs": list(documented.source_refs),
                            }
                        )
                    for alternative in strategy_selection.alternatives:
                        atoms.append(build_strategy_atom("alternative", alternative))
                if artifact_selection is not None:
                    warnings.extend(artifact_selection.warnings)
                    unknowns.extend(artifact_selection.unknowns)
                    atoms.append(build_artifact_atom("selected", selected_artifact))
                    if documented_artifact is not None:
                        atoms.append(build_artifact_atom("documented", documented_artifact))
                    for alternative in artifact_selection.alternatives:
                        atoms.append(build_artifact_atom("alternative", alternative))
                if topology_fact_rows:
                    atoms.append(
                        {
                            "atom_id": f"atom-topology-{primary_hw.lower()}",
                            "atom_kind": "fact",
                            "summary": topology_fact_rows[0]["literal_text"],
                            "source_refs": [topology_fact_rows[0]["fact_id"]],
                        }
                    )
                if comm_profile_rows:
                    atoms.append(
                        {
                            "atom_id": f"atom-comm-profile-{primary_hw.lower()}",
                            "atom_kind": "fact",
                            "summary": comm_profile_rows[0]["literal_text"],
                            "source_refs": [comm_profile_rows[0]["fact_id"]],
                        }
                    )
                if trait_fact_rows:
                    atoms.append(
                        {
                            "atom_id": f"atom-model-traits-{model_base or 'unknown'}",
                            "atom_kind": "fact",
                            "summary": trait_fact_rows[0]["literal_text"],
                            "source_refs": [trait_fact_rows[0]["fact_id"]],
                        }
                    )
                if tool_recipe_rows:
                    atoms.append(
                        {
                            "atom_id": f"atom-tool-recipe-{model_base or 'unknown'}",
                            "atom_kind": "fact",
                            "summary": tool_recipe_rows[0]["literal_text"],
                            "source_refs": [tool_recipe_rows[0]["fact_id"]],
                        }
                    )
                if runtime_constraint_rows:
                    atoms.append(
                        {
                            "atom_id": f"atom-runtime-constraint-{model_base or 'unknown'}",
                            "atom_kind": "constraint",
                            "summary": runtime_constraint_rows[0]["literal_text"],
                            "source_refs": [runtime_constraint_rows[0]["fact_id"]],
                        }
                    )
                elif substrate_rows:
                    atoms.append(
                        {
                            "atom_id": "atom-runtime-constraint",
                            "atom_kind": "fact",
                            "summary": substrate_rows[0]["literal_text"],
                            "source_refs": [f"{substrate_rows[0]['shard_family']}:runtime"],
                        }
                    )
                append_shadow_diagnostics()
                if selected_artifact is not None and selected_artifact.decision_kind == "unsupported_requires_choice":
                    capsule_text = (
                        f"{model_label} 在 {primary_hw} 上收到 native FP8 直跑请求，但该路径当前不受支持；"
                        "需要先在 ModelSlim 转换后部署 与 fp8-origin 适配之间收口路线。"
                    )
                elif selected_artifact is not None and selected_artifact.decision_kind.endswith("convert_then_deploy"):
                    if selected is not None and selected.decision_kind == "inferred_preserve_topology":
                        topology_label = "single-card" if requested_card_count == 1 else f"{requested_card_count} cards"
                        capsule_text = (
                            f"{model_label} 在 {primary_hw} 上已选择 conversion + deployment 路线；"
                            f"当前为保持 {topology_label} 物理拓扑，返回推断的未验证策略 {_strategy_topology_label(selected)}，"
                            "并同时保留 conversion runbook。"
                        )
                    else:
                        selected_label = _strategy_topology_label(selected) if selected is not None else "当前拓扑待运行时补齐"
                        capsule_text = (
                            f"{model_label} 在 {primary_hw} 上已选择 conversion + deployment 路线；"
                            f"后续应按 {selected_label} 输出 conversion + serve 两阶段 runbook。"
                        )
                elif selected is not None and selected.decision_kind == "best_perf_default":
                    capsule_text = (
                        f"{model_label} 在 {primary_hw} 上未锁定拓扑；默认返回文档化最佳性能基线 "
                        f"{_strategy_topology_label(selected)}。"
                    )
                elif selected is not None and selected.decision_kind == "documented_baseline":
                    capsule_text = (
                        f"{model_label} 在 {primary_hw} 上命中文档化部署基线 "
                        f"{_strategy_topology_label(selected)}，可以直接生成推荐脚本。"
                    )
                elif selected is not None and selected.decision_kind == "inferred_preserve_topology":
                    topology_label = "single-card" if requested_card_count == 1 else f"{requested_card_count} cards"
                    capsule_text = (
                        f"{model_label} 在 {primary_hw} 上收到 {topology_label} 请求；"
                        f"当前文档化基线是 {_strategy_topology_label(documented)}。"
                        f"为保持用户请求的物理拓扑，返回推断的未验证策略 {_strategy_topology_label(selected)}，"
                        "并显式暴露风险。"
                    )
                else:
                    alternative_labels = ", ".join(_strategy_topology_label(candidate) for candidate in (strategy_selection.alternatives if strategy_selection else ())) or "无稳定候选"
                    capsule_text = (
                        f"{model_label} 在 {primary_hw} 上的当前请求无法稳定收敛到单个 deployment artifact；"
                        f"文档化基线是 {_strategy_topology_label(documented)}，但锁定拓扑下仍存在 {alternative_labels} 等候选。"
                        "当前应该显式保留 unknown，并允许 reroute 到 design_analysis。"
                    )
                refs: tuple[str, ...] | list[str] = ()
                if selected_artifact is not None and selected_artifact.artifact_refs:
                    refs = selected_artifact.artifact_refs
                elif selected is not None and selected.artifact_refs:
                    refs = selected.artifact_refs
                elif documented is not None:
                    refs = documented.artifact_refs
                deep_refs.extend(
                    _strategy_deep_refs(
                        request_id=request["request_id"],
                        refs=refs,
                        reason="需要查看文档化基线、conversion 说明或候选策略的完整脚本与配置",
                    )
                )
            else:
                capsule_text = (
                    f"已找到 {primary_model or '当前模型'} 在 {primary_hw or '当前硬件'} 上的 deployment 边界信息，"
                    "可继续输出配置、脚本和最小验证步骤。"
                )
                atoms.extend(
                    [
                        {
                            "atom_id": "atom-runtime-policy",
                            "atom_kind": "fact",
                            "summary": policy_rows[0]["literal_text"] if policy_rows else "repo 记录了 deployment policy 约束。",
                            "source_refs": ["repo_semantics:policy/prefill"],
                        },
                        {
                            "atom_id": "atom-runtime-baseline",
                            "atom_kind": "validation",
                            "summary": "存在可比较 baseline，可作为 deployment policy 查证锚点。",
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
            comparative_requested = len(evidence_refs) >= 2 or any(
                re_term in " ".join(request["must_have"] + request["nice_to_have"]).lower()
                for re_term in ["baseline", "compare", "versus", "对照", "current"]
            )
            if comparative_requested:
                capsule_text = "baseline/current profile 已形成对照条件，可输出 comparative breakdown，并保留少量配置敏感项。"
                atoms.extend(
                    [
                        {
                            "atom_id": "atom-profile-current",
                            "atom_kind": "validation",
                            "summary": "当前 profile 指向 prefill 阶段，热点集中在调度与通信交界面。",
                            "source_refs": ["validation:matrix:prefill"],
                        },
                        {
                            "atom_id": "atom-profile-baseline",
                            "atom_kind": "validation",
                            "summary": "baseline/current 对照已满足 comparative breakdown 的最小证据门槛。",
                            "source_refs": ["validation:baseline:qwen3-next-32b:a2:tp4"],
                        },
                    ]
                )
                unknowns.append("graph mode 与 capture 边界仍可能影响局部归因")
            else:
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
            if strategy_selection is not None or artifact_selection is not None:
                selected = strategy_selection.selected if strategy_selection is not None else None
                documented = strategy_selection.documented if strategy_selection is not None else None
                selected_artifact = artifact_selection.selected if artifact_selection is not None else None
                documented_artifact = artifact_selection.documented if artifact_selection is not None else None
                model_base = (
                    selected.model_base
                    if selected is not None
                    else selected_artifact.model_base
                    if selected_artifact is not None
                    else primary_model
                )
                model_traits = (
                    selected.model_traits
                    if selected is not None
                    else selected_artifact.model_traits
                    if selected_artifact is not None
                    else tuple()
                )
                model_label = _model_label(model_base, model_traits)
                warnings.append("returning an expected envelope rather than a measured single point")
                unknowns.extend(["graph mode enabled/disabled", "batch size 尚未冻结"])
                if artifact_selection is not None:
                    warnings.extend(artifact_selection.warnings)
                    unknowns.extend(artifact_selection.unknowns)
                    atoms.append(build_artifact_atom("selected", selected_artifact))
                    if documented_artifact is not None:
                        atoms.append(build_artifact_atom("documented", documented_artifact))
                    for alternative in artifact_selection.alternatives:
                        atoms.append(build_artifact_atom("alternative", alternative))
                if strategy_selection is not None:
                    warnings.extend(strategy_selection.warnings)
                    if selected.decision_kind in {"inferred_preserve_topology", "unknown_or_reroute"}:
                        unknowns.append("最终并行拓扑仍未完全冻结")
                    atoms.append(build_strategy_atom("selected", selected))
                    if documented is not None:
                        atoms.append(build_strategy_atom("documented", documented))
                        atoms.append(
                            {
                                "atom_id": f"atom-{selected.model_base or 'model'}-expectation-validation",
                                "atom_kind": "validation",
                                "summary": documented.summary,
                                "source_refs": list(documented.source_refs),
                            }
                        )
                    for alternative in strategy_selection.alternatives:
                        atoms.append(build_strategy_atom("alternative", alternative))
                if perf_fact_rows:
                    atoms.append(
                        {
                            "atom_id": f"atom-performance-{model_base or 'unknown'}",
                            "atom_kind": "fact",
                            "summary": perf_fact_rows[0]["literal_text"],
                            "source_refs": [perf_fact_rows[0]["fact_id"]],
                        }
                    )
                if topology_fact_rows:
                    atoms.append(
                        {
                            "atom_id": f"atom-topology-{primary_hw.lower()}",
                            "atom_kind": "fact",
                            "summary": topology_fact_rows[0]["literal_text"],
                            "source_refs": [topology_fact_rows[0]["fact_id"]],
                        }
                    )
                if comm_profile_rows:
                    atoms.append(
                        {
                            "atom_id": f"atom-comm-profile-{primary_hw.lower()}",
                            "atom_kind": "fact",
                            "summary": comm_profile_rows[0]["literal_text"],
                            "source_refs": [comm_profile_rows[0]["fact_id"]],
                        }
                    )
                if tool_recipe_rows:
                    atoms.append(
                        {
                            "atom_id": f"atom-tool-recipe-{model_base or 'unknown'}",
                            "atom_kind": "fact",
                            "summary": tool_recipe_rows[0]["literal_text"],
                            "source_refs": [tool_recipe_rows[0]["fact_id"]],
                        }
                    )
                if runtime_constraint_rows:
                    atoms.append(
                        {
                            "atom_id": f"atom-runtime-constraint-{model_base or 'unknown'}",
                            "atom_kind": "constraint",
                            "summary": runtime_constraint_rows[0]["literal_text"],
                            "source_refs": [runtime_constraint_rows[0]["fact_id"]],
                        }
                    )
                elif substrate_rows:
                    atoms.append(
                        {
                            "atom_id": "atom-runtime-constraint",
                            "atom_kind": "fact",
                            "summary": substrate_rows[0]["literal_text"],
                            "source_refs": [f"{substrate_rows[0]['shard_family']}:runtime"],
                        }
                    )
                append_shadow_diagnostics()
                if selected_artifact is not None and selected_artifact.decision_kind == "unsupported_requires_choice":
                    topology_label = _strategy_topology_label(selected) if selected is not None else "当前拓扑待收口"
                    capsule_text = (
                        f"{model_label} 在 {primary_hw} 上的 TTFT、throughput 和 memory headroom 仍是 route-sensitive 的条件区间；"
                        f"native FP8 直跑当前不受支持，且请求只冻结到 {topology_label}，"
                        "因此必须先在 ModelSlim conversion 与 fp8-origin adaptation 之间收口路线，再讨论更窄的 envelope。"
                    )
                elif selected_artifact is not None and selected_artifact.decision_kind.endswith("convert_then_deploy"):
                    topology_label = _strategy_topology_label(selected) if selected is not None else "当前拓扑待运行时补齐"
                    capsule_text = (
                        f"{model_label} 在 {primary_hw} 上的 TTFT、throughput 和 memory headroom 应锚定到 converted artifact 路线；"
                        f"当前以 {topology_label} 为 topology anchor，结果必须显式保留 conversion 产物、graph mode 和 batch size 假设。"
                    )
                elif selected is not None and selected.decision_kind == "unknown_or_reroute":
                    alternative_labels = ", ".join(_strategy_topology_label(candidate) for candidate in strategy_selection.alternatives) or "多个未冻结候选"
                    capsule_text = (
                        f"{model_label} 在 {primary_hw} 上的 TTFT、throughput 和 memory headroom 需要按 topology-sensitive envelope 返回；"
                        f"当前文档化锚点是 {_strategy_topology_label(documented)}，但锁定请求下仍存在 {alternative_labels} 等候选，"
                        "因此不能给出单一性能点。"
                    )
                elif selected is not None and selected.decision_kind == "inferred_preserve_topology":
                    capsule_text = (
                        f"{model_label} 在 {primary_hw} 上的 TTFT、throughput 和 memory headroom 应锚定到推断 topology {_strategy_topology_label(selected)}；"
                        f"当前文档化锚点是 {_strategy_topology_label(documented)}，结果必须以下界/上界范围而不是单点返回。"
                    )
                else:
                    topology_label = _strategy_topology_label(selected) if selected is not None else "当前拓扑待运行时补齐"
                    capsule_text = (
                        f"{model_label} 在 {primary_hw} 上的 TTFT、throughput 和 memory headroom 应围绕 {topology_label} 返回 expected envelope；"
                        "若 graph mode、batch size 或 context length 未冻结，结论必须保留区间和假设。"
                    )
                refs: tuple[str, ...] | list[str] = ()
                if selected_artifact is not None and selected_artifact.artifact_refs:
                    refs = selected_artifact.artifact_refs
                elif selected is not None and selected.artifact_refs:
                    refs = selected.artifact_refs
                elif documented is not None:
                    refs = documented.artifact_refs
                deep_refs.extend(
                    _strategy_deep_refs(
                        request_id=request["request_id"],
                        refs=refs,
                        reason="需要查看 expectation anchor 的配置、部署与性能参考",
                    )
                )
            else:
                capsule_text = (
                    f"在 {primary_hw or '当前硬件'} 约束下，{primary_model or '当前模型'} 的 TTFT、throughput 和 memory headroom "
                    "应落在一个可解释区间内；若 batch size 和 graph mode 不固定，结果应以范围而非单点返回。"
                )
                warnings.append("batch size not fixed; returning envelope instead of single-point estimate")
                unknowns.extend(["graph mode enabled/disabled", "batch size 尚未冻结"])
                atoms.extend(
                    [
                        {
                            "atom_id": "atom-runtime-baseline",
                            "atom_kind": "validation",
                            "summary": "存在可比较 baseline，可为 expected TTFT/throughput 估计提供量级锚点。",
                            "source_refs": ["validation:baseline:qwen3-next-32b:a2:tp4"],
                        },
                        {
                            "atom_id": "atom-runtime-policy",
                            "atom_kind": "fact",
                            "summary": policy_rows[0]["literal_text"] if policy_rows else "runtime policy 限定了预期性能上界。",
                            "source_refs": ["repo_semantics:policy/prefill"],
                        },
                    ]
                )
                deep_refs.append(
                    {
                        "stub_id": "stub-runtime-baseline",
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
        elif intent == "debug_triage":
            error_blob = " ".join(selectors.get("errors", []))
            if "161001" in error_blob or "aclnnApplyRotaryPosEmbV2" in error_blob:
                capsule_text = "RuntimeError 161001 常见于 rotary path 与特定 kernel/shape 组合；现有兼容 workaround 是禁用相关 path 或收窄 shape。"
                atoms.extend(
                    [
                        {
                            "atom_id": "atom-161001-signature",
                            "atom_kind": "fact",
                            "summary": "161001 与 rotary 相关 fastpath 组合存在已知失败模式。",
                            "source_refs": ["repo_semantics:error-signature:161001"],
                        },
                        {
                            "atom_id": "atom-161001-workaround",
                            "atom_kind": "validation",
                            "summary": "兼容 workaround 是禁用相关 fastpath 或收窄输入 shape。",
                            "source_refs": ["validation:known-failure-161001"],
                        },
                    ]
                )
                if resolve_result["match_level"] != "exact":
                    unknowns.append("是否与当前固件版本完全一致")
                deep_refs.append(
                    {
                        "stub_id": "stub-161001-validation",
                        "source_ref": "validation/known-failure-161001.md",
                        "estimated_tokens": 220,
                        "reason": "需要查看完整失败上下文与版本说明",
                    }
                )
            else:
                capsule_text = "当前日志表现为通用运行时错误；已先收口错误签名、影响面和兼容 workaround。"
                atoms.append(
                    {
                        "atom_id": "atom-generic-debug-signature",
                        "atom_kind": "fact",
                        "summary": "已抽取错误签名与最小影响面，可继续做 triage。",
                        "source_refs": ["repo_semantics:error-signature:generic"],
                    }
                )
                unknowns.append("缺少更完整日志上下文")
        elif intent == "design_lookup":
            if (
                artifact_selection is not None
                or strategy_selection is not None
                or vllm_semantic_rows
                or vllm_symbol_rows
            ):
                selected = strategy_selection.selected if strategy_selection is not None else None
                documented = strategy_selection.documented if strategy_selection is not None else None
                selected_artifact = artifact_selection.selected if artifact_selection is not None else None
                documented_artifact = artifact_selection.documented if artifact_selection is not None else None
                model_base = (
                    selected.model_base
                    if selected is not None
                    else selected_artifact.model_base
                    if selected_artifact is not None
                    else primary_model
                )
                model_traits = (
                    selected.model_traits
                    if selected is not None
                    else selected_artifact.model_traits
                    if selected_artifact is not None
                    else tuple()
                )
                model_label = _model_label(model_base, model_traits)
                if artifact_selection is not None:
                    warnings.extend(artifact_selection.warnings)
                    unknowns.extend(artifact_selection.unknowns)
                    atoms.append(build_artifact_atom("selected", selected_artifact))
                    if documented_artifact is not None:
                        atoms.append(build_artifact_atom("documented", documented_artifact))
                    for alternative in artifact_selection.alternatives:
                        atoms.append(build_artifact_atom("alternative", alternative))
                if strategy_selection is not None:
                    warnings.extend(strategy_selection.warnings)
                    unknowns.extend(strategy_selection.unknowns)
                    atoms.append(build_strategy_atom("selected", selected))
                    if documented is not None:
                        atoms.append(build_strategy_atom("documented", documented))
                    for alternative in strategy_selection.alternatives:
                        atoms.append(build_strategy_atom("alternative", alternative))
                if topology_fact_rows:
                    atoms.append(
                        {
                            "atom_id": f"atom-topology-{primary_hw.lower()}",
                            "atom_kind": "fact",
                            "summary": topology_fact_rows[0]["literal_text"],
                            "source_refs": [topology_fact_rows[0]["fact_id"]],
                        }
                    )
                if comm_profile_rows:
                    atoms.append(
                        {
                            "atom_id": f"atom-comm-profile-{primary_hw.lower()}",
                            "atom_kind": "fact",
                            "summary": comm_profile_rows[0]["literal_text"],
                            "source_refs": [comm_profile_rows[0]["fact_id"]],
                        }
                    )
                if trait_fact_rows:
                    atoms.append(
                        {
                            "atom_id": f"atom-model-traits-{model_base or 'unknown'}",
                            "atom_kind": "fact",
                            "summary": trait_fact_rows[0]["literal_text"],
                            "source_refs": [trait_fact_rows[0]["fact_id"]],
                        }
                    )
                if tool_recipe_rows:
                    atoms.append(
                        {
                            "atom_id": f"atom-tool-recipe-{model_base or 'unknown'}",
                            "atom_kind": "fact",
                            "summary": tool_recipe_rows[0]["literal_text"],
                            "source_refs": [tool_recipe_rows[0]["fact_id"]],
                        }
                    )
                if runtime_constraint_rows:
                    atoms.append(
                        {
                            "atom_id": f"atom-runtime-constraint-{model_base or 'unknown'}",
                            "atom_kind": "constraint",
                            "summary": runtime_constraint_rows[0]["literal_text"],
                            "source_refs": [runtime_constraint_rows[0]["fact_id"]],
                        }
                    )
                if vllm_semantic_rows or vllm_symbol_rows:
                    atoms.extend(
                        [
                            {
                                "atom_id": "atom-vllm-upstream-semantics",
                                "atom_kind": "fact",
                                "summary": vllm_semantic_rows[0]["literal_text"] if vllm_semantic_rows else "已加载上游 engine/config 语义。",
                                "source_refs": ["vllm_semantics:engine/config"],
                            },
                            {
                                "atom_id": "atom-vllm-upstream-symbols",
                                "atom_kind": "symbol",
                                "summary": (
                                    f"关键上游符号已索引：{vllm_symbol_rows[0]['qualname']}"
                                    if vllm_symbol_rows
                                    else "已建立上游 symbol 索引。"
                                ),
                                "source_refs": ["vllm_symbols:EngineArgs.create_engine_config"],
                            },
                        ]
                    )
                append_shadow_diagnostics()
                if selected_artifact is not None and selected_artifact.decision_kind == "unsupported_requires_choice":
                    topology_label = _strategy_topology_label(selected) if selected is not None else "当前拓扑待收口"
                    capsule_text = (
                        f"{model_label} 在 {primary_hw} 上的 native FP8 直跑路线当前不受支持；"
                        f"设计分析阶段应先在 {topology_label} 这一拓扑前提下，"
                        "比较 ModelSlim conversion 与 fp8-origin adaptation 两条路线，再决定后续 deployment/adaptation 入口。"
                    )
                elif selected_artifact is not None and selected_artifact.decision_kind.endswith("convert_then_deploy"):
                    topology_label = _strategy_topology_label(selected) if selected is not None else "当前拓扑待运行时补齐"
                    capsule_text = (
                        f"{model_label} 在 {primary_hw} 上的设计分析已收口到 conversion + deployment 路线；"
                        f"当前 topology anchor 是 {topology_label}，后续只需要把 conversion runbook 和 serve runbook 进一步细化。"
                    )
                else:
                    capsule_text = "上游语义与当前 repo overlay 已形成可查询的设计分析上下文，可先输出路线分析 capsule。"
                unknowns.append("仍需结合具体 work package 才能收口最终路线")
                refs: tuple[str, ...] | list[str] = ()
                if selected_artifact is not None and selected_artifact.artifact_refs:
                    refs = selected_artifact.artifact_refs
                elif selected is not None and selected.artifact_refs:
                    refs = selected.artifact_refs
                elif documented is not None:
                    refs = documented.artifact_refs
                deep_refs.extend(
                    _strategy_deep_refs(
                        request_id=request["request_id"],
                        refs=refs,
                        reason="需要查看 route choice、tool recipe 和相关基线配置",
                    )
                )
            else:
                capsule_text = "当前 repo-only KB 无法充分支持该 intent，返回显式 miss。"
                unknowns.append("intent design_lookup requires richer upstream substrate")
        elif intent == "upstream_delta":
            if vllm_delta_rows:
                atoms.append(
                    {
                        "atom_id": "atom-vllm-release-delta",
                        "atom_kind": "fact",
                        "summary": "上游 release delta 与验证窗口已记录，可用于同步影响面分析。",
                        "source_refs": ["vllm_release_delta:release/current"],
                    }
                )
                if vllm_symbol_rows:
                    atoms.append(
                        {
                            "atom_id": "atom-vllm-delta-impacted-symbol",
                            "atom_kind": "symbol",
                            "summary": f"同步分析可落到受影响符号：{vllm_symbol_rows[0]['qualname']}。",
                            "source_refs": ["vllm_symbols:EngineArgs.create_engine_config"],
                        }
                    )
                capsule_text = "上游 release delta、影响符号与验证窗口已可打包，适合进入 upstream sync 的后续规划。"
                unknowns.append("最终 cherry-pick / sync 顺序仍需 bundle 中明确")
            else:
                capsule_text = "当前 repo-only KB 无法充分支持该 intent，返回显式 miss。"
                unknowns.append("intent upstream_delta requires release delta ingest")
        elif intent in {"debug_triage", "adaptation_codegen", "operator_codegen"}:
            capsule_text = "当前 repo-only KB 无法充分支持该 intent，返回显式 miss。"
            unknowns.append(f"intent {intent} requires deferred family support or richer substrate")
        else:
            capsule_text = "知识未命中明确 intent，返回 repo-only fallback。"
            unknowns.append("intent unsupported")

        logical_domains = request["logical_domains"]
        if not request["include_evidence_stubs"]:
            deep_refs = []
        atoms, deep_refs, estimated_tokens = _apply_budget(
            budget_token_cap=request["budget_token_cap"],
            atoms=atoms,
            deep_refs=deep_refs,
            warnings=warnings,
            unknowns=unknowns,
        )
        response = {
            "schema_version": "kb-pack-response/v1",
            "request_id": request["request_id"],
            "pack_id": f"pack-{request['request_id']}",
            "created_at": request["created_at"],
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
        "evidence_refs": args.evidence_refs or [],
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
    pack_parser.add_argument("--evidence-refs", nargs="*", default=[])
    pack_parser.add_argument("--budget-token-cap", type=int, default=1500)
    pack_parser.add_argument("--max-atoms", type=int, default=10)
    pack_parser.add_argument("--max-hops", type=int, default=1)
    pack_parser.add_argument("--include-evidence-stubs", action="store_true")
    pack_parser.add_argument("--stop-after-first-sufficient", action="store_true")
    pack_parser.add_argument("--emit", default=None)

    args = parser.parse_args(argv)
    root = (repo_root() / args.repo_root).resolve() if args.repo_root != "." else repo_root()

    try:
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
    except ContractError as exc:
        print(f"FAIL {exc}")
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
