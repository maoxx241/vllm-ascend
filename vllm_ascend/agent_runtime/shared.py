from __future__ import annotations

import hashlib
import re
from typing import Any, NamedTuple, TypedDict

from .bundle import build_continuation_state
from .contracts import copy_example, now_utc, validate_instance
from .kb import build_local, pack, resolve
from .paths import kb_root, repo_root


class RawRequest(NamedTuple):
    request_id: str
    user_text: str
    attachment_refs: list[str]
    inline_paths: list[str]
    inline_symbols: list[str]
    inline_errors: list[str]
    execution_context_hint: str | None = None


class ProgressState(TypedDict):
    bundle_exists: bool
    has_unflushed_findings: bool
    query_count_in_stage: int
    opened_deep_refs_in_stage: int
    seen_dedupe_keys: list[str]
    last_flush_at: str | None
    session_budget_used: int


HW_TOKENS = {
    "a2": "A2",
    "910b": "A2",
    "910b4": "A2",
    "a3": "A3",
    "910_9391": "A3",
}
MODEL_PATTERNS = [
    (re.compile(r"qwen3-next-32b", re.I), "qwen3-next-32b"),
    (re.compile(r"qwen3-next", re.I), "qwen3-next"),
]
FEATURE_PATTERNS = {
    "allgather_ep": re.compile(r"allgather[_ -]?ep", re.I),
    "prefill": re.compile(r"prefill", re.I),
    "decode": re.compile(r"decode", re.I),
    "tp4": re.compile(r"\btp\s*=?\s*4\b", re.I),
    "bf16": re.compile(r"bf16", re.I),
    "ctx8k": re.compile(r"(8k|8192)", re.I),
    "dynamic_batching": re.compile(r"dynamic[_ -]?batch", re.I),
}


def _detect_hw(text: str) -> list[str]:
    lowered = text.lower()
    hits = []
    for token, normalized in HW_TOKENS.items():
        if token in lowered and normalized not in hits:
            hits.append(normalized)
    return hits


def _detect_models(text: str) -> list[str]:
    hits = []
    for pattern, model in MODEL_PATTERNS:
        if pattern.search(text):
            hits.append(model)
    return hits


def _detect_features(text: str) -> list[str]:
    hits = []
    for feature, pattern in FEATURE_PATTERNS.items():
        if pattern.search(text):
            hits.append(feature)
    return hits


def _requested_artifact(text: str) -> str:
    if re.search(r"(deploy|deployment|config|prefill policy)", text, re.I):
        return "deployment_artifact_pack"
    if re.search(r"(spec|plan|design)", text, re.I):
        return "spec_plan"
    if re.search(r"(diff|test|validation)", text, re.I):
        return "analysis_report"
    return "analysis_report"


def _classify(text: str, inline_paths: list[str], inline_errors: list[str]) -> str:
    if re.search(r"(expect|estimate|ttft|throughput|headroom)", text, re.I) and "profile" not in text.lower():
        return "performance_expectation"
    if re.search(r"(profile|regression|kernel)", text, re.I):
        return "performance_breakdown"
    if inline_errors:
        return "performance_breakdown"
    if inline_paths or re.search(r"(diff|validation|test)", text, re.I):
        return "validation"
    if re.search(r"(deploy|deployment|policy|prefill)", text, re.I):
        return "deployment"
    if re.search(r"(adapt|operator|code change|patch|modify)", text, re.I):
        return "adaptation"
    return "design"


def _query_trigger_codes(kind: str) -> list[str]:
    mapping = {
        "deployment": ["baseline_or_policy_lookup", "artifact_requirement_lookup"],
        "performance_expectation": ["baseline_or_policy_lookup", "operator_constraint_lookup"],
        "performance_breakdown": ["baseline_or_policy_lookup", "evidence_gap_followup"],
        "validation": ["validation_matrix_lookup", "code_surface_lookup"],
        "design": ["route_disambiguation", "cross_surface_conflict_check"],
    }
    return mapping.get(kind, ["baseline_or_policy_lookup"])


def _confirmation_status(kind: str, text: str) -> tuple[bool, str]:
    if kind == "adaptation" or re.search(r"(code change|modify|mutation|destructive)", text, re.I):
        return True, "pending"
    return False, "not_needed"


def _base_seed_for_kind(kind: str, root: Any) -> dict[str, Any]:
    if kind == "deployment":
        return copy_example("selector-seed.deployment.json", root=root)
    if kind == "performance_expectation":
        return copy_example("selector-seed.performance.expectation.json", root=root)
    if kind == "adaptation":
        return copy_example("selector-seed.adaptation.pending-confirmation.json", root=root)
    return copy_example("selector-seed.performance.expectation.json", root=root)


def build_selector_seed(raw_request: RawRequest, root: Any | None = None) -> dict[str, Any]:
    root = root or repo_root()
    kind = _classify(raw_request.user_text, raw_request.inline_paths, raw_request.inline_errors)
    seed = _base_seed_for_kind(kind, root=root)
    features = _detect_features(raw_request.user_text)
    models = _detect_models(raw_request.user_text)
    hw = _detect_hw(raw_request.user_text)
    confirmation_required, confirmation_status = _confirmation_status(kind, raw_request.user_text)

    family_candidates = {
        "deployment": ["deployment_execution", "design_analysis"],
        "performance_expectation": ["performance_analysis"],
        "performance_breakdown": ["performance_analysis"],
        "validation": ["validation_strategy"],
        "adaptation": ["adaptation", "design_analysis"],
        "design": ["design_analysis"],
    }[kind]
    execution_mode = "direct_atomic_workflow" if kind != "design" else "spec_plan_workflow"
    deliverable = (
        "deployment_artifact_pack"
        if kind == "deployment"
        else "code_change_pack"
        if kind == "adaptation"
        else "analysis_report"
    )
    seed.update(
        {
            "request_id": raw_request.request_id,
            "created_at": now_utc(),
            "objective": raw_request.user_text,
            "requested_artifact": deliverable,
            "task_family_candidates": family_candidates,
            "execution_mode_hint": execution_mode,
            "deliverable_contract_hint": deliverable if deliverable != "analysis_report" else "analysis_report",
            "analysis_depth_hint": "full_spec_plan" if execution_mode == "spec_plan_workflow" else "lightweight_design_note",
            "normalized_entities": {
                "files": raw_request.inline_paths,
                "symbols": raw_request.inline_symbols,
                "entities": seed["normalized_entities"]["entities"],
                "errors": raw_request.inline_errors,
                "models": models or seed["normalized_entities"]["models"],
                "features": features or seed["normalized_entities"]["features"],
                "hw": hw or seed["normalized_entities"]["hw"],
                "commits": [],
                "prs": [],
                "versions": seed["normalized_entities"]["versions"],
                "configs": seed["normalized_entities"]["configs"],
            },
            "evidence_inventory": {
                "evidence_kinds": sorted(
                    {
                        "user_request_only",
                        "runtime_tuple",
                        *(
                            ["code_or_diff"]
                            if raw_request.inline_paths
                            and any(not path.startswith("tests/") for path in raw_request.inline_paths)
                            else []
                        ),
                        *(
                            ["tests_or_validation_assets"]
                            if raw_request.inline_paths
                            and any(path.startswith("tests/") or "test_" in path for path in raw_request.inline_paths)
                            else []
                        ),
                        *(["profiling"] if raw_request.inline_errors else []),
                    }
                ),
                "has_runtime_tuple": True,
                "has_baseline": bool(re.search(r"baseline", raw_request.user_text, re.I)),
                "has_validation_assets": bool(raw_request.inline_paths),
                "inline_refs": raw_request.attachment_refs + raw_request.inline_paths,
            },
            "execution_context": raw_request.execution_context_hint or "local_only",
            "code_change_expectation": "code_change_required" if confirmation_required else "no_change_expected",
            "confirmation_required": confirmation_required,
            "confirmation_status": confirmation_status,
            "confirmation_reason_codes": ["code_change_scope"] if confirmation_required else [],
            "smallest_next_step": "查一个正式 capsule 再决定是否继续推进",
            "what_is_missing": ["baseline"] if kind == "performance_breakdown" else seed["what_is_missing"],
        }
    )
    validate_instance(seed, "selector-seed.schema.json", root=root)
    return seed


def plan_from_seed(selector_seed: dict[str, Any], root: Any | None = None) -> dict[str, Any] | None:
    root = root or repo_root()
    if selector_seed["confirmation_required"] and selector_seed["confirmation_status"] in {"pending", "user_declined"}:
        return None
    family = selector_seed["task_family_candidates"][0]
    objective = selector_seed["objective"]
    if family == "deployment_execution":
        plan = copy_example("selector-plan.deployment.intake.json", root=root)
    elif family == "performance_analysis" and re.search(r"(expect|estimate|ttft|throughput)", objective, re.I):
        plan = copy_example("selector-plan.performance.expectation.atomic.json", root=root)
    elif family == "performance_analysis":
        plan = copy_example("selector-plan.performance.atomic.json", root=root)
    elif family == "validation_strategy":
        plan = copy_example("selector-plan.validation.atomic.json", root=root)
    else:
        plan = copy_example("selector-plan.design.spec.json", root=root)

    slug = re.sub(r"[^a-z0-9]+", "-", selector_seed["request_id"].lower()).strip("-")
    plan.update(
        {
            "plan_id": f"plan-{slug}",
            "request_id": selector_seed["request_id"],
            "created_at": now_utc(),
            "task_family": family,
            "selectors": selector_seed["normalized_entities"],
            "work_package_goal": selector_seed["objective"],
            "must_have": selector_seed["what_is_missing"][:3] or plan["must_have"],
            "nice_to_have": selector_seed["constraints"]["soft_constraints"][:3] or plan["nice_to_have"],
            "query_trigger_codes": _query_trigger_codes(_classify(objective, selector_seed["normalized_entities"]["files"], selector_seed["normalized_entities"]["errors"])),
            "why_this_query_now": selector_seed["smallest_next_step"],
            "notes": "; ".join(selector_seed["constraints"]["hard_constraints"]) or plan["notes"],
        }
    )
    if family == "validation_strategy":
        plan["selectors"]["files"] = selector_seed["normalized_entities"]["files"]
    validate_instance(plan, "selector-plan.schema.json", root=root)
    return plan


def _dedupe_key(selector_plan: dict[str, Any], flush: bool = False) -> str:
    suffix = "pending-flush" if flush else selector_plan["consumer_id"]
    return f"{selector_plan['request_id']}:{selector_plan['query_stage']}:{suffix}"


def evaluate_governor(
    *,
    selector_seed: dict[str, Any],
    selector_plan: dict[str, Any],
    continuation_state: dict[str, Any] | None,
    progress_state: ProgressState,
    root: Any | None = None,
) -> dict[str, Any]:
    root = root or repo_root()
    stage = selector_plan["query_stage"]
    budget_map = {
        "intake": ("intake", 1200),
        "atomic": ("atomic", 1500),
        "spec_plan": ("spec", 2400),
    }
    resolved_budget_class, stage_cap = budget_map.get(stage, ("routing", 400))
    flush_required = bool(progress_state["has_unflushed_findings"] and progress_state["query_count_in_stage"] > 0)
    warnings: list[str] = []
    denial_reason_code = None

    if not selector_plan.get("query_trigger_codes"):
        denial_reason_code = "missing_trigger_code"
    elif selector_seed["confirmation_status"] in {"pending", "user_declined"} and stage in {"atomic", "spec_plan"}:
        denial_reason_code = "stage_disallows_query"
    elif flush_required:
        denial_reason_code = "pending_flush"
        warnings.append("pending flush before second capsule")
    elif continuation_state and continuation_state["persistence_mode"] != "full_bundle":
        denial_reason_code = "missing_persistence_bundle"
    elif _dedupe_key(selector_plan) in progress_state["seen_dedupe_keys"]:
        denial_reason_code = "duplicate_plan"
    elif (
        selector_plan["origin_stage"] == "intake"
        and selector_plan["query_stage"] == "atomic"
        and selector_plan["execution_mode"] != "direct_atomic_workflow"
    ):
        denial_reason_code = "stage_disallows_query"

    allow_query = denial_reason_code is None
    decision = {
        "schema_version": "governor-decision/v1",
        "request_id": selector_plan["request_id"],
        "stage": stage,
        "allow_query": allow_query,
        "resolved_budget_class": resolved_budget_class,
        "resolved_token_cap": min(selector_plan["requested_token_cap"], stage_cap) if allow_query else 0,
        "max_capsules": selector_plan["max_capsules"] if allow_query else 0,
        "max_deep_refs": selector_plan["max_deep_refs"] if allow_query else 0,
        "flush_required": flush_required,
        "must_compact_after": progress_state["session_budget_used"] >= 51200,
        "cache_preferred": allow_query,
        "warnings": warnings,
        "denial_reason_code": denial_reason_code,
        "dedupe_key": _dedupe_key(selector_plan, flush=flush_required and not allow_query),
        "compaction_target_tokens": 3200 if progress_state["session_budget_used"] >= 51200 else None,
    }
    validate_instance(decision, "governor-decision.schema.json", root=root)
    return decision


def compile_pack_request(
    selector_plan: dict[str, Any],
    governor_decision: dict[str, Any],
    root: Any | None = None,
) -> dict[str, Any]:
    root = root or repo_root()
    mapping = {
        ("intake", "deployment_execution", "deployment-intake"): "intake_lookup",
        ("atomic", "deployment_execution", "feature-policy-resolver"): "deployment_lookup",
        ("atomic", "performance_analysis", "single-profile-breakdown"): "perf_breakdown",
        ("atomic", "performance_analysis", "comparative-profile-breakdown"): "perf_breakdown",
        ("atomic", "performance_analysis", "model-expected-performance-estimator"): "model_expectation",
        ("atomic", "validation_strategy", "change-impact-test-selector"): "validation_selection",
        ("spec_plan", "design_analysis", "design-spec"): "design_lookup",
    }
    intent = mapping.get(
        (selector_plan["query_stage"], selector_plan["task_family"], selector_plan["consumer_id"]),
        "deployment_lookup",
    )
    request = {
        "schema_version": "kb-pack-request/v2",
        "request_id": selector_plan["request_id"],
        "created_at": now_utc(),
        "intent": intent,
        "repo_root": ".",
        "resolve_policy": "auto",
        "logical_domains": selector_plan["logical_domains"],
        "physical_shard_hints": selector_plan["physical_shard_hints"],
        "selectors": selector_plan["selectors"],
        "must_have": selector_plan["must_have"],
        "nice_to_have": selector_plan["nice_to_have"],
        "evidence_refs": [],
        "budget_token_cap": governor_decision["resolved_token_cap"],
        "max_atoms": selector_plan["max_capsules"] * 10,
        "max_hops": selector_plan["hop_limit"],
        "include_evidence_stubs": selector_plan["capsule_type"] != "intake_capsule",
        "stop_after_first_sufficient": selector_plan["stop_after_first_sufficient"],
        "emit_path": f".agents/kb/local/capsules/{selector_plan['request_id']}.json",
    }
    validate_instance(request, "kb-pack-request.schema.json", root=root)
    return request


def load_capsule(
    *,
    selector_plan: dict[str, Any],
    governor_decision: dict[str, Any],
    repo_root_arg: str = ".",
    root: Any | None = None,
) -> dict[str, Any]:
    root = root or repo_root()
    repo_root_path = repo_root()
    if repo_root_arg != ".":
        repo_root_path = repo_root_path / repo_root_arg
    resolve_path = kb_root(root) / "local" / "resolve.json"
    merge_path = kb_root(root) / "local" / "merged" / "current.sqlite"
    if not resolve_path.exists():
        resolve_result = resolve(repo_root_path, emit_path=resolve_path)
    else:
        from .contracts import load_json

        resolve_result = load_json(resolve_path)
    if not merge_path.exists():
        build_local(repo_root_path, resolve_result=resolve_result, emit_sqlite=merge_path)
    request = compile_pack_request(selector_plan, governor_decision, root=root)
    return pack(repo_root_path, request=request, resolve_result=resolve_result, merged_pack=merge_path)


def generic_task_intake(raw_request: RawRequest, root: Any | None = None) -> dict[str, Any]:
    root = root or repo_root()
    seed = build_selector_seed(raw_request, root=root)
    plan = plan_from_seed(seed, root=root)
    blocked = plan is None
    return {"selector_seed": seed, "selector_plan": plan, "blocked": blocked}


def generic_spec(selector_plan: dict[str, Any], root: Any | None = None) -> dict[str, Any]:
    root = root or repo_root()
    task_id = f"task-{selector_plan['request_id']}"
    return build_continuation_state(
        task_id=task_id,
        request_id=selector_plan["request_id"],
        selector_plan=selector_plan,
        goal=selector_plan["work_package_goal"],
        root=root,
    )


def generic_analysis_checklist(selector_plan: dict[str, Any], pack_response: dict[str, Any], root: Any | None = None) -> dict[str, Any]:
    root = root or repo_root()
    card = copy_example("atomic-result-card.performance.partial.json", root=root)
    card.update(
        {
            "card_id": f"card-{selector_plan['request_id']}",
            "request_id": selector_plan["request_id"],
            "task_id": f"task-{selector_plan['request_id']}",
            "created_at": now_utc(),
            "task_family": selector_plan["task_family"],
            "atomic_skill": "generic-analysis-checklist",
            "work_package_id": selector_plan["work_package_id"],
            "source_plan_id": selector_plan["plan_id"],
            "finding_summary": pack_response["capsule_text"],
            "evidence_summary": [atom["summary"] for atom in pack_response["atoms"]],
            "source_refs": [ref for atom in pack_response["atoms"] for ref in atom["source_refs"]],
        }
    )
    validate_instance(card, "atomic-result-card.schema.json", root=root)
    return card
