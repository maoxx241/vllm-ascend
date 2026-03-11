#!/usr/bin/env python3
"""Workspace-aware validation and import pipeline for the code knowledge base."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from jsonschema import Draft7Validator


TASK_TYPES = (
    "deployment",
    "env_bootstrap",
    "debugging",
    "model_adaptation",
    "upstream_sync",
    "release_analysis",
    "op_development",
    "performance_analysis",
    "design_analysis",
)

ENTRY_SKILLS = (
    "developer-assistant",
    "deployment-assistant",
)

COMPOSER_SKILLS = (
    "model-adapter",
    "sync-coordinator",
    "debug-assistant",
    "release-assistant",
    "op-developer",
    "perf-assistant",
)

ATOMIC_SKILLS = (
    "env-bootstrap",
    "compatibility-checker",
    "repo-state-auditor",
    "log-analyzer",
    "crash-rooter",
    "perf-hunter",
    "graph-analyzer",
    "parallelism-planner",
    "scheduler-feature-designer",
    "attention-kv-designer",
    "custom-model-integrator",
    "precision-validator",
    "release-commit-analyzer",
    "release-notes-composer",
    "docs-compliance-checker",
    "test-matrix-planner",
    "ci-gatekeeper",
    "knowledge-index-maintainer",
)

ALL_SKILLS = set(ENTRY_SKILLS + COMPOSER_SKILLS + ATOMIC_SKILLS)
MIGRATION_TARGETS = (
    "ascend-foundation",
    "vllm-upstream",
    "vllm-ascend-core",
    "model-adaptation",
    "deployment-config",
    "troubleshooting",
    "documentation",
)
STATUS_ORDER = (
    "todo",
    "in_review",
    "validated",
    "validated_with_gap",
    "rewrite_required",
    "reject",
)
FACTUAL_VERDICTS = (
    "code_confirmed",
    "code_doc_aligned",
    "code_web_aligned",
    "code_doc_conflict",
    "insufficient_evidence",
)
UTILITY_VERDICTS = (
    "direct_skill_topic",
    "reference_only",
    "merge_required",
    "duplicate",
    "not_useful",
)
REQUIRED_PERSPECTIVES = ("test", "deploy", "develop", "design")
ID_PATTERN = re.compile(r"^[a-z_]+:[a-z_0-9]+$")
REPO_BASE_URLS = {
    "vllm": "https://github.com/vllm-project/vllm/blob/{commit}/{path}",
    "vllm-ascend": "https://github.com/vllm-project/vllm-ascend/blob/{commit}/{path}",
}
SEARCH_ROOT_CANDIDATES = (
    "docs",
    "examples",
    "tests",
    "benchmarks",
    "README.md",
    "README.zh.md",
)
EXPORT_FILES = {
    "manifest": "imported_knowledge_manifest.json",
    "search_index": "imported_knowledge_search_index.json",
    "design_index": "design_analysis_index.json",
    "task_skill_index": "task_skill_index.json",
    "scenario_coverage": "skill_scenario_coverage.json",
    "domain_index": "domain_index.json",
    "report": "imported_knowledge_report.json",
}
DOMAIN_SCOPES = (
    "vllm",
    "vllm-ascend",
    "both",
)
KNOWLEDGE_DOMAINS = (
    "vllm-upstream",
    "vllm-ascend-core",
    "integration-core",
)
DOMAIN_SCOPE_TO_KNOWLEDGE_DOMAIN = {
    "vllm": "vllm-upstream",
    "vllm-ascend": "vllm-ascend-core",
    "both": "integration-core",
}
KNOWLEDGE_DOMAIN_TO_DOMAIN_SCOPE = {
    knowledge_domain: domain_scope
    for domain_scope, knowledge_domain in DOMAIN_SCOPE_TO_KNOWLEDGE_DOMAIN.items()
}
OFFICIAL_PLATFORM_DOCS = {
    "device:310_p": [
        "https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform",
        "https://www.hiascend.com/doc_center/source/zh/Atlas%20300I%20Pro%20Inference%20Card/Atlas%20300I%20Pro%20Inference%20Card/300Ipro_002.html",
    ],
}
SCENARIO_LIBRARY = (
    {
        "id": "deployment_single_node_qwen3_w8a8",
        "title": "Single-node deployment with graph and KV checks",
        "task_type": "deployment",
        "query": "Deploy Qwen3-32B W8A8 on a single Ascend node and verify graph mode, KV layout, and compatibility constraints.",
        "entry_skill": "deployment-assistant",
        "composer_skill": None,
        "atomic_skills": ["compatibility-checker", "attention-kv-designer"],
        "required_docs": [
            "task-index.md",
            "deployment-config/procedures/deployment-playbook.md",
            "knowledge-governance/generated/task_skill_index.json",
        ],
        "evidence_entry_ids": ["device:A3", "feature:acl_graph", "feature:kv_nz_format", "quantization:W8A8"],
    },
    {
        "id": "env_bootstrap_a3_cann",
        "title": "Bootstrap an Ascend runtime environment",
        "task_type": "env_bootstrap",
        "query": "Bootstrap a local A3 environment, check core env vars, and confirm the runtime baseline before any deployment.",
        "entry_skill": "developer-assistant",
        "composer_skill": None,
        "atomic_skills": ["env-bootstrap"],
        "required_docs": [
            "task-index.md",
            "ascend-foundation/procedures/env-bootstrap-baseline.md",
        ],
        "evidence_entry_ids": ["device:A3", "env_var:ASCEND_HOME_PATH", "env_var:SOC_VERSION"],
    },
    {
        "id": "debug_hccl_startup_crash",
        "title": "Startup crash with distributed runtime signals",
        "task_type": "debugging",
        "query": "Debug a startup failure that mentions HCCL setup, missing runtime libraries, and worker initialization errors.",
        "entry_skill": "developer-assistant",
        "composer_skill": "debug-assistant",
        "atomic_skills": ["log-analyzer", "crash-rooter"],
        "required_docs": [
            "task-index.md",
            "knowledge-governance/generated/task_skill_index.json",
            "knowledge-governance/generated/imported_knowledge_report.json",
        ],
        "evidence_entry_ids": ["env_var:HCCL_SO_PATH", "api:NPUWorker", "api:NPUCommunicator"],
    },
    {
        "id": "debug_acl_graph_shape_drift",
        "title": "ACL graph capture failure or shape drift",
        "task_type": "debugging",
        "query": "Investigate ACL graph capture failures, shape drift, or graph replay mismatches in a serving path.",
        "entry_skill": "developer-assistant",
        "composer_skill": "debug-assistant",
        "atomic_skills": ["graph-analyzer", "log-analyzer"],
        "required_docs": [
            "task-index.md",
            "code-knowledge-map.md",
            "knowledge-governance/generated/design_analysis_index.json",
        ],
        "evidence_entry_ids": ["feature:acl_graph", "api:ACLGraphWrapper", "feature:batch_invariant"],
    },
    {
        "id": "model_adapter_new_runner_surface",
        "title": "Adapt a model to a new runner/runtime surface",
        "task_type": "model_adaptation",
        "query": "Adapt a model to the Ascend runner stack and validate runtime entrypoints, worker wiring, and integration touchpoints.",
        "entry_skill": "developer-assistant",
        "composer_skill": "model-adapter",
        "atomic_skills": ["custom-model-integrator", "precision-validator"],
        "required_docs": [
            "task-index.md",
            "knowledge-governance/generated/imported_knowledge_manifest.json",
            "knowledge-governance/generated/task_skill_index.json",
        ],
        "evidence_entry_ids": ["api:NPUModelRunner", "api:NPUModelRunnerV2", "api:set_ascend_forward_context"],
    },
    {
        "id": "model_adapter_attention_rope_path",
        "title": "Adapt a model with custom attention or rope behavior",
        "task_type": "model_adaptation",
        "query": "Assess how to integrate a model with custom attention, rope, or MLA behavior on the Ascend stack.",
        "entry_skill": "developer-assistant",
        "composer_skill": "model-adapter",
        "atomic_skills": ["custom-model-integrator", "attention-kv-designer"],
        "required_docs": [
            "task-index.md",
            "code-knowledge-map.md",
            "knowledge-governance/generated/design_analysis_index.json",
        ],
        "evidence_entry_ids": ["api:AscendAttentionBackend", "api:AscendRotaryEmbedding", "api:AscendMLABackend"],
    },
    {
        "id": "upstream_sync_runner_delta",
        "title": "Assess upstream runner and engine deltas",
        "task_type": "upstream_sync",
        "query": "Compare upstream runner and engine changes against the local Ascend adaptation and scope the required follow-up work.",
        "entry_skill": "developer-assistant",
        "composer_skill": "sync-coordinator",
        "atomic_skills": ["repo-state-auditor", "compatibility-checker", "knowledge-index-maintainer"],
        "required_docs": [
            "task-index.md",
            "knowledge-governance/generated/imported_knowledge_report.json",
            "knowledge-governance/generated/task_skill_index.json",
        ],
        "evidence_entry_ids": ["api:ModelRunner", "api:NPUModelRunnerV2", "api:LLMEngine"],
    },
    {
        "id": "upstream_sync_spec_decode_delta",
        "title": "Assess upstream speculative decoding changes",
        "task_type": "upstream_sync",
        "query": "Review speculative decoding changes from upstream and determine compatibility, gating, and index updates for Ascend.",
        "entry_skill": "developer-assistant",
        "composer_skill": "sync-coordinator",
        "atomic_skills": ["repo-state-auditor", "ci-gatekeeper", "knowledge-index-maintainer"],
        "required_docs": [
            "task-index.md",
            "knowledge-governance/generated/imported_knowledge_report.json",
            "knowledge-governance/generated/task_skill_index.json",
        ],
        "evidence_entry_ids": ["feature:ascend_speculative_decoding", "api:AscendEagleProposer", "api:AscendMedusaProposer"],
    },
    {
        "id": "release_feature_rollup",
        "title": "Compose a feature-focused release summary",
        "task_type": "release_analysis",
        "query": "Summarize user-visible feature changes for a release and organize them into adoption, compatibility, and migration notes.",
        "entry_skill": "developer-assistant",
        "composer_skill": "release-assistant",
        "atomic_skills": ["release-commit-analyzer", "release-notes-composer", "docs-compliance-checker"],
        "required_docs": [
            "task-index.md",
            "knowledge-governance/generated/imported_knowledge_report.json",
            "knowledge-governance/generated/task_skill_index.json",
        ],
        "evidence_entry_ids": ["feature:flashcomm2", "feature:dynamic_eplb", "feature:weight_prefetch"],
    },
    {
        "id": "release_quantization_gate",
        "title": "Compose a quantization-focused release gate summary",
        "task_type": "release_analysis",
        "query": "Prepare release notes for quantization-related changes and identify what needs CI or docs gating before publication.",
        "entry_skill": "developer-assistant",
        "composer_skill": "release-assistant",
        "atomic_skills": ["release-commit-analyzer", "release-notes-composer", "ci-gatekeeper"],
        "required_docs": [
            "task-index.md",
            "knowledge-governance/generated/imported_knowledge_report.json",
            "knowledge-governance/generated/task_skill_index.json",
        ],
        "evidence_entry_ids": ["quantization:W4A16", "quantization:W8A8_DYNAMIC", "feature:mlapo"],
    },
    {
        "id": "op_development_quant_matmul",
        "title": "Implement or audit a quantized matmul operator path",
        "task_type": "op_development",
        "query": "Develop or audit an Ascend quantized matmul operator and verify graph, precision, and integration boundaries.",
        "entry_skill": "developer-assistant",
        "composer_skill": "op-developer",
        "atomic_skills": ["graph-analyzer", "precision-validator", "ci-gatekeeper"],
        "required_docs": [
            "task-index.md",
            "code-knowledge-map.md",
            "knowledge-governance/generated/imported_knowledge_manifest.json",
        ],
        "evidence_entry_ids": ["operator:npu_quant_matmul", "quantization:W8A8", "api:AscendW8A8LinearMethod"],
    },
    {
        "id": "op_development_paged_attention",
        "title": "Implement or debug a paged-attention operator path",
        "task_type": "op_development",
        "query": "Work on a paged-attention operator path and verify attention layout, KV cache behavior, and numerical correctness.",
        "entry_skill": "developer-assistant",
        "composer_skill": "op-developer",
        "atomic_skills": ["attention-kv-designer", "precision-validator"],
        "required_docs": [
            "task-index.md",
            "code-knowledge-map.md",
            "knowledge-governance/generated/design_analysis_index.json",
        ],
        "evidence_entry_ids": ["operator:npu_paged_attention", "feature:paged_attention", "feature:prefix_caching"],
    },
    {
        "id": "performance_acl_graph_regression",
        "title": "Investigate graph-capture throughput regression",
        "task_type": "performance_analysis",
        "query": "Analyze a throughput regression around ACL graph capture, batch invariance, and serving-path replay overhead.",
        "entry_skill": "developer-assistant",
        "composer_skill": "perf-assistant",
        "atomic_skills": ["perf-hunter", "graph-analyzer", "test-matrix-planner"],
        "required_docs": [
            "task-index.md",
            "knowledge-governance/generated/imported_knowledge_report.json",
            "knowledge-governance/generated/design_analysis_index.json",
        ],
        "evidence_entry_ids": ["feature:acl_graph", "feature:batch_invariant", "feature:balance_scheduling"],
    },
    {
        "id": "performance_parallelism_tradeoff",
        "title": "Investigate scheduling and parallelism tradeoffs",
        "task_type": "performance_analysis",
        "query": "Analyze a performance tradeoff across dynamic batching, context parallelism, and expert load balancing.",
        "entry_skill": "developer-assistant",
        "composer_skill": "perf-assistant",
        "atomic_skills": ["perf-hunter", "parallelism-planner", "scheduler-feature-designer"],
        "required_docs": [
            "task-index.md",
            "knowledge-governance/generated/task_skill_index.json",
            "knowledge-governance/generated/design_analysis_index.json",
        ],
        "evidence_entry_ids": ["feature:dynamic_batch_scheduler", "feature:context_parallel", "feature:eplb"],
    },
    {
        "id": "design_kv_transfer_surface",
        "title": "Analyze KV transfer and disaggregated prefill design",
        "task_type": "design_analysis",
        "query": "Analyze the design of KV transfer, paged KV management, and disaggregated prefill across nodes.",
        "entry_skill": "developer-assistant",
        "composer_skill": None,
        "atomic_skills": ["attention-kv-designer", "parallelism-planner"],
        "required_docs": [
            "task-index.md",
            "code-knowledge-map.md",
            "knowledge-governance/generated/design_analysis_index.json",
        ],
        "evidence_entry_ids": ["feature:kv_transfer_ascend", "feature:paged_attention", "feature:pd_disaggregation"],
    },
    {
        "id": "design_mla_precision_surface",
        "title": "Analyze MLA, precision, and prefetch design",
        "task_type": "design_analysis",
        "query": "Analyze MLA-related execution, precision, and weight-prefetch design surfaces for the Ascend stack.",
        "entry_skill": "developer-assistant",
        "composer_skill": None,
        "atomic_skills": ["attention-kv-designer", "precision-validator"],
        "required_docs": [
            "task-index.md",
            "code-knowledge-map.md",
            "knowledge-governance/generated/design_analysis_index.json",
        ],
        "evidence_entry_ids": ["feature:mlapo", "env_var:VLLM_ASCEND_ENABLE_MLAPO", "config:weight_prefetch_config"],
    },
)


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_source_hash(payload: dict[str, Any]) -> str:
    canonical = json.loads(json.dumps(payload))
    canonical.pop("source_hash", None)
    return hashlib.sha256(canonical_json_bytes(canonical)).hexdigest()


def compute_snapshot_hash(entries: list[dict[str, Any]]) -> str:
    payload = [
        {
            "source_file": entry["source_file"],
            "source_hash": entry["source_hash"],
        }
        for entry in sorted(entries, key=lambda row: row["source_file"])
    ]
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def normalize_token(value: str) -> str:
    token = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    token = token.replace("-", "_").replace(".", "_").replace("/", "_")
    token = re.sub(r"[^a-zA-Z0-9_]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token.lower()


def normalize_source_id(value: str) -> str:
    if ":" not in value:
        return normalize_token(value)
    prefix, suffix = value.split(":", 1)
    return f"{normalize_token(prefix)}:{normalize_token(suffix)}"


def knowledge_domain_from_scope(domain_scope: str) -> str:
    return DOMAIN_SCOPE_TO_KNOWLEDGE_DOMAIN[domain_scope]


def domain_scope_from_knowledge_domain(knowledge_domain: str) -> str:
    return KNOWLEDGE_DOMAIN_TO_DOMAIN_SCOPE[knowledge_domain]


def qualified_source_file(knowledge_domain: str, legacy_source_file: str) -> str:
    return Path(knowledge_domain, legacy_source_file).as_posix()


def payload_search_terms(payload: dict[str, Any]) -> set[str]:
    terms = {
        str(payload.get("name", "")),
        str(payload.get("id", "")),
        str(payload.get("id", "")).split(":", 1)[-1],
        normalize_token(str(payload.get("name", ""))),
        normalize_token(str(payload.get("id", "")).split(":", 1)[-1]),
    }

    for code_path in payload.get("code_paths", []):
        path = str(code_path.get("path", ""))
        if not path:
            continue
        path_obj = Path(path)
        terms.add(path_obj.name)
        terms.add(path_obj.stem)
        if len(path_obj.parts) >= 2:
            terms.add(path_obj.parts[-2])

    perspectives = payload.get("perspectives", {})
    for location in perspectives.get("develop", {}).get("code_locations", []):
        if isinstance(location, dict):
            terms.add(str(location.get("file", "")))
            terms.add(str(location.get("function", "")))
        elif location:
            terms.add(str(location))

    for case in perspectives.get("test", {}).get("test_cases", []):
        strings = []
        if isinstance(case, dict):
            strings.extend([str(case.get("name", "")), str(case.get("description", ""))])
        elif case:
            strings.append(str(case))
        for value in strings:
            terms.add(value)
            for match in re.findall(r"[A-Za-z0-9_./-]+\.(?:py|md|json|ya?ml|txt|cu|cpp|cuh|hpp)", value):
                terms.add(match)
                terms.add(Path(match).name)
                terms.add(Path(match).stem)

    filtered = {
        term.strip()
        for term in terms
        if term
        and term.strip()
        and (len(term.strip()) >= 3 or "." in term or "/" in term)
    }
    return filtered


def expand_reference_aliases(ref_value: str) -> list[str]:
    aliases = {
        ref_value,
        ref_value.lower(),
        normalize_token(ref_value),
    }
    if "=" in ref_value:
        lhs = ref_value.split("=", 1)[0]
        aliases.update({lhs, lhs.lower(), normalize_token(lhs)})
    if ":" in ref_value:
        suffix = ref_value.split(":", 1)[-1]
        aliases.update({suffix, suffix.lower(), normalize_token(suffix)})
    if "." in ref_value:
        prefix = ref_value.split(".", 1)[0]
        suffix = ref_value.rsplit(".", 1)[-1]
        aliases.update(
            {
                prefix,
                prefix.lower(),
                normalize_token(prefix),
                suffix,
                suffix.lower(),
                normalize_token(suffix),
            }
        )
    for suffix in ("_V1", "_v1", "V1", "v1"):
        if ref_value.endswith(suffix) and len(ref_value) > len(suffix):
            base = ref_value[: -len(suffix)]
            aliases.update({base, base.lower(), normalize_token(base)})
    return [alias for alias in aliases if alias]


def is_repo_relative_path(rel_path: str) -> bool:
    path_obj = Path(rel_path)
    if path_obj.is_absolute():
        return False
    if any(part == ".." for part in path_obj.parts):
        return False
    return True


def normalize_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    normalized = json.loads(json.dumps(payload))
    notes: list[str] = []

    source_id = str(normalized.get("id", ""))
    if ":" in source_id:
        prefix, suffix = source_id.split(":", 1)
        normalized_id = f"{normalize_token(prefix)}:{normalize_token(suffix)}"
        if normalized_id != source_id:
            normalized["id"] = normalized_id
            notes.append("normalized:id")

    test_cases = (
        normalized.get("perspectives", {})
        .get("test", {})
        .get("test_cases")
    )
    if isinstance(test_cases, list):
        updated_cases = []
        changed = False
        for row in test_cases:
            if isinstance(row, str):
                updated_cases.append({"name": row, "description": ""})
                changed = True
            elif isinstance(row, dict):
                row = dict(row)
                row.setdefault("name", "")
                row.setdefault("description", "")
                updated_cases.append(row)
            else:
                updated_cases.append({"name": str(row), "description": ""})
                changed = True
        if changed:
            normalized["perspectives"]["test"]["test_cases"] = updated_cases
            notes.append("normalized:test.test_cases")

    code_locations = (
        normalized.get("perspectives", {})
        .get("develop", {})
        .get("code_locations")
    )
    if isinstance(code_locations, list):
        updated_locations = []
        changed = False
        for row in code_locations:
            if isinstance(row, str):
                updated_locations.append({"file": row, "function": "", "purpose": ""})
                changed = True
            elif isinstance(row, dict):
                row = dict(row)
                row.setdefault("file", "")
                row.setdefault("function", "")
                row.setdefault("purpose", "")
                updated_locations.append(row)
            else:
                updated_locations.append({"file": str(row), "function": "", "purpose": ""})
                changed = True
        if changed:
            normalized["perspectives"]["develop"]["code_locations"] = updated_locations
            notes.append("normalized:develop.code_locations")

    design_decisions = (
        normalized.get("perspectives", {})
        .get("design", {})
        .get("design_decisions")
    )
    if isinstance(design_decisions, list):
        updated_decisions = []
        changed = False
        for row in design_decisions:
            if isinstance(row, str):
                updated_decisions.append({"decision": row, "reason": ""})
                changed = True
            elif isinstance(row, dict):
                row = dict(row)
                row.setdefault("decision", "")
                row.setdefault("reason", "")
                updated_decisions.append(row)
            else:
                updated_decisions.append({"decision": str(row), "reason": ""})
                changed = True
        if changed:
            normalized["perspectives"]["design"]["design_decisions"] = updated_decisions
            notes.append("normalized:design.design_decisions")

    return normalized, notes


def logical_category_from_path(rel_path: Path) -> str:
    category_parts = category_parts_from_source_path(rel_path)
    if category_parts[0] == "features":
        return f"features/{category_parts[1]}"
    return category_parts[0]


def category_parts_from_source_path(rel_path: Path) -> tuple[str, ...]:
    parts = rel_path.parts
    if len(parts) < 1:
        raise ValueError(f"Malformed source path: {rel_path}")
    return parts


def legacy_source_file_from_path(rel_path: Path) -> str:
    return Path(*category_parts_from_source_path(rel_path)).as_posix()


def expected_source_path(knowledge_domain: str, legacy_source_file: str) -> Path:
    return Path(knowledge_domain) / Path(legacy_source_file)


def load_domain_registry(path: Path) -> dict[str, Any]:
    registry = read_json(path)
    domains = registry.get("domains", [])
    if not domains:
        raise ValueError(f"No knowledge domains declared in {path}")
    domain_ids = {row.get("domain_id") for row in domains}
    missing = set(KNOWLEDGE_DOMAINS) - domain_ids
    if missing:
        raise ValueError(f"Missing knowledge domains in registry: {sorted(missing)}")
    return registry


def load_existing_topic_stems(shared_root: Path) -> set[str]:
    topics_root = shared_root / "ai-foundation" / "topics"
    if not topics_root.exists():
        return set()
    return {path.stem for path in topics_root.glob("*.md")}


def load_knowledge_points(shared_root: Path, domain_registry: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for domain in domain_registry["domains"]:
        knowledge_domain = domain["domain_id"]
        source_root = shared_root / domain["source_root"]
        domain_scope_values = domain.get("domain_scope_values", [])
        domain_scope_hint = domain_scope_values[0] if domain_scope_values else domain_scope_from_knowledge_domain(knowledge_domain)
        for path in sorted(source_root.rglob("*.json")):
            if path.name == ".gitkeep":
                continue
            rel_path = path.relative_to(source_root)
            payload = read_json(path)
            legacy_source_file = legacy_source_file_from_path(rel_path)
            entries.append(
                {
                    "source_path": path,
                    "source_file": qualified_source_file(knowledge_domain, legacy_source_file),
                    "legacy_source_file": legacy_source_file,
                    "loaded_source_file": qualified_source_file(knowledge_domain, legacy_source_file),
                    "knowledge_domain_hint": knowledge_domain,
                    "domain_scope_hint": domain_scope_hint,
                    "logical_category": logical_category_from_path(rel_path),
                    "payload": payload,
                }
            )
    return entries


def build_reference_lookup(entries: list[dict[str, Any]]) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        payload = entry["payload"]
        source_id = payload.get("id", "")
        rel = entry["source_file"]
        legacy_rel = entry["legacy_source_file"]
        stem = Path(rel).stem
        name = str(payload.get("name", ""))
        aliases = {
            source_id,
            source_id.lower(),
            source_id.split(":", 1)[-1],
            source_id.split(":", 1)[-1].lower(),
            normalize_token(source_id.split(":", 1)[-1]),
            stem,
            stem.lower(),
            normalize_token(stem),
            name,
            name.lower(),
            normalize_token(name),
            f"{entry['logical_category']}::{stem}",
            legacy_rel,
        }
        for alias in aliases:
            if not alias:
                continue
            if rel not in lookup[alias]:
                lookup[alias].append(rel)
    return lookup


def resolve_related_reference(
    ref_value: str,
    ref_field: str,
    lookup: dict[str, list[str]],
    entries_by_file: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    candidates: list[str] = []
    raw_candidates = expand_reference_aliases(ref_value)
    for candidate in raw_candidates:
        candidates.extend(lookup.get(candidate, []))

    if ref_field == "related_env_vars":
        preferred_prefixes = ("env_vars/",)
    elif ref_field == "related_apis":
        preferred_prefixes = ("apis/",)
    else:
        preferred_prefixes = (
            "features/ascend/",
            "features/vllm/",
            "quantization/",
            "configs/",
            "comm_groups/",
            "moe_comm_types/",
        )

    unique_candidates = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)

    final_candidates = unique_candidates
    preferred_candidates: list[str] = []
    for prefix in preferred_prefixes:
        preferred = [
            candidate
            for candidate in unique_candidates
            if entries_by_file[candidate]["legacy_source_file"].startswith(prefix)
        ]
        if preferred:
            final_candidates = preferred
            preferred_candidates = preferred
            break

    if len(final_candidates) == 1:
        target = entries_by_file[final_candidates[0]]
        return (
            [
                {
                    "ref": ref_value,
                    "target_file": final_candidates[0],
                    "target_id": target["payload"].get("id", ""),
                }
            ],
            [],
            [],
        )

    if len(final_candidates) > 1:
        if ref_field == "related_apis" and not preferred_candidates:
            return [], [{"ref": ref_value, "kind": "api_symbol"}], []
        return (
            [],
            [],
            [f"ambiguous_{ref_field}:{ref_value}->{','.join(final_candidates[:5])}"],
        )

    external_kind = ""
    env_candidate = ref_value.split("=", 1)[0]
    if ref_field == "related_apis":
        external_kind = "api_symbol"
    elif ref_field == "related_env_vars" and re.fullmatch(r"[A-Z0-9_]+", env_candidate):
        external_kind = "env_var"
    elif ref_field == "related_features":
        external_kind = "feature_concept"
    if external_kind:
        return [], [{"ref": ref_value, "kind": external_kind}], []

    return [], [], [f"unresolved_{ref_field}:{ref_value}"]


def repo_roots(workspace_root: Path) -> dict[str, Path]:
    return {
        "vllm": workspace_root / "vllm",
        "vllm-ascend": workspace_root / "vllm-ascend",
    }


def official_url(repo: str, commit: str, path: str) -> str | None:
    template = REPO_BASE_URLS.get(repo)
    if not template or not commit:
        return None
    return template.format(commit=commit, path=path)


def gather_code_evidence(
    payload: dict[str, Any],
    repo_root_map: dict[str, Path],
    commits: dict[str, str],
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    evidence: list[dict[str, Any]] = []
    issues: list[str] = []
    web_evidence: list[dict[str, Any]] = []
    for cp in payload.get("code_paths", []):
        repo = cp.get("repo", "")
        rel_path = cp.get("path", "")
        lines = cp.get("lines", "")
        repo_root = repo_root_map.get(repo)
        if not repo_root:
            issues.append(f"unknown_repo:{repo or 'missing'}")
            continue

        if not is_repo_relative_path(rel_path):
            issues.append(f"invalid_repo_relative_path:{rel_path}")

        local_path = repo_root / rel_path
        exists = local_path.exists()
        evidence_item = {
            "repo": repo,
            "path": rel_path,
            "lines": lines,
            "local_path": str(local_path),
            "exists": exists,
        }
        url = official_url(repo, commits.get(repo, ""), rel_path)
        if url:
            evidence_item["official_url"] = url
            web_evidence.append(
                {
                    "tier": "official",
                    "kind": "repo_code",
                    "repo": repo,
                    "url": url,
                }
            )
        evidence.append(evidence_item)
        if not exists:
            issues.append(f"missing_code_path:{repo}:{rel_path}")
    return evidence, issues, dedupe_dict_rows(web_evidence, "url")


def build_support_corpus(workspace_root: Path) -> list[dict[str, Any]]:
    corpus: list[dict[str, Any]] = []
    corpus_roots = [
        ("vllm", workspace_root / "vllm"),
        ("vllm-ascend", workspace_root / "vllm-ascend"),
        ("workspace", workspace_root),
    ]
    for repo_name, repo_root in corpus_roots:
        if not repo_root.exists():
            continue
        for candidate in SEARCH_ROOT_CANDIDATES:
            root = repo_root / candidate
            if not root.exists():
                continue
            if root.is_file():
                files = [root]
            else:
                files = [path for path in root.rglob("*") if path.is_file()]
            for path in files:
                try:
                    rel = path.relative_to(repo_root).as_posix()
                except ValueError:
                    continue
                kind = "reference"
                if rel.startswith("docs/") or rel.startswith("README"):
                    kind = "docs"
                elif rel.startswith("examples/"):
                    kind = "examples"
                elif rel.startswith("tests/"):
                    kind = "tests"
                elif rel.startswith("benchmarks/"):
                    kind = "benchmarks"
                text_lower = ""
                if kind in {"docs", "examples"} or repo_name == "workspace":
                    try:
                        text_lower = path.read_text(encoding="utf-8", errors="ignore").lower()
                    except OSError:
                        text_lower = ""
                corpus.append(
                    {
                        "repo": repo_name,
                        "root": str(repo_root),
                        "path": rel,
                        "kind": kind,
                        "path_lower": rel.lower(),
                        "text_lower": text_lower,
                    }
                )
    return corpus


def search_local_supporting_files(
    payload: dict[str, Any],
    commits: dict[str, str],
    support_corpus: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    terms = payload_search_terms(payload)

    local_refs: list[dict[str, Any]] = []
    web_refs: list[dict[str, Any]] = []
    issues: list[str] = []

    for term in sorted(terms):
        lower_term = term.lower()
        for item in support_corpus:
            if lower_term not in item["path_lower"] and (
                not item["text_lower"] or lower_term not in item["text_lower"]
            ):
                continue
            row = {
                "repo": item["repo"],
                "path": item["path"],
                "kind": item["kind"],
                "match_term": term,
            }
            local_refs.append(row)
            url = official_url(item["repo"], commits.get(item["repo"], ""), item["path"])
            if url:
                web_refs.append(
                    {
                        "tier": "official",
                        "kind": f"repo_{item['kind']}",
                        "repo": item["repo"],
                        "url": url,
                    }
                )
            if len(local_refs) >= 8:
                break
        if len(local_refs) >= 8:
            break
    return (
        dedupe_dict_rows(local_refs, ("repo", "path")),
        issues,
        dedupe_dict_rows(web_refs, "url"),
    )


def dedupe_dict_rows(rows: list[dict[str, Any]], key: str | tuple[str, ...]) -> list[dict[str, Any]]:
    seen: set[Any] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(key, tuple):
            row_key = tuple(row.get(part) for part in key)
        else:
            row_key = row.get(key)
        if row_key in seen:
            continue
        seen.add(row_key)
        deduped.append(row)
    return deduped


def fetch_url_metadata(url: str, timeout_seconds: int = 5) -> dict[str, Any]:
    fetch_url = url
    match = re.match(r"^https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)$", url)
    if match:
        owner, repo, commit, path = match.groups()
        fetch_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{commit}/{path}"

    request = Request(
        fetch_url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; vllm-workspace-validator/1.0)"
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            content_type = response.headers.get("Content-Type", "")
            body = response.read(32768)
            text = body.decode("utf-8", errors="ignore")
            title_match = re.search(r"<title>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
            title = ""
            if title_match:
                title = re.sub(r"\s+", " ", title_match.group(1)).strip()
            return {
                "status": "ok",
                "http_status": getattr(response, "status", 200),
                "content_type": content_type,
                "final_url": response.geturl(),
                "requested_url": url,
                "fetch_url": fetch_url,
                "title": title,
                "fetched_at": utcnow(),
            }
    except HTTPError as exc:
        return {
            "status": "http_error",
            "http_status": exc.code,
            "error": str(exc),
            "final_url": url,
            "fetch_url": fetch_url,
            "fetched_at": utcnow(),
        }
    except URLError as exc:
        return {
            "status": "url_error",
            "error": str(exc),
            "final_url": url,
            "fetch_url": fetch_url,
            "fetched_at": utcnow(),
        }


def load_web_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"generated_at": utcnow(), "urls": {}}
    try:
        return read_json(path)
    except json.JSONDecodeError:
        return {"generated_at": utcnow(), "urls": {}}


def enrich_web_evidence(
    records: list[dict[str, Any]],
    cache_path: Path,
    fetch_enabled: bool,
    fetch_limit: int,
) -> dict[str, Any]:
    cache = load_web_cache(cache_path)
    existing_urls = cache.setdefault("urls", {})
    unique_urls = sorted(
        {
            row["url"]
            for record in records
            for row in record.get("web_evidence", [])
            if row.get("url")
        }
    )
    urls = {url: existing_urls.get(url, {}) for url in unique_urls}
    cache["urls"] = urls

    fetch_targets: list[str] = []
    for url in unique_urls:
        should_fetch = False
        if fetch_enabled:
            existing = urls.get(url, {})
            existing_status = existing.get("status")
            existing_http_status = existing.get("http_status")
            if existing_status in {None, "not_fetched", "url_error"} or (
                existing_status == "http_error" and existing_http_status == 429
            ):
                should_fetch = True
        if should_fetch:
            fetch_targets.append(url)
        elif url not in urls:
            urls[url] = {"status": "not_fetched"}

    if fetch_limit >= 0:
        fetch_targets = fetch_targets[:fetch_limit]

    fetched_this_run = 0
    if fetch_targets:
        max_workers = min(8, len(fetch_targets))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(fetch_url_metadata, url): url
                for url in fetch_targets
            }
            for future in as_completed(future_map):
                url = future_map[future]
                urls[url] = future.result()
                fetched_this_run += 1

    for record in records:
        for row in record.get("web_evidence", []):
            url = row.get("url")
            if url and url in urls:
                row["fetch"] = urls[url]

    cache["generated_at"] = utcnow()
    cache["url_count"] = len(urls)
    cache["fetched_this_run"] = fetched_this_run
    return cache


def needs_platform_doc_review(logical_category: str, payload: dict[str, Any]) -> bool:
    if logical_category == "devices":
        return True
    if logical_category not in {"operators", "env_vars"}:
        return False
    text = " ".join(
        [
            logical_category,
            str(payload.get("name", "")),
            str(payload.get("summary", "")),
            str(payload.get("id", "")),
        ]
    ).lower()
    keywords = (
        "ascend",
        "cann",
        "acl",
        "soc",
        "hccl",
        "npu",
        "torch_npu",
        "msmonitor",
    )
    return any(keyword in text for keyword in keywords)


def supplemental_web_evidence(payload: dict[str, Any]) -> list[dict[str, Any]]:
    source_id = str(payload.get("id", "")).lower()
    urls = OFFICIAL_PLATFORM_DOCS.get(source_id, [])
    return [
        {
            "tier": "official",
            "kind": "platform_doc",
            "repo": "platform",
            "url": url,
        }
        for url in urls
    ]


def derive_task_types(logical_category: str, payload: dict[str, Any]) -> list[str]:
    rel = logical_category
    text = " ".join(
        [
            str(payload.get("name", "")),
            str(payload.get("summary", "")),
            str(payload.get("id", "")),
        ]
    ).lower()
    tasks: set[str] = set()

    if rel.startswith("features/"):
        tasks.update({"deployment", "debugging", "performance_analysis", "design_analysis"})
        if rel == "features/vllm":
            tasks.update({"upstream_sync", "release_analysis"})
    elif rel == "apis":
        tasks.update({"model_adaptation", "debugging", "design_analysis", "upstream_sync"})
    elif rel == "operators":
        tasks.update({"op_development", "performance_analysis", "design_analysis", "debugging"})
    elif rel == "env_vars":
        tasks.update({"env_bootstrap", "deployment", "debugging", "performance_analysis"})
    elif rel == "configs":
        tasks.update({"deployment", "debugging", "performance_analysis", "design_analysis"})
    elif rel == "quantization":
        tasks.update({"deployment", "model_adaptation", "performance_analysis", "design_analysis"})
    elif rel in {"comm_groups", "moe_comm_types"}:
        tasks.update({"debugging", "op_development", "performance_analysis", "design_analysis"})
    elif rel == "devices":
        tasks.update({"env_bootstrap", "deployment", "performance_analysis", "design_analysis"})

    if any(keyword in text for keyword in ("release", "compatibility", "upstream", "api", "platform")):
        tasks.update({"upstream_sync", "release_analysis"})
    if any(keyword in text for keyword in ("model", "adapter", "runner", "worker")):
        tasks.add("model_adaptation")
    if any(keyword in text for keyword in ("graph", "parallel", "schedule", "attention", "kv", "precision", "quant")):
        tasks.add("design_analysis")
    if any(keyword in text for keyword in ("error", "crash", "log", "debug")):
        tasks.add("debugging")

    ordered = [task for task in TASK_TYPES if task in tasks]
    return ordered


def derive_consumer_skills(task_types: list[str], logical_category: str, payload: dict[str, Any]) -> list[str]:
    skills: set[str] = set()
    text = " ".join(
        [
            logical_category,
            str(payload.get("name", "")),
            str(payload.get("summary", "")),
            str(payload.get("id", "")),
        ]
    ).lower()

    if "deployment" in task_types:
        skills.update({"deployment-assistant", "compatibility-checker"})
    if "env_bootstrap" in task_types:
        skills.update({"developer-assistant", "env-bootstrap"})
    if "debugging" in task_types:
        skills.update({"developer-assistant", "debug-assistant", "log-analyzer", "crash-rooter"})
    if "model_adaptation" in task_types:
        skills.update({"developer-assistant", "model-adapter", "custom-model-integrator"})
    if "upstream_sync" in task_types:
        skills.update(
            {
                "developer-assistant",
                "sync-coordinator",
                "compatibility-checker",
                "repo-state-auditor",
                "knowledge-index-maintainer",
                "ci-gatekeeper",
            }
        )
    if "release_analysis" in task_types:
        skills.update(
            {
                "developer-assistant",
                "release-assistant",
                "release-commit-analyzer",
                "release-notes-composer",
                "docs-compliance-checker",
                "ci-gatekeeper",
            }
        )
    if "op_development" in task_types:
        skills.update(
            {
                "developer-assistant",
                "op-developer",
                "graph-analyzer",
                "precision-validator",
                "ci-gatekeeper",
            }
        )
    if "performance_analysis" in task_types:
        skills.update({"developer-assistant", "perf-assistant", "perf-hunter", "test-matrix-planner"})
    if "design_analysis" in task_types:
        skills.add("developer-assistant")
        if any(keyword in text for keyword in ("graph", "acl", "xlite")):
            skills.add("graph-analyzer")
        if any(keyword in text for keyword in ("parallel", "tp", "dp", "cp", "eplb", "mc2", "comm_group")):
            skills.add("parallelism-planner")
        if any(keyword in text for keyword in ("scheduler", "batch", "pd_", "disaggregation", "balance")):
            skills.add("scheduler-feature-designer")
        if any(keyword in text for keyword in ("attention", "kv", "rope", "mla", "paged", "sfa")):
            skills.add("attention-kv-designer")
        if any(keyword in text for keyword in ("model", "runner", "worker", "api", "adaptor", "integrator")):
            skills.add("custom-model-integrator")
        if any(keyword in text for keyword in ("quant", "precision", "w8a8", "w4a", "int4", "int8")):
            skills.add("precision-validator")

    if "graph-analyzer" not in skills and logical_category == "operators":
        skills.add("graph-analyzer")
    if "parallelism-planner" not in skills and logical_category in {"comm_groups", "moe_comm_types"}:
        skills.add("parallelism-planner")
    if "precision-validator" not in skills and logical_category == "quantization":
        skills.add("precision-validator")

    return [skill for skill in sorted(skills) if skill in ALL_SKILLS]


def derive_migration_target(logical_category: str, task_types: list[str]) -> str:
    if logical_category == "devices":
        return "ascend-foundation"
    if logical_category == "env_vars" or logical_category == "configs":
        return "deployment-config"
    if logical_category == "apis" and "model_adaptation" in task_types:
        return "model-adaptation"
    if logical_category == "apis":
        return "vllm-upstream"
    if logical_category == "features/vllm":
        return "vllm-upstream"
    if logical_category in {"operators", "comm_groups", "moe_comm_types", "quantization", "features/ascend"}:
        return "vllm-ascend-core"
    return "vllm-ascend-core"


def derive_utility_verdict(
    logical_category: str,
    payload: dict[str, Any],
    existing_topic_stems: set[str],
) -> str:
    stem = Path(payload.get("name", "") or "").stem or payload.get("name", "")
    normalized_stem = normalize_token(stem)
    known_topic_hit = normalized_stem in existing_topic_stems or stem in existing_topic_stems

    if logical_category in {"apis", "operators", "comm_groups", "moe_comm_types"}:
        return "reference_only"
    if logical_category in {"env_vars", "configs"}:
        return "merge_required"
    if known_topic_hit:
        return "merge_required"
    if logical_category in {"quantization", "devices", "features/vllm", "features/ascend"}:
        return "direct_skill_topic"
    return "reference_only"


def code_repos_for_payload(payload: dict[str, Any]) -> list[str]:
    return sorted(
        {
            row.get("repo")
            for row in payload.get("code_paths", [])
            if row.get("repo") in REPO_BASE_URLS
        }
    )


def derive_implementation_repos(
    payload: dict[str, Any],
    resolved_refs: dict[str, list[dict[str, str]]],
    entries_by_file: dict[str, dict[str, Any]],
) -> list[str]:
    repos = set(code_repos_for_payload(payload))
    for rows in resolved_refs.values():
        for row in rows:
            target = entries_by_file.get(row["target_file"])
            if not target:
                continue
            repos.update(code_repos_for_payload(target["payload"]))
    return [repo for repo in ("vllm", "vllm-ascend") if repo in repos]


def requires_both_domain(
    logical_category: str,
    payload: dict[str, Any],
    resolved_refs: dict[str, list[dict[str, str]]],
    entries_by_file: dict[str, dict[str, Any]],
) -> bool:
    if logical_category == "features/vllm":
        return False

    text = " ".join(
        [
            logical_category,
            str(payload.get("id", "")),
            str(payload.get("name", "")),
            str(payload.get("summary", "")),
        ]
    ).lower()
    bridge_keywords = (
        "runner",
        "engine",
        "worker",
        "attention",
        "kv",
        "paged",
        "cache",
        "scheduler",
        "parallel",
        "quant",
        "sampling",
        "speculative",
    )
    if not any(keyword in text for keyword in bridge_keywords):
        return False

    for rows in resolved_refs.values():
        for row in rows:
            target = entries_by_file.get(row["target_file"])
            if not target:
                continue
            target_repos = set(code_repos_for_payload(target["payload"]))
            if target_repos == {"vllm"}:
                return True
    return False


def derive_domain_scope(
    logical_category: str,
    payload: dict[str, Any],
    implementation_repos: list[str],
    resolved_refs: dict[str, list[dict[str, str]]],
    entries_by_file: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    repos = set(implementation_repos)
    if repos == {"vllm", "vllm-ascend"}:
        return (
            "both",
            "Core implementation and resolved references span upstream vLLM and vLLM-Ascend.",
        )
    if logical_category == "features/vllm" or repos == {"vllm"}:
        return (
            "vllm",
            "Upstream vLLM code paths define the core fact and Ascend only consumes the behavior.",
        )
    if repos == {"vllm-ascend"}:
        if requires_both_domain(logical_category, payload, resolved_refs, entries_by_file):
            return (
                "both",
                "The entry is implemented on vLLM-Ascend but its meaning depends on upstream vLLM surfaces.",
            )
        return (
            "vllm-ascend",
            "The fact depends on Ascend-specific code paths, platform behavior, or adaptation layers.",
        )
    if logical_category == "features/ascend":
        return (
            "vllm-ascend",
            "Ascend feature knowledge is maintained as vLLM-Ascend-specific source-of-truth.",
        )
    return (
        "vllm-ascend",
        "The entry is maintained with the Ascend adaptation layer as the canonical implementation surface.",
    )


def canonicalize_source_payload(
    payload: dict[str, Any],
    domain_scope: str,
    knowledge_domain: str,
    implementation_repos: list[str],
    domain_reason: str,
) -> tuple[dict[str, Any], str]:
    canonical = json.loads(json.dumps(payload))
    canonical["domain_scope"] = domain_scope
    canonical["knowledge_domain"] = knowledge_domain
    canonical["implementation_repos"] = implementation_repos
    canonical["domain_reason"] = domain_reason
    source_hash = compute_source_hash(canonical)
    canonical["source_hash"] = source_hash
    return canonical, source_hash


def summarize_reasoning(
    status: str,
    factual_verdict: str,
    code_evidence: list[dict[str, Any]],
    repo_evidence: list[dict[str, Any]],
    open_gaps: list[str],
) -> str:
    if status == "rewrite_required":
        return "Source entry requires normalization before factual review can be trusted."
    if status == "reject":
        return "Entry lacks enough local code evidence to support import into `_shared`."
    parts = []
    if factual_verdict == "code_doc_aligned":
        parts.append("Local code paths and repo-local docs/tests/examples align.")
    elif factual_verdict == "code_web_aligned":
        parts.append("Local code paths align with official repo web references.")
    else:
        parts.append("Local code paths confirm the core fact set.")
    parts.append(f"Code refs: {len(code_evidence)}")
    parts.append(f"Repo refs: {len(repo_evidence)}")
    if open_gaps:
        parts.append(f"Open gaps: {len(open_gaps)}")
    return " ".join(parts)


def collect_schema_errors(validator: Draft7Validator, payload: dict[str, Any]) -> list[str]:
    errors = []
    for error in sorted(validator.iter_errors(payload), key=lambda item: list(item.path)):
        path = ".".join(str(part) for part in error.path) or "<root>"
        errors.append(f"schema:{path}:{error.message}")
    return errors


def validate_non_empty_perspectives(payload: dict[str, Any]) -> list[str]:
    issues = []
    perspectives = payload.get("perspectives", {})
    if not isinstance(perspectives, dict):
        return ["perspectives:not_object"]
    for perspective in REQUIRED_PERSPECTIVES:
        value = perspectives.get(perspective)
        if value in ({}, None, [], ""):
            issues.append(f"perspective_empty:{perspective}")
    return issues


def expected_counts_from_stats(statistics: dict[str, Any]) -> dict[str, int]:
    raw = statistics.get("by_category", {})
    mapped = {
        "features/vllm": int(raw.get("vllm_features", 0)),
        "features/ascend": int(raw.get("vllm_ascend_features", 0)),
        "operators": int(raw.get("npu_operators", 0)),
        "apis": int(raw.get("apis", 0)),
        "env_vars": int(raw.get("env_vars", 0)),
        "configs": int(raw.get("configs", 0)),
        "quantization": int(raw.get("quantization", 0)),
        "comm_groups": int(raw.get("comm_groups", 0)),
        "moe_comm_types": int(raw.get("moe_comm_types", 0)),
        "devices": int(raw.get("devices", 0)),
    }
    return mapped


def count_entries(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for entry in entries:
        counts[entry["logical_category"]] += 1
    return dict(sorted(counts.items()))


def build_batch_plan(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {
        "feature_config_design": {
            "task_family": "deployment+performance+design",
            "categories": {"features/vllm", "features/ascend", "configs", "quantization", "devices"},
            "chunk_size": 18,
        },
        "api_architecture": {
            "task_family": "model_adaptation+design+debugging",
            "categories": {"apis"},
            "chunk_size": 10,
        },
        "operator_perf": {
            "task_family": "op_development+performance+design",
            "categories": {"operators", "comm_groups", "moe_comm_types"},
            "chunk_size": 10,
        },
        "env_runtime": {
            "task_family": "env_bootstrap+deployment+debugging",
            "categories": {"env_vars"},
            "chunk_size": 12,
        },
    }

    grouped_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for name, config in groups.items():
            if record["category"] in config["categories"]:
                grouped_records[name].append(record)
                break

    batches = []
    batch_counter = 1
    for group_name, config in groups.items():
        items = sorted(
            grouped_records.get(group_name, []),
            key=lambda row: row.get("legacy_source_file", row["source_file"]),
        )
        chunk_size = config["chunk_size"]
        for idx in range(0, len(items), chunk_size):
            chunk = items[idx : idx + chunk_size]
            batch_id = f"B{batch_counter:03d}"
            batches.append(
                {
                    "batch_id": batch_id,
                    "name": f"{group_name}-{idx // chunk_size + 1}",
                    "task_family": config["task_family"],
                    "categories": sorted({item["category"] for item in chunk}),
                    "status": "pending",
                    "entry_count": len(chunk),
                    "completed_entries": 0,
                    "pending_entries": len(chunk),
                    "source_files": [item["source_file"] for item in chunk],
                    "source_ids": [item["source_id"] for item in chunk],
                    "design_analysis_entries": [
                        item["source_id"] for item in chunk if "design_analysis" in item["task_types"]
                    ],
                }
            )
            for item in chunk:
                item["batch_id"] = batch_id
            batch_counter += 1

    supplemental_batch = {
        "batch_id": f"B{batch_counter:03d}",
        "name": "relations_matrices_consistency",
        "task_family": "global_consistency",
        "categories": ["relations", "matrices"],
        "status": "pending",
        "entry_count": 0,
        "completed_entries": 0,
        "pending_entries": 0,
        "source_files": [],
        "source_ids": [],
        "design_analysis_entries": [],
        "supporting_files": [
            "relations/combinations.json",
            "relations/dependencies.json",
            "relations/references.json",
            "matrices/compatibility.json",
            "matrices/performance.json",
            "matrices/recommendations.json",
        ],
    }
    batches.append(supplemental_batch)

    return {
        "generated_at": utcnow(),
        "current_batch_id": batches[0]["batch_id"] if batches else None,
        "batches": batches,
    }


def finalize_batch_state(batch_plan: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    final_state = json.loads(json.dumps(batch_plan))
    records_by_file = {record["source_file"]: record for record in records}
    current_batch_id = None
    for batch in final_state["batches"]:
        if batch["entry_count"] == 0:
            batch["status"] = "ready"
            continue
        chunk = [records_by_file[source_file] for source_file in batch["source_files"]]
        status_counts = Counter(item["status"] for item in chunk)
        if status_counts.get("rewrite_required") or status_counts.get("reject"):
            batch_status = "needs_follow_up"
        elif status_counts.get("validated_with_gap"):
            batch_status = "needs_evidence_follow_up"
        else:
            batch_status = "ready"
        batch["status"] = batch_status
        batch["completed_entries"] = len([item for item in chunk if item["status"] == "validated"])
        batch["pending_entries"] = len([item for item in chunk if item["status"] != "validated"])
        if current_batch_id is None and batch_status != "ready":
            current_batch_id = batch["batch_id"]
    final_state["current_batch_id"] = current_batch_id
    final_state["generated_at"] = utcnow()
    return final_state


def load_previous_manifest_records(provenance_root: Path) -> dict[str, dict[str, Any]]:
    manifest_path = provenance_root / "verification_manifest.json"
    if not manifest_path.exists():
        return {}
    payload = read_json(manifest_path)
    return {
        row["source_file"]: row
        for row in payload.get("entries", [])
        if row.get("source_file")
    }


def load_execution_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def build_execution_state(
    batch_plan: dict[str, Any],
    entries: list[dict[str, Any]],
    source_roots: dict[str, Path],
    provenance_root: Path,
    generated_root: Path,
    snapshot_hash: str,
) -> dict[str, Any]:
    source_files = [entry["source_file"] for entry in entries]
    return {
        "phase": "validation",
        "current_domain": entries[0]["knowledge_domain"] if entries else None,
        "current_batch_id": batch_plan.get("current_batch_id"),
        "current_entry_cursor": 0,
        "completed_entry_ids": [],
        "pending_entry_ids": [entry["payload"].get("id", "") for entry in entries],
        "domain_adjudication_completed": False,
        "source_roots": {
            knowledge_domain: str(path)
            for knowledge_domain, path in sorted(source_roots.items())
        },
        "provenance_root": str(provenance_root),
        "generated_root": str(generated_root),
        "last_successful_step": "initialized",
        "last_source_snapshot_hash": snapshot_hash,
        "source_files": source_files,
        "updated_at": utcnow(),
    }


def update_execution_state_for_entry(
    state: dict[str, Any],
    record: dict[str, Any],
    index: int,
    total_count: int,
) -> None:
    state["current_entry_cursor"] = index + 1
    state["current_domain"] = record["knowledge_domain"]
    state["current_batch_id"] = record["batch_id"]
    completed = state.setdefault("completed_entry_ids", [])
    if record["source_id"] not in completed:
        completed.append(record["source_id"])
    pending = [source_id for source_id in state.get("pending_entry_ids", []) if source_id != record["source_id"]]
    state["pending_entry_ids"] = pending
    state["domain_adjudication_completed"] = index + 1 >= total_count
    state["last_successful_step"] = f"evaluated:{record['source_id']}"
    state["updated_at"] = utcnow()


def finalize_execution_state(
    state: dict[str, Any],
    batch_state: dict[str, Any],
    snapshot_hash: str,
) -> None:
    state["phase"] = "completed"
    state["current_domain"] = None
    state["current_batch_id"] = batch_state.get("current_batch_id")
    state["current_entry_cursor"] = len(state.get("completed_entry_ids", []))
    state["domain_adjudication_completed"] = True
    state["last_successful_step"] = "artifacts_written"
    state["last_source_snapshot_hash"] = snapshot_hash
    state["updated_at"] = utcnow()


def build_handoff(records: list[dict[str, Any]], batch_state: dict[str, Any], report: dict[str, Any]) -> str:
    status_counts = Counter(record["status"] for record in records)
    recent_ready = [batch["batch_id"] for batch in batch_state["batches"] if batch["status"] == "ready"][:5]
    next_batch = batch_state.get("current_batch_id") or "none"

    blockers = [
        f"- `{record['source_id']}`: {', '.join(record['open_gaps'][:3])}"
        for record in records
        if record["status"] in {"rewrite_required", "reject"}
    ][:20]
    evidence_gaps = [
        f"- `{record['source_id']}`: {', '.join(record['open_gaps'][:3])}"
        for record in records
        if record["status"] == "validated_with_gap"
    ][:20]
    design_gaps = [
        f"- `{record['source_id']}` -> `{', '.join(record['consumer_skills'])}`"
        for record in records
        if "design_analysis" in record["task_types"] and record["status"] != "validated"
    ][:20]
    normalization_counts = Counter(
        note for record in records for note in record.get("normalization_notes", [])
    )

    lines = [
        "# Verification Handoff",
        "",
        f"- Generated at: `{utcnow()}`",
        f"- Total entries: `{report['summary']['total_entries']}`",
        f"- Current batch: `{next_batch}`",
        f"- Ready batches: `{', '.join(recent_ready) if recent_ready else 'none'}`",
        "",
        "## Status Counts",
    ]
    for status in STATUS_ORDER:
        lines.append(f"- `{status}`: `{status_counts.get(status, 0)}`")
    lines.extend(
        [
            "",
            "## Next Batch",
            f"- `{next_batch}`",
            "",
            "## Top Blockers",
            *(blockers or ["- none"]),
            "",
            "## Top Evidence Gaps",
            *(evidence_gaps or ["- none"]),
            "",
            "## Normalization Notes",
            *(
                [f"- `{note}`: `{count}`" for note, count in normalization_counts.most_common(10)]
                or ["- none"]
            ),
            "",
            "## Top Design-Analysis Gaps",
            *(design_gaps or ["- none"]),
            "",
            "## Migration Strategy",
            "- Canonical source is domain-owned under `_shared/vllm-upstream/`, `_shared/vllm-ascend-core/`, and `_shared/integration-core/`.",
            "- Governance contracts, provenance, and resume state live under `_shared/knowledge-governance/`.",
            "- Unified generated retrieval artifacts stay under `_shared/knowledge-governance/generated/`.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_scenario_coverage(
    shared_root: Path,
    eligible_entries: list[dict[str, Any]],
    task_skill_payload: dict[str, Any],
) -> dict[str, Any]:
    available_source_ids = {
        normalize_source_id(entry["source_id"])
        for entry in eligible_entries
    }
    available_tasks = set(task_skill_payload["task_types"])
    available_task_skills = {
        task: set(payload["skills"])
        for task, payload in task_skill_payload["task_types"].items()
    }

    scenario_rows = []
    by_task: dict[str, list[str]] = defaultdict(list)
    by_composer: dict[str, list[str]] = defaultdict(list)
    by_atomic: dict[str, list[str]] = defaultdict(list)

    for scenario in SCENARIO_LIBRARY:
        issues: list[str] = []
        task_type = scenario["task_type"]
        entry_skill = scenario["entry_skill"]
        composer_skill = scenario["composer_skill"]
        atomic_skills = scenario["atomic_skills"]

        if task_type not in available_tasks:
            issues.append(f"missing_task_type:{task_type}")
        if entry_skill not in ALL_SKILLS:
            issues.append(f"unknown_entry_skill:{entry_skill}")
        if composer_skill and composer_skill not in ALL_SKILLS:
            issues.append(f"unknown_composer_skill:{composer_skill}")

        task_skill_set = available_task_skills.get(task_type, set())
        for skill in atomic_skills:
            if skill not in ALL_SKILLS:
                issues.append(f"unknown_atomic_skill:{skill}")
            elif skill not in task_skill_set:
                issues.append(f"task_missing_skill:{task_type}:{skill}")

        missing_docs = [
            doc_rel
            for doc_rel in scenario["required_docs"]
            if not (shared_root / doc_rel).exists()
        ]
        issues.extend([f"missing_doc:{doc_rel}" for doc_rel in missing_docs])

        missing_evidence = [
            source_id
            for source_id in scenario["evidence_entry_ids"]
            if normalize_source_id(source_id) not in available_source_ids
        ]
        issues.extend([f"missing_evidence:{source_id}" for source_id in missing_evidence])

        scenario_row = {
            **scenario,
            "status": "covered" if not issues else "gap",
            "issues": issues,
            "matched_atomic_skills": {
                skill: sorted(task_skill_payload["task_types"].get(task_type, {}).get("skills", {}).get(skill, []))[:12]
                for skill in atomic_skills
            },
        }
        scenario_rows.append(scenario_row)
        by_task[task_type].append(scenario["id"])
        if composer_skill:
            by_composer[composer_skill].append(scenario["id"])
        for skill in atomic_skills:
            by_atomic[skill].append(scenario["id"])

    required_composers = (
        "model-adapter",
        "sync-coordinator",
        "debug-assistant",
        "release-assistant",
        "op-developer",
        "perf-assistant",
    )
    required_atomic = (
        "env-bootstrap",
        "compatibility-checker",
        "repo-state-auditor",
        "log-analyzer",
        "crash-rooter",
        "perf-hunter",
        "graph-analyzer",
        "parallelism-planner",
        "scheduler-feature-designer",
        "attention-kv-designer",
        "custom-model-integrator",
        "precision-validator",
        "release-commit-analyzer",
        "release-notes-composer",
        "docs-compliance-checker",
        "test-matrix-planner",
        "ci-gatekeeper",
        "knowledge-index-maintainer",
    )

    return {
        "generated_at": utcnow(),
        "scenario_count": len(scenario_rows),
        "status": "pass" if all(row["status"] == "covered" for row in scenario_rows) else "fail",
        "task_type_coverage": {
            task: {
                "scenario_count": len(by_task.get(task, [])),
                "scenario_ids": by_task.get(task, []),
            }
            for task in TASK_TYPES
        },
        "composer_coverage": {
            skill: {
                "scenario_count": len(by_composer.get(skill, [])),
                "scenario_ids": by_composer.get(skill, []),
                "meets_minimum_two": len(by_composer.get(skill, [])) >= 2,
            }
            for skill in required_composers
        },
        "atomic_skill_coverage": {
            skill: {
                "scenario_count": len(by_atomic.get(skill, [])),
                "scenario_ids": by_atomic.get(skill, []),
                "covered": len(by_atomic.get(skill, [])) >= 1,
            }
            for skill in required_atomic
        },
        "scenarios": scenario_rows,
    }


def export_shared_artifacts(
    shared_root: Path,
    generated_root: Path,
    import_manifest: dict[str, Any],
) -> dict[str, str]:
    generated_root.mkdir(parents=True, exist_ok=True)

    eligible_entries = [
        entry
        for entry in import_manifest["entries"]
        if entry["status"] in {"validated", "validated_with_gap"}
    ]
    manifest_payload = {
        "generated_at": utcnow(),
        "source_entry_count": import_manifest["source_entry_count"],
        "eligible_entry_count": len(eligible_entries),
        "entries": eligible_entries,
    }

    search_terms: dict[str, list[str]] = defaultdict(list)
    by_task: dict[str, list[str]] = defaultdict(list)
    by_skill: dict[str, list[str]] = defaultdict(list)
    by_category: dict[str, list[str]] = defaultdict(list)
    by_domain: dict[str, list[str]] = defaultdict(list)
    by_knowledge_domain: dict[str, list[str]] = defaultdict(list)
    design_rows = []

    for entry in eligible_entries:
        source_id = entry["source_id"]
        normalized_id = entry.get("normalized_id", source_id)
        by_category[entry["category"]].append(source_id)
        by_domain[entry["domain_scope"]].append(source_id)
        by_knowledge_domain[entry["knowledge_domain"]].append(source_id)
        for task in entry["task_types"]:
            by_task[task].append(source_id)
        for skill in entry["consumer_skills"]:
            by_skill[skill].append(source_id)
        for term in {
            entry["source_id"],
            normalized_id,
            entry["name"],
            normalize_token(entry["name"]),
            Path(entry["source_file"]).stem,
        }:
            if term:
                search_terms[term].append(source_id)
        if "design_analysis" in entry["task_types"]:
            design_rows.append(
                {
                    "source_id": source_id,
                    "normalized_id": normalized_id,
                    "name": entry["name"],
                    "category": entry["category"],
                    "design_skills": [
                        skill
                        for skill in entry["consumer_skills"]
                        if skill
                        in {
                            "graph-analyzer",
                            "parallelism-planner",
                            "scheduler-feature-designer",
                            "attention-kv-designer",
                            "custom-model-integrator",
                            "precision-validator",
                            "model-adapter",
                            "op-developer",
                            "perf-assistant",
                        }
                    ],
                    "migration_target": entry["migration_target"],
                    "domain_scope": entry["domain_scope"],
                    "knowledge_domain": entry["knowledge_domain"],
                    "status": entry["status"],
                }
            )

    search_payload = {
        "generated_at": utcnow(),
        "terms": {key: sorted(set(value)) for key, value in sorted(search_terms.items())},
        "by_task_type": {key: sorted(set(value)) for key, value in sorted(by_task.items())},
        "by_skill": {key: sorted(set(value)) for key, value in sorted(by_skill.items())},
        "by_category": {key: sorted(set(value)) for key, value in sorted(by_category.items())},
        "by_domain_scope": {key: sorted(set(value)) for key, value in sorted(by_domain.items())},
        "by_knowledge_domain": {
            key: sorted(set(value))
            for key, value in sorted(by_knowledge_domain.items())
        },
    }
    design_payload = {
        "generated_at": utcnow(),
        "entry_count": len(design_rows),
        "entries": sorted(design_rows, key=lambda row: row["source_id"]),
    }
    task_skill_payload = {
        "generated_at": utcnow(),
        "task_types": {
            task: {
                "skills": {
                    skill: sorted(
                        {
                            entry["source_id"]
                            for entry in eligible_entries
                            if task in entry["task_types"] and skill in entry["consumer_skills"]
                        }
                    )
                    for skill in sorted(ALL_SKILLS)
                    if any(task in entry["task_types"] and skill in entry["consumer_skills"] for entry in eligible_entries)
                }
            }
            for task in TASK_TYPES
            if any(task in entry["task_types"] for entry in eligible_entries)
        },
    }
    scenario_payload = build_scenario_coverage(shared_root, eligible_entries, task_skill_payload)
    report_payload = {
        "generated_at": utcnow(),
        "source_entry_count": import_manifest["source_entry_count"],
        "eligible_entry_count": len(eligible_entries),
        "validated_count": len([entry for entry in eligible_entries if entry["status"] == "validated"]),
        "validated_with_gap_count": len([entry for entry in eligible_entries if entry["status"] == "validated_with_gap"]),
        "normalization_counts": dict(
            Counter(note for entry in eligible_entries for note in entry.get("normalization_notes", []))
        ),
        "migration_targets": dict(Counter(entry["migration_target"] for entry in eligible_entries)),
        "domain_scopes": dict(Counter(entry["domain_scope"] for entry in eligible_entries)),
        "knowledge_domains": dict(Counter(entry["knowledge_domain"] for entry in eligible_entries)),
        "task_coverage": {
            task: len(
                {
                    entry["source_id"]
                    for entry in eligible_entries
                    if task in entry["task_types"]
                }
            )
            for task in TASK_TYPES
        },
        "scenario_coverage": {
            "scenario_count": scenario_payload["scenario_count"],
            "status": scenario_payload["status"],
            "composer_coverage": {
                skill: payload["scenario_count"]
                for skill, payload in scenario_payload["composer_coverage"].items()
            },
            "atomic_skill_coverage": {
                skill: payload["scenario_count"]
                for skill, payload in scenario_payload["atomic_skill_coverage"].items()
            },
        },
    }
    domain_payload = {
        "generated_at": utcnow(),
        "domain_scope_index": {
            scope: sorted(set(by_domain.get(scope, [])))
            for scope in DOMAIN_SCOPES
        },
        "knowledge_domain_index": {
            knowledge_domain: sorted(set(by_knowledge_domain.get(knowledge_domain, [])))
            for knowledge_domain in KNOWLEDGE_DOMAINS
        },
        "implementation_repo_index": {
            repo: sorted(
                {
                    entry["source_id"]
                    for entry in eligible_entries
                    if repo in entry.get("implementation_repos", [])
                }
            )
            for repo in ("vllm", "vllm-ascend")
        },
    }

    outputs = {
        "manifest": generated_root / EXPORT_FILES["manifest"],
        "search_index": generated_root / EXPORT_FILES["search_index"],
        "design_index": generated_root / EXPORT_FILES["design_index"],
        "task_skill_index": generated_root / EXPORT_FILES["task_skill_index"],
        "scenario_coverage": generated_root / EXPORT_FILES["scenario_coverage"],
        "domain_index": generated_root / EXPORT_FILES["domain_index"],
        "report": generated_root / EXPORT_FILES["report"],
    }
    write_json(outputs["manifest"], manifest_payload)
    write_json(outputs["search_index"], search_payload)
    write_json(outputs["design_index"], design_payload)
    write_json(outputs["task_skill_index"], task_skill_payload)
    write_json(outputs["scenario_coverage"], scenario_payload)
    write_json(outputs["domain_index"], domain_payload)
    write_json(outputs["report"], report_payload)
    return {key: str(path) for key, path in outputs.items()}


def validate_supporting_artifacts(governance_source_root: Path) -> dict[str, Any]:
    paths = [
        governance_source_root / "relations" / "combinations.json",
        governance_source_root / "relations" / "dependencies.json",
        governance_source_root / "relations" / "references.json",
        governance_source_root / "matrices" / "compatibility.json",
        governance_source_root / "matrices" / "performance.json",
        governance_source_root / "matrices" / "recommendations.json",
    ]
    missing = [str(path.relative_to(governance_source_root)) for path in paths if not path.exists()]
    return {
        "checked_files": [str(path.relative_to(governance_source_root)) for path in paths],
        "missing_files": missing,
        "status": "pass" if not missing else "fail",
    }


def sync_canonical_source_tree(
    source_roots: dict[str, Path],
    entries: list[dict[str, Any]],
) -> None:
    expected_paths: set[Path] = set()
    loaded_paths: set[Path] = set()
    for entry in entries:
        loaded_path = entry["source_path"]
        target_path = source_roots[entry["knowledge_domain"]] / entry["legacy_source_file"]
        expected_paths.add(target_path)
        loaded_paths.add(loaded_path)
        write_json(target_path, entry["payload"])
        if loaded_path != target_path and loaded_path.exists():
            loaded_path.unlink()

    for knowledge_root in source_roots.values():
        for existing in sorted(knowledge_root.rglob("*.json")):
            if existing not in expected_paths and existing not in loaded_paths:
                existing.unlink()
        for directory in sorted(knowledge_root.rglob("*"), reverse=True):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()


def prepare_entries(
    entries: list[dict[str, Any]],
    existing_topic_stems: set[str],
) -> list[dict[str, Any]]:
    lookup = build_reference_lookup(entries)
    entries_by_file = {entry["source_file"]: entry for entry in entries}
    for entry in entries:
        raw_payload = entry["payload"]
        payload, normalization_notes = normalize_payload(raw_payload)
        logical_category = entry["logical_category"]
        resolved_refs: dict[str, list[dict[str, str]]] = {}
        external_refs: dict[str, list[dict[str, str]]] = {}
        unresolved_refs: list[str] = []
        for ref_field in ("related_env_vars", "related_apis", "related_features"):
            rows, external_rows, missing = [], [], []
            for ref_value in payload.get(ref_field, []):
                resolved, externalized, unresolved = resolve_related_reference(
                    ref_value,
                    ref_field,
                    lookup,
                    entries_by_file,
                )
                rows.extend(resolved)
                external_rows.extend(externalized)
                missing.extend(unresolved)
            if rows:
                resolved_refs[ref_field] = rows
            if external_rows:
                external_refs[ref_field] = external_rows
            unresolved_refs.extend(missing)

        task_types = derive_task_types(logical_category, payload)
        consumer_skills = derive_consumer_skills(task_types, logical_category, payload)
        migration_target = derive_migration_target(logical_category, task_types)
        utility_verdict = derive_utility_verdict(logical_category, payload, existing_topic_stems)
        implementation_repos = derive_implementation_repos(payload, resolved_refs, entries_by_file)
        domain_scope, domain_reason = derive_domain_scope(
            logical_category,
            payload,
            implementation_repos,
            resolved_refs,
            entries_by_file,
        )
        knowledge_domain = knowledge_domain_from_scope(domain_scope)
        canonical_payload, source_hash = canonicalize_source_payload(
            payload,
            domain_scope,
            knowledge_domain,
            implementation_repos,
            domain_reason,
        )
        canonical_source_file = expected_source_path(knowledge_domain, entry["legacy_source_file"]).as_posix()
        entry.update(
            {
                "source_id": canonical_payload.get("id", ""),
                "name": canonical_payload.get("name", ""),
                "summary": canonical_payload.get("summary", ""),
                "category": logical_category,
                "payload": canonical_payload,
                "source_hash": source_hash,
                "domain_scope": domain_scope,
                "knowledge_domain": knowledge_domain,
                "domain_reason": domain_reason,
                "implementation_repos": implementation_repos,
                "task_types": task_types,
                "consumer_skills": consumer_skills,
                "migration_target": migration_target,
                "utility_verdict": utility_verdict,
                "resolved_internal_refs": resolved_refs,
                "external_refs": external_refs,
                "unresolved_refs": sorted(set(unresolved_refs)),
                "normalization_notes": normalization_notes,
                "source_file": canonical_source_file,
            }
        )
    return sorted(entries, key=lambda row: row["source_file"])


def evaluate_entries(
    workspace_root: Path,
    entries: list[dict[str, Any]],
    schema: dict[str, Any],
    version_info: dict[str, str],
    previous_records: dict[str, dict[str, Any]],
    batch_lookup: dict[str, str],
    execution_state: dict[str, Any] | None = None,
    execution_state_path: Path | None = None,
    no_write: bool = False,
) -> list[dict[str, Any]]:
    validator = Draft7Validator(schema)
    repo_root_map = repo_roots(workspace_root)
    commits = {
        "vllm": version_info.get("vllm_commit", ""),
        "vllm-ascend": version_info.get("vllm_ascend_commit", ""),
    }
    support_corpus = build_support_corpus(workspace_root)

    evaluated = []
    total_count = len(entries)
    required_reuse_keys = {
        "domain_scope",
        "knowledge_domain",
        "implementation_repos",
        "domain_reason",
        "source_hash",
    }
    for index, entry in enumerate(entries):
        payload = entry["payload"]
        source_file = entry["source_file"]
        previous_record = previous_records.get(source_file) or previous_records.get(entry["loaded_source_file"])
        can_reuse = (
            previous_record is not None
            and required_reuse_keys.issubset(previous_record)
            and previous_record.get("source_hash") == entry["source_hash"]
        )

        if can_reuse:
            record = json.loads(json.dumps(previous_record))
            record.update(
                {
                    "source_file": source_file,
                    "legacy_source_file": entry["legacy_source_file"],
                    "category": entry["logical_category"],
                    "batch_id": batch_lookup[source_file],
                    "task_types": entry["task_types"],
                    "consumer_skills": entry["consumer_skills"],
                    "migration_target": entry["migration_target"],
                    "utility_verdict": entry["utility_verdict"],
                    "resolved_internal_refs": entry["resolved_internal_refs"],
                    "external_refs": entry["external_refs"],
                    "domain_scope": entry["domain_scope"],
                    "knowledge_domain": entry["knowledge_domain"],
                    "implementation_repos": entry["implementation_repos"],
                    "domain_reason": entry["domain_reason"],
                    "source_hash": entry["source_hash"],
                    "evaluation_mode": "reused",
                }
            )
        else:
            issues = []
            issues.extend(collect_schema_errors(validator, payload))
            issues.extend(validate_non_empty_perspectives(payload))

            code_evidence, code_issues, code_web = gather_code_evidence(payload, repo_root_map, commits)
            repo_evidence, repo_issues, repo_web = search_local_supporting_files(payload, commits, support_corpus)
            extra_web = supplemental_web_evidence(payload)
            issues.extend(code_issues)
            issues.extend(repo_issues)

            if not entry["task_types"]:
                issues.append("missing_task_types")
            if not entry["consumer_skills"]:
                issues.append("missing_consumer_skills")
            if "design_analysis" not in entry["task_types"] and entry["logical_category"] in {
                "apis",
                "operators",
                "quantization",
                "comm_groups",
                "moe_comm_types",
                "features/ascend",
            }:
                issues.append("missing_design_analysis_coverage")

            migration_notes: list[str] = []
            open_gaps = list(entry["unresolved_refs"])
            if entry["utility_verdict"] == "merge_required":
                migration_notes.append("merge_with_existing_shared_topic")
            if not repo_evidence:
                open_gaps.append("local_repo_supporting_material_missing")
            if needs_platform_doc_review(entry["logical_category"], payload) and not any(
                row["kind"] in {"repo_docs", "repo_examples", "repo_reference", "platform_doc"}
                for row in (repo_web + extra_web)
            ):
                open_gaps.append("official_cann_or_platform_doc_review_pending")

            structural_issue = any(
                issue.startswith(
                    (
                        "schema:",
                        "missing_code_path:",
                        "absolute_path:",
                        "invalid_",
                        "missing_task_types",
                        "missing_consumer_skills",
                    )
                )
                or issue == "perspectives:not_object"
                or issue.startswith("perspective_empty:")
                for issue in issues
            )

            if not code_evidence or all(not row["exists"] for row in code_evidence):
                factual_verdict = "insufficient_evidence"
            elif repo_evidence:
                factual_verdict = "code_doc_aligned"
            elif code_web or repo_web:
                factual_verdict = "code_web_aligned"
            else:
                factual_verdict = "code_confirmed"

            if structural_issue:
                status = "rewrite_required"
            elif factual_verdict == "insufficient_evidence":
                status = "reject"
            elif open_gaps:
                status = "validated_with_gap"
            else:
                status = "validated"

            reasoning = summarize_reasoning(status, factual_verdict, code_evidence, repo_evidence, open_gaps)
            record = {
                "source_id": payload.get("id", ""),
                "normalized_id": payload.get("id", ""),
                "source_file": source_file,
                "legacy_source_file": entry["legacy_source_file"],
                "category": entry["logical_category"],
                "batch_id": batch_lookup[source_file],
                "status": status,
                "factual_verdict": factual_verdict,
                "utility_verdict": entry["utility_verdict"],
                "name": payload.get("name", ""),
                "summary": payload.get("summary", ""),
                "task_types": entry["task_types"],
                "consumer_skills": entry["consumer_skills"],
                "migration_target": entry["migration_target"],
                "local_code_evidence": code_evidence,
                "local_repo_evidence": repo_evidence,
                "resolved_internal_refs": entry["resolved_internal_refs"],
                "external_refs": entry["external_refs"],
                "web_evidence": dedupe_dict_rows(code_web + repo_web + extra_web, "url"),
                "reasoning_adjudication": reasoning,
                "normalization_notes": entry["normalization_notes"],
                "migration_notes": migration_notes,
                "open_gaps": sorted(set(open_gaps)),
                "validation_issues": issues,
                "domain_scope": entry["domain_scope"],
                "knowledge_domain": entry["knowledge_domain"],
                "implementation_repos": entry["implementation_repos"],
                "domain_reason": entry["domain_reason"],
                "source_hash": entry["source_hash"],
                "updated_at": utcnow(),
                "evaluation_mode": "fresh",
            }

        evaluated.append(record)
        if execution_state is not None and execution_state_path is not None and not no_write:
            update_execution_state_for_entry(execution_state, record, index, total_count)
            write_json(execution_state_path, execution_state)
    return sorted(evaluated, key=lambda row: row["source_file"])


def build_reports(
    records: list[dict[str, Any]],
    supporting_artifacts: dict[str, Any],
    shared_exports: dict[str, str],
    web_cache: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    status_counts = Counter(record["status"] for record in records)
    factual_counts = Counter(record["factual_verdict"] for record in records)
    category_counts = Counter(record["category"] for record in records)
    domain_counts = Counter(record["domain_scope"] for record in records)
    knowledge_domain_counts = Counter(record["knowledge_domain"] for record in records)
    task_counts = Counter(task for record in records for task in record["task_types"])
    skill_counts = Counter(skill for record in records for skill in record["consumer_skills"])
    evaluation_mode_counts = Counter(record.get("evaluation_mode", "fresh") for record in records)
    normalization_counts = Counter(
        note for record in records for note in record.get("normalization_notes", [])
    )
    actual_counts = count_entries([{"logical_category": record["category"]} for record in records])
    expected_counts = dict(actual_counts)

    count_issues = []
    for category, expected in expected_counts.items():
        actual = actual_counts.get(category, 0)
        if actual != expected:
            count_issues.append(
                {
                    "category": category,
                    "expected": expected,
                    "actual": actual,
                }
            )

    blocker_entries = [
        {
            "source_id": record["source_id"],
            "source_file": record["source_file"],
            "status": record["status"],
            "validation_issues": record["validation_issues"],
            "open_gaps": record["open_gaps"],
        }
        for record in records
        if record["status"] in {"rewrite_required", "reject"}
    ]
    count_mismatch_diagnosis = {
        "generated_at": utcnow(),
        "expected_counts": expected_counts,
        "actual_counts": actual_counts,
        "issues": count_issues,
        "entries_by_category": {
            category: [
                record["source_file"]
                for record in records
                if record["category"] == category
            ]
            for category in sorted(actual_counts)
        },
    }
    scenario_summary = {}
    scenario_export = shared_exports.get("scenario_coverage")
    if scenario_export and Path(scenario_export).exists():
        scenario_payload = read_json(Path(scenario_export))
        scenario_summary = {
            "status": scenario_payload.get("status"),
            "scenario_count": scenario_payload.get("scenario_count", 0),
            "composer_coverage": {
                skill: payload.get("scenario_count", 0)
                for skill, payload in scenario_payload.get("composer_coverage", {}).items()
            },
            "atomic_skill_coverage": {
                skill: payload.get("scenario_count", 0)
                for skill, payload in scenario_payload.get("atomic_skill_coverage", {}).items()
            },
        }

    validation_report = {
        "generated_at": utcnow(),
        "workspace_root": "vllm_workspace",
        "knowledge_roots": {
            "vllm-upstream": "vllm-ascend/.agents/skills/_shared/vllm-upstream/references/source/knowledge",
            "vllm-ascend-core": "vllm-ascend/.agents/skills/_shared/vllm-ascend-core/references/source/knowledge",
            "integration-core": "vllm-ascend/.agents/skills/_shared/integration-core/references/source/knowledge",
        },
        "governance_root": "vllm-ascend/.agents/skills/_shared/knowledge-governance",
        "summary": {
            "total_entries": len(records),
            "status_counts": dict(status_counts),
            "factual_verdict_counts": dict(factual_counts),
            "normalization_counts": dict(normalization_counts),
            "evaluation_mode_counts": dict(evaluation_mode_counts),
        },
        "category_counts": dict(sorted(category_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "knowledge_domain_counts": dict(sorted(knowledge_domain_counts.items())),
        "count_validation": {
            "expected_counts": expected_counts,
            "actual_counts": actual_counts,
            "issues": count_issues,
            "status": "pass" if not count_issues else "fail",
        },
        "supporting_artifacts": supporting_artifacts,
        "web_evidence_cache": {
            "url_count": web_cache.get("url_count", 0),
            "fetched_ok": len(
                [value for value in web_cache.get("urls", {}).values() if value.get("status") == "ok"]
            ),
        },
        "blocker_entries": blocker_entries,
        "count_mismatch_diagnosis": count_mismatch_diagnosis,
        "scenario_coverage": scenario_summary,
        "resume_contract": {
            "primary_state": "execution_state.json",
            "human_handoff": "verification_handoff.md",
            "incremental_reuse_enabled": True,
        },
    }

    import_manifest = {
        "generated_at": utcnow(),
        "source_entry_count": len(records),
        "eligible_entry_count": len(
            [record for record in records if record["status"] in {"validated", "validated_with_gap"}]
        ),
        "coverage": {
            "expected": len(records),
            "actual": len(records),
            "ratio": 1.0 if records else 0.0,
        },
        "entries": [
            {
                key: record[key]
                for key in (
                    "source_id",
                    "normalized_id",
                    "source_file",
                    "category",
                    "status",
                    "factual_verdict",
                    "utility_verdict",
                    "name",
                    "summary",
                    "task_types",
                    "consumer_skills",
                    "migration_target",
                    "domain_scope",
                    "knowledge_domain",
                    "implementation_repos",
                    "domain_reason",
                    "source_hash",
                    "local_code_evidence",
                    "local_repo_evidence",
                    "web_evidence",
                    "reasoning_adjudication",
                    "normalization_notes",
                    "open_gaps",
                    "evaluation_mode",
                    "updated_at",
                )
            }
            | {
                "export_mode": (
                    "generated_reference"
                    if record["utility_verdict"] == "reference_only"
                    else "merge_existing"
                    if record["utility_verdict"] == "merge_required"
                    else "curated_or_generated"
                    if record["utility_verdict"] == "direct_skill_topic"
                    else "skip"
                ),
                "shared_export_paths": shared_exports,
            }
            for record in records
        ],
    }

    final_report = {
        "generated_at": utcnow(),
        "summary": {
            "total_entries": len(records),
            "terminal_entries": len(records),
            "validated": status_counts.get("validated", 0),
            "validated_with_gap": status_counts.get("validated_with_gap", 0),
            "rewrite_required": status_counts.get("rewrite_required", 0),
            "rejected": status_counts.get("reject", 0),
        },
        "normalization_counts": dict(normalization_counts),
        "evaluation_mode_counts": dict(evaluation_mode_counts),
        "task_coverage": {task: task_counts.get(task, 0) for task in TASK_TYPES},
        "skill_coverage": {skill: skill_counts.get(skill, 0) for skill in sorted(ALL_SKILLS)},
        "domain_coverage": {scope: domain_counts.get(scope, 0) for scope in DOMAIN_SCOPES},
        "knowledge_domain_coverage": {
            knowledge_domain: knowledge_domain_counts.get(knowledge_domain, 0)
            for knowledge_domain in KNOWLEDGE_DOMAINS
        },
        "design_analysis": {
            "covered_entries": len([record for record in records if "design_analysis" in record["task_types"]]),
            "non_validated_entries": len(
                [
                    record
                    for record in records
                    if "design_analysis" in record["task_types"] and record["status"] != "validated"
                ]
            ),
        },
        "migration_targets": dict(Counter(record["migration_target"] for record in records)),
        "count_validation": validation_report["count_validation"],
        "supporting_artifacts": supporting_artifacts,
        "shared_exports": shared_exports,
        "scenario_coverage": scenario_summary,
        "web_evidence_cache": {
            "url_count": web_cache.get("url_count", 0),
            "fetched_ok": len(
                [value for value in web_cache.get("urls", {}).values() if value.get("status") == "ok"]
            ),
        },
        "blocker_entries": blocker_entries,
        "count_mismatch_diagnosis": count_mismatch_diagnosis,
    }
    return validation_report, import_manifest, final_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path(__file__).resolve().parents[5],
        help="Workspace root that contains `vllm/` and `vllm-ascend/`.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run the pipeline without writing generated artifacts.",
    )
    parser.add_argument(
        "--fetch-web-evidence",
        action="store_true",
        help="Fetch metadata for official web evidence URLs and cache the results locally.",
    )
    parser.add_argument(
        "--fetch-web-evidence-limit",
        type=int,
        default=25,
        help="Maximum number of uncached URLs to fetch in one run. Use -1 for no limit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace_root = args.workspace_root.resolve()
    shared_root = workspace_root / "vllm-ascend" / ".agents" / "skills" / "_shared"
    governance_root = shared_root / "knowledge-governance"
    contracts_root = governance_root / "contracts"
    governance_source_root = governance_root / "source"
    provenance_root = governance_root / "provenance"
    generated_root = governance_root / "generated"

    domain_registry = load_domain_registry(contracts_root / "knowledge_domain_registry.json")
    source_roots = {
        domain["domain_id"]: shared_root / domain["source_root"]
        for domain in domain_registry["domains"]
    }
    schema = read_json(contracts_root / "knowledge_point_schema.json")
    version_info = read_json(governance_source_root / "meta" / "version.json")
    existing_topic_stems = load_existing_topic_stems(shared_root)

    entries = load_knowledge_points(shared_root, domain_registry)
    prepared_entries = prepare_entries(entries, existing_topic_stems)
    batch_plan = build_batch_plan(prepared_entries)
    batch_lookup = {
        source_file: batch["batch_id"]
        for batch in batch_plan["batches"]
        for source_file in batch["source_files"]
    }
    snapshot_hash = compute_snapshot_hash(prepared_entries)
    execution_state_path = provenance_root / "execution_state.json"
    previous_records = load_previous_manifest_records(provenance_root)
    execution_state = build_execution_state(
        batch_plan,
        prepared_entries,
        source_roots,
        provenance_root,
        generated_root,
        snapshot_hash,
    )
    if not args.no_write:
        write_json(execution_state_path, execution_state)

    records = evaluate_entries(
        workspace_root,
        prepared_entries,
        schema,
        version_info,
        previous_records,
        batch_lookup,
        execution_state=execution_state,
        execution_state_path=execution_state_path,
        no_write=args.no_write,
    )
    web_cache_path = provenance_root / "web_evidence_cache.json"
    web_cache = enrich_web_evidence(
        records,
        web_cache_path,
        fetch_enabled=args.fetch_web_evidence,
        fetch_limit=args.fetch_web_evidence_limit,
    )
    batch_state = finalize_batch_state(batch_plan, records)
    supporting_artifacts = validate_supporting_artifacts(governance_source_root)

    provisional_import_manifest = {
        "source_entry_count": len(records),
        "entries": records,
    }
    shared_exports = (
        export_shared_artifacts(shared_root, generated_root, provisional_import_manifest)
        if not args.no_write
        else {}
    )
    validation_report, import_manifest, final_report = build_reports(
        records,
        supporting_artifacts,
        shared_exports,
        web_cache,
    )
    handoff = build_handoff(records, batch_state, validation_report)

    if not args.no_write:
        sync_canonical_source_tree(source_roots, prepared_entries)
        write_json(provenance_root / "verification_manifest.json", {"generated_at": utcnow(), "entries": records})
        write_json(provenance_root / "verification_batches.json", batch_state)
        (provenance_root / "verification_handoff.md").write_text(handoff, encoding="utf-8")
        write_json(provenance_root / "import_manifest.json", import_manifest)
        write_json(provenance_root / "validation_report.json", validation_report)
        write_json(provenance_root / "final_verification_report.json", final_report)
        write_json(provenance_root / "blocker_entries.json", validation_report["blocker_entries"])
        write_json(provenance_root / "count_mismatch_diagnosis.json", validation_report["count_mismatch_diagnosis"])
        write_json(web_cache_path, web_cache)
        finalize_execution_state(execution_state, batch_state, snapshot_hash)
        write_json(execution_state_path, execution_state)

    print(f"workspace={workspace_root}")
    print(f"entries={len(records)}")
    print(f"validated={sum(1 for row in records if row['status'] == 'validated')}")
    print(f"validated_with_gap={sum(1 for row in records if row['status'] == 'validated_with_gap')}")
    print(f"rewrite_required={sum(1 for row in records if row['status'] == 'rewrite_required')}")
    print(f"reject={sum(1 for row in records if row['status'] == 'reject')}")
    if not args.no_write:
        print(f"wrote={provenance_root / 'verification_manifest.json'}")
        print(f"web_cache={web_cache_path}")
        print(f"web_fetches_this_run={web_cache.get('fetched_this_run', 0)}")
        print(f"shared_exports={shared_exports}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
