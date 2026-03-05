#!/usr/bin/env python3
"""Build global parameter/env knowledge base from vllm + vllm-ascend sources."""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


FLAG_PATTERN = re.compile(r"(?<![A-Za-z0-9_])(--[a-z0-9][a-z0-9\-]*)(?![A-Za-z0-9_])")

FEATURE_PRIORITY = [
    "quantization",
    "graph_mode",
    "tensor_parallel",
    "data_parallel",
    "expert_parallel",
    "context_parallel",
    "prefill_decode_disaggregation",
    "prefix_cache",
    "lora",
    "speculative_decode",
    "weight_prefetch",
    "sleep_mode",
    "throughput_tuning",
    "memory_tuning",
    "network_serving",
    "security_auth",
    "multimodal",
    "logging_debug",
    "profiling_observability",
    "model_selection",
    "general_runtime",
]

FEATURE_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "quantization",
        (
            "quant",
            "int8",
            "int4",
            "w8a8",
            "w4a4",
            "gptq",
            "awq",
            "fp8",
        ),
    ),
    (
        "graph_mode",
        (
            "graph",
            "cudagraph",
            "compilation",
            "enforce-eager",
            "optimization-level",
            "kernel-config",
        ),
    ),
    (
        "tensor_parallel",
        (
            "tensor-parallel",
            "tp-size",
            "--tp",
            "all2all",
            "mm-encoder-tp",
            "matmul-allreduce",
        ),
    ),
    (
        "data_parallel",
        (
            "data-parallel",
            "dp-",
            "external-dp",
            "distributed-executor",
            "master-addr",
            "master-port",
            "hccl",
        ),
    ),
    (
        "expert_parallel",
        (
            "expert-parallel",
            "eplb",
            "expert-placement",
            "routed-experts",
            "moe",
        ),
    ),
    (
        "context_parallel",
        (
            "context-parallel",
            "prefill-context-parallel",
            "decode-context-parallel",
            "cp-kv",
            "dcp-kv",
            "ascend_enable_context_parallel",
        ),
    ),
    (
        "prefill_decode_disaggregation",
        (
            "prefill",
            "decode-servers",
            "prefill-servers",
            "prefiller",
            "decoder-hosts",
            "kv-transfer",
            "ec-transfer",
            "disagg",
            "connector",
        ),
    ),
    (
        "prefix_cache",
        (
            "prefix-caching",
            "prefix-repetition",
            "prefix-cache",
            "hash-algo",
        ),
    ),
    ("lora", ("lora",)),
    (
        "speculative_decode",
        (
            "speculative",
            "num_speculative",
            "draft",
            "mtp",
        ),
    ),
    (
        "weight_prefetch",
        (
            "prefetch",
            "weight_prefetch",
        ),
    ),
    (
        "sleep_mode",
        (
            "sleep-mode",
            "sleep_mode",
        ),
    ),
    (
        "throughput_tuning",
        (
            "async-scheduling",
            "max-num-batched-tokens",
            "max-num-seqs",
            "request-rate",
            "batch",
            "chunked-prefill",
            "scheduler",
            "scheduling-policy",
            "flashcomm",
            "balance_scheduling",
            "dbo",
        ),
    ),
    (
        "memory_tuning",
        (
            "gpu-memory-utilization",
            "swap-space",
            "cpu-offload",
            "max-model-len",
            "block-size",
            "kv-cache",
            "mm-processor-cache",
            "num-gpu-blocks-override",
            "long-prefill",
        ),
    ),
    (
        "network_serving",
        (
            "host",
            "port",
            "api-server",
            "api-url",
            "base-url",
            "server",
            "served-model-name",
            "root-path",
            "endpoint",
            "rpc-port",
            "allowed-",
            "uvicorn",
        ),
    ),
    (
        "security_auth",
        (
            "api-key",
            "ssl",
            "allow-credentials",
            "allowed-origins",
            "allowed-methods",
            "allowed-headers",
            "hf-token",
            "trust-remote-code",
        ),
    ),
    (
        "multimodal",
        (
            "mm-",
            "audio",
            "image",
            "video",
            "media",
            "embeds",
        ),
    ),
    (
        "logging_debug",
        (
            "log",
            "verbose",
            "trace",
            "debug",
            "error-stack",
            "nvtx",
        ),
    ),
    (
        "profiling_observability",
        (
            "metric",
            "profiler",
            "otlp",
            "collect-detailed-traces",
            "show-hidden-metrics",
        ),
    ),
    (
        "model_selection",
        (
            "model",
            "tokenizer",
            "revision",
            "code-revision",
            "download-dir",
            "dtype",
            "hf-",
            "runner",
            "task",
            "convert",
        ),
    ),
]

FEATURE_USAGE_HINT: dict[str, str] = {
    "quantization": "Controls model precision and quantized weight loading path.",
    "graph_mode": "Controls graph/eager execution and compile behavior.",
    "tensor_parallel": "Splits model tensors across NPUs/GPUs for scale-out inference.",
    "data_parallel": "Replicates workers for throughput and multi-node serving.",
    "expert_parallel": "Enables MoE expert routing parallelism; only valid on MoE models.",
    "context_parallel": "Splits long-context KV processing across ranks.",
    "prefill_decode_disaggregation": "Separates prefill/decode services or connectors.",
    "prefix_cache": "Reuses shared prompt prefixes to reduce prefill cost.",
    "lora": "Enables adapter loading and runtime LoRA routing.",
    "speculative_decode": "Enables draft/speculative decoding acceleration path.",
    "weight_prefetch": "Warms model weight blocks before decode to reduce stalls.",
    "sleep_mode": "Enables idle-time memory/power saving mode.",
    "throughput_tuning": "Tunes scheduler and batching for higher throughput.",
    "memory_tuning": "Bounds memory pressure and sequence length behavior.",
    "network_serving": "Controls API host/port/endpoints and serving interface.",
    "security_auth": "Controls authentication, TLS, and request trust boundaries.",
    "multimodal": "Controls multimodal I/O paths and media preprocessing.",
    "logging_debug": "Controls logs, debug verbosity, and troubleshooting signal.",
    "profiling_observability": "Controls profiling, traces, and metrics visibility.",
    "model_selection": "Selects model/tokenizer/artifact and runner mode.",
    "general_runtime": "General runtime behavior; review source and profile defaults.",
}

FEATURE_COMBO_HINTS: dict[str, list[str]] = {
    "quantization": ["--model", "--dtype", "--tensor-parallel-size", "--compilation-config"],
    "graph_mode": ["--compilation-config", "--enforce-eager", "--max-num-batched-tokens"],
    "tensor_parallel": ["--tensor-parallel-size", "--data-parallel-size", "--distributed-executor-backend"],
    "data_parallel": ["--data-parallel-size", "--data-parallel-address", "--data-parallel-rpc-port"],
    "expert_parallel": ["--enable-expert-parallel", "--tensor-parallel-size", "--data-parallel-size"],
    "context_parallel": ["--prefill-context-parallel-size", "--decode-context-parallel-size", "--max-model-len"],
    "prefill_decode_disaggregation": ["--kv-transfer-config", "--data-parallel-size", "--data-parallel-address"],
    "prefix_cache": ["--enable-prefix-caching", "--prefix-caching-hash-algo", "--max-model-len"],
    "lora": ["--enable-lora", "--lora-modules", "--max-loras"],
    "speculative_decode": ["--speculative-config", "--max-num-batched-tokens", "--async-scheduling"],
    "weight_prefetch": ["--additional-config", "--max-num-batched-tokens", "--gpu-memory-utilization"],
    "sleep_mode": ["--enable-sleep-mode", "--gpu-memory-utilization", "--max-model-len"],
    "throughput_tuning": ["--async-scheduling", "--max-num-batched-tokens", "--max-num-seqs"],
    "memory_tuning": ["--gpu-memory-utilization", "--max-model-len", "--block-size"],
    "network_serving": ["--host", "--port", "--served-model-name"],
    "security_auth": ["--api-key", "--ssl-certfile", "--allowed-origins"],
    "multimodal": ["--limit-mm-per-prompt", "--mm-processor-cache-gb", "--allowed-local-media-path"],
    "logging_debug": ["--disable-log-stats", "--max-log-len", "--log-config-file"],
    "profiling_observability": ["--profiler-config", "--collect-detailed-traces", "--otlp-traces-endpoint"],
    "model_selection": ["--model", "--tokenizer", "--revision", "--trust-remote-code"],
    "general_runtime": ["--model", "--device", "--dtype"],
}

BLOCKED_CASES = [
    {
        "profile": "qwen3-32b-w8a8",
        "blocked_feature": "int4_quantization",
        "reason": "Profile uses W8A8 artifact; int4 needs dedicated W4A4 artifact and validated kernel path.",
        "fallback": "Keep W8A8 or switch to an int4-ready profile/artifact.",
    },
    {
        "profile": "qwen3-32b-w8a8",
        "blocked_feature": "expert_parallel",
        "reason": "Qwen3-32B is dense, not MoE; EP is not applicable.",
        "fallback": "Use TP/DP tuning instead.",
    },
]


@dataclass(frozen=True)
class FlagRef:
    flag: str
    source: str


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _priority(feature: str) -> int:
    try:
        return FEATURE_PRIORITY.index(feature)
    except ValueError:
        return len(FEATURE_PRIORITY)


def _normalize_key(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _derive_feature_tags(name: str) -> list[str]:
    key = _normalize_key(name)
    tags: set[str] = set()

    for feature, needles in FEATURE_RULES:
        if any(needle in key for needle in needles):
            tags.add(feature)

    if not tags:
        tags.add("general_runtime")

    return sorted(tags, key=_priority)


def _extract_env_from_vllm(vllm_root: Path) -> dict[str, dict[str, object]]:
    env_file = vllm_root / "vllm" / "envs.py"
    if not env_file.exists():
        return {}

    text = _read(env_file)
    envs: dict[str, dict[str, object]] = {}

    for match in re.finditer(r"^\s*([A-Z][A-Z0-9_]+)\s*:\s*", text, flags=re.MULTILINE):
        name = match.group(1)
        envs.setdefault(name, {"sources": set(), "kind": "vllm_env"})
        envs[name]["sources"].add("vllm/envs.py:type_checking")

    start = text.find("environment_variables")
    if start != -1:
        snippet = text[start:]
        for match in re.finditer(r'"([A-Z][A-Z0-9_]+)"\s*:', snippet):
            name = match.group(1)
            envs.setdefault(name, {"sources": set(), "kind": "vllm_env"})
            envs[name]["sources"].add("vllm/envs.py:environment_variables")

    for value in envs.values():
        value["sources"] = sorted(value["sources"])

    return dict(sorted(envs.items()))


def _extract_env_from_ascend(ascend_root: Path) -> dict[str, dict[str, object]]:
    env_file = ascend_root / "vllm_ascend" / "envs.py"
    if not env_file.exists():
        return {}

    text = _read(env_file)
    envs: dict[str, dict[str, object]] = {}

    for match in re.finditer(r'"([A-Z][A-Z0-9_]+)"\s*:', text):
        name = match.group(1)
        envs.setdefault(name, {"sources": set(), "kind": "vllm_ascend_env"})
        envs[name]["sources"].add("vllm_ascend/envs.py:env_variables")

    for value in envs.values():
        value["sources"] = sorted(value["sources"])

    return dict(sorted(envs.items()))


def _extract_add_argument_flags(py_file: Path, rel: str) -> list[FlagRef]:
    text = _read(py_file)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    flags: list[FlagRef] = []

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        value = arg.value.strip()
                        if value.startswith("--") and FLAG_PATTERN.fullmatch(value):
                            flags.append(FlagRef(flag=value, source=rel))
            self.generic_visit(node)

    Visitor().visit(tree)
    return flags


def _extract_dataclass_fields(py_file: Path, rel: str, class_names: set[str]) -> list[FlagRef]:
    text = _read(py_file)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    flags: list[FlagRef] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in class_names:
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    name = stmt.target.id
                    if name.startswith("_"):
                        continue
                    flag = "--" + name.replace("_", "-")
                    if FLAG_PATTERN.fullmatch(flag):
                        flags.append(FlagRef(flag=flag, source=f"{rel}:{node.name}"))
    return flags


def _extract_shell_like_flags(files: Iterable[Path], base: Path) -> tuple[set[str], dict[str, set[str]], Counter]:
    flag_sources: dict[str, set[str]] = defaultdict(set)
    counter: Counter = Counter()

    for file in files:
        rel = str(file.relative_to(base))
        text = _read(file)
        for flag in FLAG_PATTERN.findall(text):
            counter[flag] += 1
            flag_sources[flag].add(rel)

    return set(flag_sources.keys()), flag_sources, counter


def _collect_flag_cooccurrence(files: Iterable[Path], base: Path) -> Counter:
    pairs: Counter = Counter()

    for file in files:
        text = _read(file)
        flags = sorted(set(FLAG_PATTERN.findall(text)))
        if len(flags) < 2:
            continue
        for left, right in itertools.combinations(flags, 2):
            pairs[(left, right)] += 1

    return pairs


def _collect_vllm_flags(vllm_root: Path) -> dict[str, dict[str, object]]:
    flags: dict[str, dict[str, object]] = {}

    targets = [
        vllm_root / "vllm" / "entrypoints" / "openai" / "cli_args.py",
        vllm_root / "vllm" / "engine" / "arg_utils.py",
        vllm_root / "vllm" / "entrypoints" / "cli" / "serve.py",
    ]

    refs: list[FlagRef] = []
    for path in targets:
        if not path.exists():
            continue
        rel = str(path.relative_to(vllm_root))
        refs.extend(_extract_add_argument_flags(path, rel))
        refs.extend(
            _extract_dataclass_fields(
                path,
                rel,
                class_names={"FrontendArgs", "EngineArgs", "AsyncEngineArgs"},
            )
        )

    for ref in refs:
        flags.setdefault(ref.flag, {"sources": set(), "kind": "vllm_arg"})
        flags[ref.flag]["sources"].add(ref.source)

    for value in flags.values():
        value["sources"] = sorted(value["sources"])

    return dict(sorted(flags.items()))


def _collect_vllm_ascend_flags(
    ascend_root: Path,
) -> tuple[dict[str, dict[str, object]], Counter, Counter, list[Path]]:
    scan_files: list[Path] = []
    for pattern in [
        "examples/**/*.py",
        "examples/**/*.sh",
        "docs/source/user_guide/**/*.md",
        "docs/source/tutorials/**/*.md",
        "tests/e2e/**/*.yaml",
        "tests/e2e/**/*.yml",
    ]:
        scan_files.extend(ascend_root.glob(pattern))

    flags_set, flag_sources, counter = _extract_shell_like_flags(scan_files, ascend_root)
    pair_counter = _collect_flag_cooccurrence(scan_files, ascend_root)

    data: dict[str, dict[str, object]] = {}
    for flag in sorted(flags_set):
        data[flag] = {
            "sources": sorted(flag_sources[flag]),
            "kind": "vllm_ascend_arg",
            "freq": int(counter[flag]),
        }

    return data, counter, pair_counter, scan_files


def _top_partners(name: str, pair_counter: Counter, limit: int = 3) -> list[str]:
    partner_counter: Counter = Counter()
    for (left, right), count in pair_counter.items():
        if left == name:
            partner_counter[right] += count
        elif right == name:
            partner_counter[left] += count

    return [key for key, _ in partner_counter.most_common(limit)]


def _filter_partners_by_feature(partners: list[str], primary_feature: str, limit: int = 3) -> list[str]:
    kept: list[str] = []
    for partner in partners:
        partner_tags = _derive_feature_tags(partner)
        if primary_feature in partner_tags:
            kept.append(partner)
        if len(kept) >= limit:
            break
    return kept


def _build_feature_to_params(entries: dict[str, dict[str, object]]) -> dict[str, list[str]]:
    feature_to_params: dict[str, list[str]] = defaultdict(list)
    for name, meta in entries.items():
        tags: list[str] = meta.get("feature_tags", [])
        if not tags:
            continue
        feature_to_params[tags[0]].append(name)

    return dict(feature_to_params)


def _enrich_entries(
    raw: dict[str, dict[str, object]],
    scope: str,
    pair_counter: Counter,
    fallback_by_feature: dict[str, list[str]],
) -> dict[str, dict[str, object]]:
    enriched: dict[str, dict[str, object]] = {}

    for name, meta in raw.items():
        tags = _derive_feature_tags(name)
        primary = tags[0]
        partners = _top_partners(name, pair_counter)
        partners = _filter_partners_by_feature(partners, primary)

        if not partners:
            fallback = [x for x in FEATURE_COMBO_HINTS.get(primary, []) if x != name]
            if not fallback:
                fallback = [x for x in fallback_by_feature.get(primary, []) if x != name]
            partners = fallback[:3]

        enriched[name] = {
            "scope": scope,
            "kind": meta.get("kind", ""),
            "sources": meta.get("sources", []),
            "frequency": int(meta.get("freq", 0)) if "freq" in meta else None,
            "feature_tags": tags,
            "primary_feature": primary,
            "usage_hint": FEATURE_USAGE_HINT.get(primary, FEATURE_USAGE_HINT["general_runtime"]),
            "combination_candidates": partners,
        }

    return dict(sorted(enriched.items()))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _render_source_preview(meta: dict[str, object], max_sources: int = 3) -> str:
    sources = list(meta.get("sources", []))
    preview = ", ".join(sources[:max_sources])
    if len(sources) > max_sources:
        preview += f" (+{len(sources)-max_sources} more)"
    return preview


def _render_inventory_rows(data: dict[str, dict[str, object]]) -> list[str]:
    rows: list[str] = []
    for name, meta in data.items():
        rows.append(f"| `{name}` | {meta.get('kind','')} | {_render_source_preview(meta)} |")
    return rows


def _render_feature_rows(data: dict[str, dict[str, object]]) -> list[str]:
    rows: list[str] = []
    for name, meta in data.items():
        secondary = ", ".join(f"`{x}`" for x in meta.get("feature_tags", [])[1:3]) or "-"
        combos = ", ".join(f"`{x}`" for x in meta.get("combination_candidates", [])[:3]) or "-"
        usage = str(meta.get("usage_hint", "")).replace("|", "/")
        rows.append(
            f"| `{name}` | `{meta.get('primary_feature','general_runtime')}` | {secondary} | {usage} | {combos} |"
        )
    return rows


def _render_top_pairs(pair_counter: Counter, limit: int = 25) -> list[str]:
    lines: list[str] = []
    for idx, ((left, right), count) in enumerate(pair_counter.most_common(limit), start=1):
        lines.append(f"{idx}. `{left}` + `{right}` (co-occurrence files: {count})")
    return lines


def _build_feature_summary(*datasets: dict[str, dict[str, object]]) -> dict[str, int]:
    summary: Counter = Counter()
    for data in datasets:
        for meta in data.values():
            feature = meta.get("primary_feature", "general_runtime")
            summary[str(feature)] += 1
    return dict(summary)


def _write_markdown_docs(
    ascend_root: Path,
    now: str,
    vllm_env: dict[str, dict[str, object]],
    ascend_env: dict[str, dict[str, object]],
    vllm_args: dict[str, dict[str, object]],
    ascend_args: dict[str, dict[str, object]],
    enriched_vllm_args: dict[str, dict[str, object]],
    enriched_vllm_env: dict[str, dict[str, object]],
    enriched_ascend_args: dict[str, dict[str, object]],
    enriched_ascend_env: dict[str, dict[str, object]],
    pair_counter: Counter,
) -> None:
    shared_root = ascend_root / ".agents" / "skills" / "_shared"

    vllm_doc = shared_root / "vllm-foundation" / "references" / "vllm-inputs-and-envs-global.md"
    ascend_doc = shared_root / "vllm-ascend-core" / "references" / "vllm-ascend-inputs-and-envs-global.md"
    combo_doc = shared_root / "deployment-config" / "references" / "global-parameter-combination-guide.md"
    feature_map_doc = shared_root / "deployment-config" / "references" / "global-parameter-feature-map.md"

    vllm_doc.parent.mkdir(parents=True, exist_ok=True)
    ascend_doc.parent.mkdir(parents=True, exist_ok=True)
    combo_doc.parent.mkdir(parents=True, exist_ok=True)
    feature_map_doc.parent.mkdir(parents=True, exist_ok=True)

    vllm_doc.write_text(
        "\n".join(
            [
                "---",
                "knowledge_id: vllm-foundation.inputs-and-envs-global",
                "domain: vllm-foundation",
                "knowledge_type: reference",
                "summary: Global inventory of vLLM serve arguments and environment variables.",
                "applicable_vllm_versions: [\">=0.15.0\", \"<0.17.0\"]",
                "applicable_cann_versions: [\">=8.0.0\"]",
                f"last_verified: \"{now}\"",
                "watch_files:",
                "  - \"../vllm/vllm/envs.py\"",
                "  - \"../vllm/vllm/entrypoints/openai/cli_args.py\"",
                "  - \"../vllm/vllm/engine/arg_utils.py\"",
                "depends_on:",
                "  - \"../../INDEX.md\"",
                "source_commit: \"workspace-head\"",
                "freshness: \"fresh\"",
                "---",
                "",
                "# vLLM Global Inputs and Envs",
                "",
                f"Generated at: `{now}`",
                "",
                f"- vLLM env vars discovered: **{len(vllm_env)}**",
                f"- vLLM serve args discovered: **{len(vllm_args)}**",
                "",
                "## vLLM Serve Arguments (inventory)",
                "",
                "| Argument | Kind | Source preview |",
                "| --- | --- | --- |",
                *_render_inventory_rows(vllm_args),
                "",
                "## vLLM Environment Variables (inventory)",
                "",
                "| Variable | Kind | Source preview |",
                "| --- | --- | --- |",
                *_render_inventory_rows(vllm_env),
                "",
                "Detailed semantics and combinations:",
                "- `../../deployment-config/references/global-parameter-feature-map.md`",
                "- `../../deployment-config/references/global-parameter-combination-guide.md`",
                "",
                "Machine-readable artifacts:",
                "- `generated/vllm_args_inventory.json`",
                "- `generated/vllm_env_inventory.json`",
                "",
                "Back to [INDEX](../../INDEX.md).",
                "",
            ]
        ),
        encoding="utf-8",
    )

    ascend_doc.write_text(
        "\n".join(
            [
                "---",
                "knowledge_id: vllm-ascend-core.inputs-and-envs-global",
                "domain: vllm-ascend-core",
                "knowledge_type: reference",
                "summary: Global inventory of vLLM-Ascend environment variables and observed deployment arguments.",
                "applicable_vllm_versions: [\">=0.15.0\", \"<0.17.0\"]",
                "applicable_cann_versions: [\">=8.0.0\"]",
                f"last_verified: \"{now}\"",
                "watch_files:",
                "  - \"vllm_ascend/envs.py\"",
                "  - \"docs/source/user_guide/feature_guide/index.md\"",
                "  - \"docs/source/tutorials/models/index.md\"",
                "  - \"examples/run_dp_server.sh\"",
                "depends_on:",
                "  - \"../../INDEX.md\"",
                "  - \"references/repo-full-knowledge-map.md\"",
                "source_commit: \"workspace-head\"",
                "freshness: \"fresh\"",
                "---",
                "",
                "# vLLM-Ascend Global Inputs and Envs",
                "",
                f"Generated at: `{now}`",
                "",
                f"- vLLM-Ascend env vars discovered: **{len(ascend_env)}**",
                f"- vLLM-Ascend args observed across docs/examples/tests: **{len(ascend_args)}**",
                "",
                "## vLLM-Ascend Environment Variables (inventory)",
                "",
                "| Variable | Kind | Source preview |",
                "| --- | --- | --- |",
                *_render_inventory_rows(ascend_env),
                "",
                "## vLLM-Ascend Arguments (observed inventory)",
                "",
                "| Argument | Kind | Source preview |",
                "| --- | --- | --- |",
                *_render_inventory_rows(ascend_args),
                "",
                "Detailed semantics and combinations:",
                "- `../../deployment-config/references/global-parameter-feature-map.md`",
                "- `../../deployment-config/references/global-parameter-combination-guide.md`",
                "",
                "Machine-readable artifacts:",
                "- `generated/vllm_ascend_args_inventory.json`",
                "- `generated/vllm_ascend_env_inventory.json`",
                "",
                "Back to [INDEX](../../INDEX.md).",
                "",
            ]
        ),
        encoding="utf-8",
    )

    feature_map_doc.write_text(
        "\n".join(
            [
                "---",
                "knowledge_id: deployment-config.global-parameter-feature-map",
                "domain: deployment-config",
                "knowledge_type: reference",
                "summary: Global semantic map for vLLM and vLLM-Ascend args/envs with usage and combination hints.",
                "applicable_vllm_versions: [\">=0.15.0\", \"<0.17.0\"]",
                "applicable_cann_versions: [\">=8.0.0\"]",
                f"last_verified: \"{now}\"",
                "watch_files:",
                "  - \"../vllm-foundation/references/vllm-inputs-and-envs-global.md\"",
                "  - \"../vllm-ascend-core/references/vllm-ascend-inputs-and-envs-global.md\"",
                "depends_on:",
                "  - \"../../INDEX.md\"",
                "source_commit: \"workspace-head\"",
                "freshness: \"fresh\"",
                "---",
                "",
                "# Global Parameter Feature Map",
                "",
                "This document gives a global view for weak-model execution: every discovered parameter/env is mapped to feature semantics, usage intent, and common combinations.",
                "",
                "## Coverage",
                "",
                f"- vLLM args: **{len(enriched_vllm_args)}**",
                f"- vLLM envs: **{len(enriched_vllm_env)}**",
                f"- vLLM-Ascend args (observed): **{len(enriched_ascend_args)}**",
                f"- vLLM-Ascend envs: **{len(enriched_ascend_env)}**",
                "",
                "## Feature tags",
                "",
                "`quantization`, `graph_mode`, `tensor_parallel`, `data_parallel`, `expert_parallel`, `context_parallel`, `prefill_decode_disaggregation`, `prefix_cache`, `lora`, `speculative_decode`, `weight_prefetch`, `sleep_mode`, `throughput_tuning`, `memory_tuning`, `network_serving`, `security_auth`, `multimodal`, `logging_debug`, `profiling_observability`, `model_selection`, `general_runtime`.",
                "",
                "## vLLM Serve Args -> Semantics",
                "",
                "| Parameter | Primary feature | Secondary features | Usage | Common combinations |",
                "| --- | --- | --- | --- | --- |",
                *_render_feature_rows(enriched_vllm_args),
                "",
                "## vLLM Env Vars -> Semantics",
                "",
                "| Variable | Primary feature | Secondary features | Usage | Common combinations |",
                "| --- | --- | --- | --- | --- |",
                *_render_feature_rows(enriched_vllm_env),
                "",
                "## vLLM-Ascend Args -> Semantics",
                "",
                "| Parameter | Primary feature | Secondary features | Usage | Common combinations |",
                "| --- | --- | --- | --- | --- |",
                *_render_feature_rows(enriched_ascend_args),
                "",
                "## vLLM-Ascend Env Vars -> Semantics",
                "",
                "| Variable | Primary feature | Secondary features | Usage | Common combinations |",
                "| --- | --- | --- | --- | --- |",
                *_render_feature_rows(enriched_ascend_env),
                "",
                "Back to [INDEX](../../INDEX.md).",
                "",
            ]
        ),
        encoding="utf-8",
    )

    top_pairs = _render_top_pairs(pair_counter, limit=30)
    if not top_pairs:
        top_pairs = ["1. No co-occurrence pairs detected; rely on feature-level fallback combos."]

    combo_doc.write_text(
        "\n".join(
            [
                "---",
                "knowledge_id: deployment-config.global-parameter-combination-guide",
                "domain: deployment-config",
                "knowledge_type: procedure",
                "summary: Global combination guidance for vLLM and vLLM-Ascend parameters and env vars.",
                "applicable_vllm_versions: [\">=0.15.0\", \"<0.17.0\"]",
                "applicable_cann_versions: [\">=8.0.0\"]",
                f"last_verified: \"{now}\"",
                "watch_files:",
                "  - \"../vllm-foundation/references/vllm-inputs-and-envs-global.md\"",
                "  - \"../vllm-ascend-core/references/vllm-ascend-inputs-and-envs-global.md\"",
                "  - \"../vllm-ascend-core/concepts/model-feature-compatibility-matrix.md\"",
                "depends_on:",
                "  - \"../../INDEX.md\"",
                "source_commit: \"workspace-head\"",
                "freshness: \"fresh\"",
                "---",
                "",
                "# Global Parameter Combination Guide",
                "",
                "## Global decision order",
                "",
                "1. Resolve intent to canonical features.",
                "2. Check profile-level hard blocks before rendering commands.",
                "3. Select core arg stack and env stack by feature tags.",
                "4. Generate start/validate/rollback package.",
                "",
                "## High-impact feature stacks",
                "",
                "1. Quantized throughput stack",
                "- `--quantization` + `--model` + `--tensor-parallel-size` + `--max-num-batched-tokens`.",
                "2. Graph acceleration stack",
                "- `--compilation-config` + `--enforce-eager` (for A/B fallback) + `--max-model-len`.",
                "3. Parallel scale-out stack",
                "- `--tensor-parallel-size` + `--data-parallel-size` + `--distributed-executor-backend`.",
                "4. Long context stack",
                "- `--prefill-context-parallel-size` + `--decode-context-parallel-size` + `--max-model-len`.",
                "5. Prefill/decode split stack",
                "- `--kv-transfer-config` + DP addressing + decode/prefill endpoint args.",
                "",
                "## Co-occurrence evidence (from docs/examples/tests)",
                "",
                *top_pairs,
                "",
                "## Hard blocks in demo profiles",
                "",
                "- `qwen3-32b-w8a8 + int4_quantization`: blocked.",
                "- `qwen3-32b-w8a8 + expert_parallel`: blocked.",
                "",
                "## Error-case handling",
                "",
                "- If user asks `qwen3 32b w8a8 开 int4`: return blocked reason + suggest int4-ready artifact/profile switch.",
                "- If user asks `qwen3 32b w8a8 开 ep`: return dense-model incompatibility + suggest TP/DP path.",
                "",
                "## Weak model guardrails",
                "",
                "- Never execute ambiguous request directly; ask one clarification with <=3 candidates.",
                "- Keep one-decision-per-step output: params table -> commands -> validation -> rollback.",
                "",
                "Back to [INDEX](../../INDEX.md).",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Path to vllm-ascend repo root. Defaults to auto-detected path.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    default_ascend_root = script_dir.parents[3]
    ascend_root = Path(args.repo_root).resolve() if args.repo_root else default_ascend_root
    vllm_root = (ascend_root.parent / "vllm").resolve()

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    vllm_env = _extract_env_from_vllm(vllm_root)
    ascend_env = _extract_env_from_ascend(ascend_root)
    vllm_args = _collect_vllm_flags(vllm_root)
    ascend_args, ascend_arg_freq, pair_counter, scan_files = _collect_vllm_ascend_flags(ascend_root)

    # Build fallback candidate pool from observed args first, then enrich all datasets.
    raw_for_fallback: dict[str, dict[str, object]] = {}
    raw_for_fallback.update(vllm_args)
    raw_for_fallback.update(ascend_args)

    temp_enriched = {
        name: {
            "feature_tags": _derive_feature_tags(name),
        }
        for name in raw_for_fallback
    }
    fallback_by_feature = _build_feature_to_params(temp_enriched)

    enriched_vllm_args = _enrich_entries(vllm_args, "vllm_arg", pair_counter, fallback_by_feature)
    enriched_vllm_env = _enrich_entries(vllm_env, "vllm_env", pair_counter, fallback_by_feature)
    enriched_ascend_args = _enrich_entries(ascend_args, "vllm_ascend_arg", pair_counter, fallback_by_feature)
    enriched_ascend_env = _enrich_entries(ascend_env, "vllm_ascend_env", pair_counter, fallback_by_feature)

    out_vllm = ascend_root / ".agents" / "skills" / "_shared" / "vllm-foundation" / "references" / "generated"
    out_ascend = ascend_root / ".agents" / "skills" / "_shared" / "vllm-ascend-core" / "references" / "generated"
    out_deploy = ascend_root / ".agents" / "skills" / "_shared" / "deployment-config" / "references" / "generated"

    _write_json(out_vllm / "vllm_env_inventory.json", vllm_env)
    _write_json(out_vllm / "vllm_args_inventory.json", vllm_args)
    _write_json(out_ascend / "vllm_ascend_env_inventory.json", ascend_env)
    _write_json(out_ascend / "vllm_ascend_args_inventory.json", ascend_args)
    _write_json(out_ascend / "vllm_ascend_args_frequency.json", dict(ascend_arg_freq.most_common()))

    _write_json(out_deploy / "global_parameter_kb.json", {
        "generated_at": now,
        "datasets": {
            "vllm_args": enriched_vllm_args,
            "vllm_envs": enriched_vllm_env,
            "vllm_ascend_args": enriched_ascend_args,
            "vllm_ascend_envs": enriched_ascend_env,
        },
        "blocked_cases": BLOCKED_CASES,
    })
    _write_json(
        out_deploy / "global_feature_summary.json",
        _build_feature_summary(enriched_vllm_args, enriched_vllm_env, enriched_ascend_args, enriched_ascend_env),
    )
    _write_json(
        out_deploy / "global_flag_pairings.json",
        [
            {"left": left, "right": right, "cooccurrence_files": count}
            for (left, right), count in pair_counter.most_common(500)
        ],
    )
    _write_json(out_deploy / "global_scan_files.json", [str(path.relative_to(ascend_root)) for path in scan_files])

    _write_markdown_docs(
        ascend_root=ascend_root,
        now=now,
        vllm_env=vllm_env,
        ascend_env=ascend_env,
        vllm_args=vllm_args,
        ascend_args=ascend_args,
        enriched_vllm_args=enriched_vllm_args,
        enriched_vllm_env=enriched_vllm_env,
        enriched_ascend_args=enriched_ascend_args,
        enriched_ascend_env=enriched_ascend_env,
        pair_counter=pair_counter,
    )

    summary = {
        "vllm_env_count": len(vllm_env),
        "vllm_arg_count": len(vllm_args),
        "vllm_ascend_env_count": len(ascend_env),
        "vllm_ascend_arg_count": len(ascend_args),
        "feature_tag_count": len(_build_feature_summary(enriched_vllm_args, enriched_vllm_env, enriched_ascend_args, enriched_ascend_env)),
        "pairing_count": int(sum(pair_counter.values())),
        "generated_at": now,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
