from __future__ import annotations

import importlib
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from .paths import workspace_root
from .strategy import ArtifactSelection, SelectorContext, StrategySelection

TRUTHY = {"1", "true", "yes", "on"}
GATE_ENV = "VLLM_ASCEND_TYPED_KB_SELECTOR_RUNTIME_ADAPTER"


def shadow_gate_enabled() -> bool:
    return os.getenv(GATE_ENV, "0").strip().lower() in TRUTHY


@lru_cache(maxsize=1)
def _load_shadow_module(tools_dir: str):
    tools_path = Path(tools_dir)
    if not tools_path.exists():
        return None
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    importlib.invalidate_caches()
    return importlib.import_module("selector_runtime_shadow_seam")


def _bindings_path(root: Path | None = None) -> Path:
    return workspace_root(root) / "artifacts" / "kb_inventory" / "typed_kb_tables_v2" / "selector_binding_candidates.jsonl"


def _tools_dir(root: Path | None = None) -> Path:
    return workspace_root(root) / "tools" / "kb_inventory"


def _selector_state(
    context: SelectorContext,
    strategy_selection: StrategySelection | None,
) -> dict[str, Any]:
    state: dict[str, Any] = {}
    requested_parallelism = dict(context.requested_parallelism)
    selected = strategy_selection.selected if strategy_selection is not None else None
    tp = requested_parallelism.get("tp") or (selected.tensor_parallel if selected is not None else None)
    dp = requested_parallelism.get("dp") or (selected.data_parallel if selected is not None else None)
    ep = requested_parallelism.get("ep") or (selected.expert_parallel if selected is not None else None)
    if tp is not None:
        state["parallel.tp"] = tp
    if dp is not None:
        state["parallel.dp"] = dp
    if ep is not None:
        state["parallel.ep"] = ep
    if context.topology_locked:
        state["topology.locked"] = True
    return state


def _runtime_state(
    context: SelectorContext,
    strategy_selection: StrategySelection | None,
    artifact_selection: ArtifactSelection | None,
) -> dict[str, Any]:
    selected = strategy_selection.selected if strategy_selection is not None else None
    selected_artifact = artifact_selection.selected if artifact_selection is not None else None
    traits = selected.model_traits if selected is not None else selected_artifact.model_traits if selected_artifact is not None else context.model_traits
    feature_modes: dict[str, Any] = {}
    model_traits = {trait: True for trait in traits}
    parallel_axes: dict[str, Any] = {}
    if selected is not None:
        if selected.tensor_parallel is not None:
            parallel_axes["tp"] = selected.tensor_parallel
        if selected.data_parallel is not None:
            parallel_axes["dp"] = selected.data_parallel
        if selected.expert_parallel is not None:
            parallel_axes["ep"] = selected.expert_parallel
    return {
        "feature_modes": feature_modes,
        "parallel_axes": parallel_axes,
        "model_traits": model_traits,
        "role_topology": {},
        "cache_profile": {},
        "logical_device_count": (
            selected.logical_npus if selected is not None else context.logical_npus
        ),
        "physical_card_count": (
            selected.physical_cards if selected is not None else context.physical_cards
        ),
    }


def _primary_decision(
    context: SelectorContext,
    strategy_selection: StrategySelection | None,
    artifact_selection: ArtifactSelection | None,
) -> dict[str, Any]:
    selected = strategy_selection.selected if strategy_selection is not None else None
    selected_artifact = artifact_selection.selected if artifact_selection is not None else None
    return {
        "decision_kind": selected.decision_kind if selected is not None else "strategy_unavailable",
        "artifact_decision_kind": selected_artifact.decision_kind if selected_artifact is not None else "artifact_unavailable",
        "model_base": selected.model_base if selected is not None else selected_artifact.model_base if selected_artifact is not None else context.model_base,
        "hw": selected.hw if selected is not None else selected_artifact.hw if selected_artifact is not None else context.hw,
    }


def build_shadow_diagnostics(
    *,
    root: Path | None,
    request_id: str,
    context: SelectorContext,
    strategy_selection: StrategySelection | None,
    artifact_selection: ArtifactSelection | None,
) -> dict[str, Any] | None:
    if not shadow_gate_enabled():
        return None

    bindings_path = _bindings_path(root)
    tools_dir = _tools_dir(root)
    if not bindings_path.exists() or not tools_dir.exists():
        return {
            "atoms": [
                {
                    "atom_id": "shadow:selected|status=unavailable|gate=1",
                    "atom_kind": "unknown",
                    "summary": "typed-KB shadow seam gate 已开启，但当前 branch 缺少 selector binding candidates 或 tools 入口；主决策保持不变。",
                    "source_refs": [],
                }
            ],
            "warnings": [
                "typed-KB shadow seam enabled but bindings/tooling are unavailable; primary decision preserved"
            ],
            "unknowns": ["missing typed_kb selector binding candidates or shadow seam tooling"],
        }

    shadow_module = _load_shadow_module(str(tools_dir))
    if shadow_module is None:
        return {
            "atoms": [
                {
                    "atom_id": "shadow:selected|status=unavailable|gate=1",
                    "atom_kind": "unknown",
                    "summary": "typed-KB shadow seam gate 已开启，但 tools 层模块未能加载；主决策保持不变。",
                    "source_refs": [],
                }
            ],
            "warnings": [
                "typed-KB shadow seam enabled but shadow module failed to load; primary decision preserved"
            ],
            "unknowns": ["shadow seam module unavailable"],
        }

    bindings = shadow_module.read_jsonl(bindings_path)
    envelope = shadow_module.run_shadow_seam(
        case_id=request_id,
        primary_decision=_primary_decision(context, strategy_selection, artifact_selection),
        selector_state=_selector_state(context, strategy_selection),
        runtime_state=_runtime_state(context, strategy_selection, artifact_selection),
        bindings=bindings,
        gate_enabled=True,
    )
    applicable_binding_ids = list(envelope.applicable_binding_ids)
    atoms: list[dict[str, Any]] = [
        {
            "atom_id": (
                f"shadow:selected|status={envelope.shadow_status}|gate=1|"
                f"applicable={len(applicable_binding_ids)}|recommendation={envelope.shadow_recommendation}"
            ),
            "atom_kind": "fact",
            "summary": (
                "typed-KB shadow seam 已执行；主决策保持不变，"
                f"shadow_status={envelope.shadow_status}，applicable_bindings={len(applicable_binding_ids)}。"
            ),
            "source_refs": applicable_binding_ids[:8],
        }
    ]
    warnings: list[str] = []
    unknowns: list[str] = []
    non_pass = [row for row in envelope.shadow_results if row["actual_outcome"] != "pass"]
    if non_pass:
        first = non_pass[0]
        atoms.append(
            {
                "atom_id": f"shadow:detail|binding={first['binding_id']}|outcome={first['actual_outcome']}",
                "atom_kind": "unknown" if first["actual_outcome"] in {"needs_runtime_probe", "needs_clarification"} else "constraint",
                "summary": f"typed-KB shadow result: {first['reason']}",
                "source_refs": [first["binding_id"]],
            }
        )
    if envelope.shadow_status == "shadow_blocking":
        warnings.append(
            "typed-KB shadow seam found blocking constraints; primary decision preserved because the gate is shadow-only"
        )
    elif envelope.shadow_status == "shadow_incomplete":
        warnings.append(
            "typed-KB shadow seam found incomplete runtime evidence; primary decision preserved because the gate is shadow-only"
        )
        unknowns.extend(row["reason"] for row in non_pass if row["actual_outcome"] in {"needs_runtime_probe", "needs_clarification"})
    return {
        "atoms": atoms,
        "warnings": warnings,
        "unknowns": unknowns,
    }
