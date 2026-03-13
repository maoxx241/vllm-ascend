from __future__ import annotations

from typing import Any

from .contracts import ContractError
from .shared import RawRequest, build_selector_seed, intake_from_seed


def public_entry(raw_request: RawRequest, root: Any | None = None) -> dict[str, Any]:
    return build_selector_seed(raw_request, root=root)


def vllm_ascend_assistant(raw_request: RawRequest, root: Any | None = None) -> dict[str, Any]:
    selector_seed = public_entry(raw_request, root=root)
    return intake_from_seed(selector_seed, root=root)


def _enforce_family(result: dict[str, Any], expected_family: str) -> dict[str, Any]:
    selector_plan = result["selector_plan"]
    if selector_plan is None:
        return result
    if selector_plan["task_family"] != expected_family:
        raise ContractError(
            f"canonical intake expected {expected_family}, got {selector_plan['task_family']}"
        )
    return result


def deployment_intake(raw_request: RawRequest, root: Any | None = None) -> dict[str, Any]:
    return _enforce_family(vllm_ascend_assistant(raw_request, root=root), "deployment_execution")


def perf_intake(raw_request: RawRequest, root: Any | None = None) -> dict[str, Any]:
    return _enforce_family(vllm_ascend_assistant(raw_request, root=root), "performance_analysis")


def validation_strategy_intake(raw_request: RawRequest, root: Any | None = None) -> dict[str, Any]:
    return _enforce_family(vllm_ascend_assistant(raw_request, root=root), "validation_strategy")


def debug_intake(raw_request: RawRequest, root: Any | None = None) -> dict[str, Any]:
    return _enforce_family(vllm_ascend_assistant(raw_request, root=root), "debugging")


def design_analysis_intake(raw_request: RawRequest, root: Any | None = None) -> dict[str, Any]:
    return _enforce_family(vllm_ascend_assistant(raw_request, root=root), "design_analysis")


def upstream_sync_intake(raw_request: RawRequest, root: Any | None = None) -> dict[str, Any]:
    return _enforce_family(vllm_ascend_assistant(raw_request, root=root), "upstream_sync")


def adaptation_intake(raw_request: RawRequest, root: Any | None = None) -> dict[str, Any]:
    result = vllm_ascend_assistant(raw_request, root=root)
    if result["selector_plan"] is None:
        return result
    return _enforce_family(result, "adaptation")


def operator_development_intake(raw_request: RawRequest, root: Any | None = None) -> dict[str, Any]:
    result = vllm_ascend_assistant(raw_request, root=root)
    if result["selector_plan"] is None:
        return result
    return _enforce_family(result, "operator_development")


__all__ = [
    "adaptation_intake",
    "debug_intake",
    "design_analysis_intake",
    "deployment_intake",
    "operator_development_intake",
    "perf_intake",
    "public_entry",
    "upstream_sync_intake",
    "validation_strategy_intake",
    "vllm_ascend_assistant",
]
