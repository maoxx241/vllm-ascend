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


__all__ = [
    "deployment_intake",
    "perf_intake",
    "public_entry",
    "validation_strategy_intake",
    "vllm_ascend_assistant",
]
