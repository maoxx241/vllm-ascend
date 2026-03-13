from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .topology import logical_npus_for_hw, requested_card_count_from_features

STRATEGY_ATOM_PREFIX = "strategy:"
ARTIFACT_ATOM_PREFIX = "artifact:"
PARALLELISM_RE = re.compile(r"^(tp|dp|ep)_(\d+)$")


@dataclass(frozen=True)
class SelectorContext:
    model_base: str | None
    model_traits: tuple[str, ...]
    features: tuple[str, ...]
    configs: tuple[str, ...]
    hw: str | None
    physical_cards: int | None
    logical_npus: int | None
    topology_locked: bool
    user_priority: str
    requested_parallelism: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class StrategyCandidate:
    decision_kind: str
    model_base: str | None
    model_traits: tuple[str, ...]
    hw: str | None
    physical_cards: int | None
    logical_npus: int | None
    tensor_parallel: int | None
    data_parallel: int | None
    expert_parallel: int | None
    parallelism_family: str | None
    render_preset: str | None
    validation_id: str | None
    summary: str
    source_refs: tuple[str, ...]
    artifact_refs: tuple[str, ...]
    documented: bool
    unvalidated: bool
    confidence: str
    comm_profile: str | None
    env_profile: str | None


@dataclass(frozen=True)
class ArtifactCandidate:
    decision_kind: str
    model_base: str | None
    model_traits: tuple[str, ...]
    hw: str | None
    artifact_path_kind: str
    quantization_trait: str | None
    tool_name: str | None
    serving_quantization: str | None
    summary: str
    source_refs: tuple[str, ...]
    artifact_refs: tuple[str, ...]
    documented: bool
    unvalidated: bool
    confidence: str


@dataclass(frozen=True)
class ArtifactSelection:
    selected: ArtifactCandidate
    documented: ArtifactCandidate | None
    alternatives: tuple[ArtifactCandidate, ...]
    warnings: tuple[str, ...]
    unknowns: tuple[str, ...]


@dataclass(frozen=True)
class BaselineValidation:
    validation_id: str
    model_base: str | None
    model_traits: tuple[str, ...]
    hw: str | None
    mode: str
    physical_cards: int | None
    logical_npus: int | None
    tensor_parallel: int | None
    data_parallel: int | None
    expert_parallel: int | None
    render_preset: str | None
    summary: str
    artifact_refs: tuple[str, ...]
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class StrategySelection:
    selected: StrategyCandidate
    documented: StrategyCandidate | None
    alternatives: tuple[StrategyCandidate, ...]
    warnings: tuple[str, ...]
    unknowns: tuple[str, ...]


def _sorted_traits(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({value for value in values if value}))


def selector_context_from_selectors(selectors: dict[str, Any], runtime_soc: str | None = None) -> SelectorContext:
    features = list(selectors.get("features", []))
    configs = list(selectors.get("configs", []))
    models = list(selectors.get("models", []))
    model_traits: set[str] = set()
    base_model = models[0] if models else None
    if base_model and base_model.endswith("-w8a8"):
        base_model = base_model.removesuffix("-w8a8")
        model_traits.add("quant_w8a8")
    if "quant_w8a8" in features or "quant_w8a8" in configs:
        model_traits.add("quant_w8a8")
    if "quant_w4a8" in features or "quant_w4a8" in configs:
        model_traits.add("quant_w4a8")
    if "quant_w4a4" in features or "quant_w4a4" in configs:
        model_traits.add("quant_w4a4")
    if "bf16" in features or "precision_bf16" in configs:
        model_traits.add("precision_bf16")

    requested_parallelism: list[tuple[str, int]] = []
    for feature in features:
        if feature == "tp4":
            requested_parallelism.append(("tp", 4))
        if feature == "tp8":
            requested_parallelism.append(("tp", 8))
    for config in configs:
        match = PARALLELISM_RE.match(config)
        if match:
            requested_parallelism.append((match.group(1), int(match.group(2))))

    physical_cards = requested_card_count_from_features(features)
    hw_values = selectors.get("hw", [])
    hw = hw_values[0] if hw_values else runtime_soc
    logical_npus = logical_npus_for_hw(hw, physical_cards)
    topology_locked = (
        "topology_locked" in features
        or physical_cards is not None
        or bool(requested_parallelism)
    )

    if "priority_validated_only" in features:
        user_priority = "validated_only"
    elif "priority_keep_topology" in features:
        user_priority = "keep_requested_topology"
    elif "priority_best_perf" in features:
        user_priority = "best_perf_default"
    elif topology_locked:
        user_priority = "keep_requested_topology"
    else:
        user_priority = "best_perf_default"

    return SelectorContext(
        model_base=base_model,
        model_traits=_sorted_traits(model_traits),
        features=tuple(sorted(set(features))),
        configs=tuple(sorted(set(configs))),
        hw=hw,
        physical_cards=physical_cards,
        logical_npus=logical_npus,
        topology_locked=topology_locked,
        user_priority=user_priority,
        requested_parallelism=tuple(sorted(set(requested_parallelism))),
    )


def _canonical_model_base(target_name: str | None, metadata: dict[str, Any]) -> str | None:
    value = metadata.get("model_base") or target_name
    if isinstance(value, str) and value.endswith("-w8a8"):
        return value.removesuffix("-w8a8")
    return value if isinstance(value, str) else None


def _default_traits(target_name: str | None) -> tuple[str, ...]:
    if target_name and target_name.endswith("-w8a8"):
        return ("quant_w8a8",)
    return ()


def baselines_from_rows(
    validation_rows: Iterable[dict[str, Any] | Any],
    entity_name_by_id: dict[str, str],
) -> tuple[BaselineValidation, ...]:
    baselines: list[BaselineValidation] = []
    for row in validation_rows:
        target_name = entity_name_by_id.get(row["target_id"])
        env = json.loads(row["env_json"] or "{}")
        metadata = json.loads(row["metadata_json"] or "{}")
        model_base = _canonical_model_base(target_name, metadata)
        traits = _sorted_traits(metadata.get("model_traits", []) or _default_traits(target_name))
        baselines.append(
            BaselineValidation(
                validation_id=row["validation_id"],
                model_base=model_base,
                model_traits=traits,
                hw=metadata.get("hw") or env.get("hw"),
                mode=row["mode"],
                physical_cards=metadata.get("physical_cards") or env.get("physical_cards"),
                logical_npus=metadata.get("logical_npus") or env.get("logical_npus"),
                tensor_parallel=metadata.get("tensor_parallel"),
                data_parallel=metadata.get("data_parallel"),
                expert_parallel=metadata.get("expert_parallel"),
                render_preset=metadata.get("render_preset"),
                summary=row["summary"],
                artifact_refs=tuple(json.loads(row["artifact_refs_json"] or "[]")),
                source_refs=(row["validation_id"],),
            )
        )
    return tuple(baselines)


def topology_multiplier_from_rows(
    fact_rows: Iterable[dict[str, Any] | Any],
    *,
    hw: str | None,
) -> int | None:
    if hw is None:
        return None
    for row in fact_rows:
        metadata = json.loads(row["metadata_json"] or "{}")
        scope = json.loads(row["scope_json"] or "{}")
        fact_hw = metadata.get("hw") or metadata.get("soc")
        if isinstance(fact_hw, list):
            matches = hw in fact_hw
        elif isinstance(fact_hw, str):
            matches = fact_hw == hw
        else:
            scope_hw = scope.get("hw")
            matches = hw in scope_hw if isinstance(scope_hw, list) else scope_hw == hw
        if not matches:
            continue
        for key in ("physical_card_to_logical_npus", "physical_to_logical_npu_ratio"):
            value = metadata.get(key) or scope.get(key)
            if isinstance(value, int):
                return value
    return None


def _trait_score(requested: tuple[str, ...], baseline: BaselineValidation) -> tuple[int, int, str]:
    requested_set = set(requested)
    baseline_set = set(baseline.model_traits)
    if requested_set == baseline_set:
        return (0, 0, baseline.validation_id)
    if requested_set and requested_set <= baseline_set:
        return (1, len(baseline_set - requested_set), baseline.validation_id)
    if not requested_set and not baseline_set:
        return (0, 0, baseline.validation_id)
    return (2, abs(len(baseline_set) - len(requested_set)), baseline.validation_id)


def _matching_baselines(context: SelectorContext, baselines: tuple[BaselineValidation, ...]) -> list[BaselineValidation]:
    matches = [
        baseline
        for baseline in baselines
        if baseline.model_base == context.model_base and baseline.hw == context.hw
    ]
    matches.sort(key=lambda baseline: (_trait_score(context.model_traits, baseline), baseline.mode != "documented_baseline"))
    return matches


def _requested_quantization_trait(context: SelectorContext) -> str | None:
    for trait in ("quant_w8a8", "quant_w4a8", "quant_w4a4"):
        if trait in context.model_traits:
            return trait
    return None


def _comm_profile(
    *,
    logical_npus: int | None,
    tensor_parallel: int | None,
    data_parallel: int | None,
    expert_parallel: int | None,
) -> str | None:
    if logical_npus is None:
        return None
    tp = tensor_parallel or 1
    dp = data_parallel or 1
    ep = expert_parallel or 1
    if max(tp, dp, ep) > 1 and logical_npus > 1:
        return "local_multi_logical_npu"
    return "single_device"


def _env_profile(
    *,
    logical_npus: int | None,
    tensor_parallel: int | None,
    data_parallel: int | None,
    expert_parallel: int | None,
    quantization_trait: str | None,
) -> str | None:
    comm = _comm_profile(
        logical_npus=logical_npus,
        tensor_parallel=tensor_parallel,
        data_parallel=data_parallel,
        expert_parallel=expert_parallel,
    )
    if comm != "local_multi_logical_npu":
        return "single_device_env" if logical_npus else None
    if quantization_trait:
        return "local_tp_quantized_env"
    return "local_tp_env"


def _artifact_candidate(
    *,
    decision_kind: str,
    context: SelectorContext,
    artifact_path_kind: str,
    quantization_trait: str | None,
    tool_name: str | None,
    serving_quantization: str | None,
    summary: str,
    source_refs: Iterable[str],
    artifact_refs: Iterable[str],
    documented: bool,
    unvalidated: bool,
    confidence: str,
) -> ArtifactCandidate:
    return ArtifactCandidate(
        decision_kind=decision_kind,
        model_base=context.model_base,
        model_traits=context.model_traits,
        hw=context.hw,
        artifact_path_kind=artifact_path_kind,
        quantization_trait=quantization_trait,
        tool_name=tool_name,
        serving_quantization=serving_quantization,
        summary=summary,
        source_refs=tuple(source_refs),
        artifact_refs=tuple(artifact_refs),
        documented=documented,
        unvalidated=unvalidated,
        confidence=confidence,
    )


def _candidate(
    *,
    decision_kind: str,
    context: SelectorContext,
    baseline: BaselineValidation | None,
    physical_cards: int | None,
    logical_npus: int | None,
    tensor_parallel: int | None,
    data_parallel: int | None,
    expert_parallel: int | None,
    parallelism_family: str | None,
    render_preset: str | None,
    summary: str,
    source_refs: Iterable[str],
    artifact_refs: Iterable[str],
    documented: bool,
    unvalidated: bool,
    confidence: str,
    model_base: str | None = None,
    model_traits: Iterable[str] | None = None,
    hw: str | None = None,
) -> StrategyCandidate:
    return StrategyCandidate(
        decision_kind=decision_kind,
        model_base=model_base or context.model_base,
        model_traits=_sorted_traits(model_traits or context.model_traits),
        hw=hw or context.hw,
        physical_cards=physical_cards,
        logical_npus=logical_npus,
        tensor_parallel=tensor_parallel,
        data_parallel=data_parallel,
        expert_parallel=expert_parallel,
        parallelism_family=parallelism_family,
        render_preset=render_preset or (baseline.render_preset if baseline else None),
        validation_id=baseline.validation_id if baseline else None,
        summary=summary,
        source_refs=tuple(source_refs),
        artifact_refs=tuple(artifact_refs),
        documented=documented,
        unvalidated=unvalidated,
        confidence=confidence,
        comm_profile=_comm_profile(
            logical_npus=logical_npus,
            tensor_parallel=tensor_parallel,
            data_parallel=data_parallel,
            expert_parallel=expert_parallel,
        ),
        env_profile=_env_profile(
            logical_npus=logical_npus,
            tensor_parallel=tensor_parallel,
            data_parallel=data_parallel,
            expert_parallel=expert_parallel,
            quantization_trait=_requested_quantization_trait(context),
        ),
    )


def _documented_candidate(context: SelectorContext, baseline: BaselineValidation) -> StrategyCandidate:
    summary = (
        f"documented baseline on {baseline.hw}: TP{baseline.tensor_parallel or 0}"
        f" / {baseline.physical_cards or '?'} cards / {baseline.logical_npus or '?'} logical NPUs"
    )
    return _candidate(
        decision_kind="documented_baseline",
        context=context,
        baseline=baseline,
        physical_cards=baseline.physical_cards,
        logical_npus=baseline.logical_npus,
        tensor_parallel=baseline.tensor_parallel,
        data_parallel=baseline.data_parallel,
        expert_parallel=baseline.expert_parallel,
        parallelism_family="tp" if baseline.tensor_parallel else None,
        render_preset=baseline.render_preset,
        summary=summary,
        source_refs=baseline.source_refs,
        artifact_refs=baseline.artifact_refs,
        documented=True,
        unvalidated=False,
        confidence="high",
        model_traits=baseline.model_traits,
    )


def select_artifact_path(
    context: SelectorContext,
    baselines: tuple[BaselineValidation, ...],
    *,
    artifact_fact_rows: Iterable[dict[str, Any] | Any] = (),
    tool_recipe_rows: Iterable[dict[str, Any] | Any] = (),
    runtime_constraint_rows: Iterable[dict[str, Any] | Any] = (),
) -> ArtifactSelection | None:
    if context.model_base is None or context.hw is None:
        return None

    matches = _matching_baselines(context, baselines)
    baseline = matches[0] if matches else None
    quant_trait = _requested_quantization_trait(context)
    features = set(context.features)
    warnings: list[str] = []
    unknowns: list[str] = []
    alternatives: list[ArtifactCandidate] = []
    tool_source_refs = [row["fact_id"] for row in tool_recipe_rows]
    artifact_source_refs = [row["fact_id"] for row in artifact_fact_rows]
    runtime_source_refs = [row["fact_id"] for row in runtime_constraint_rows]

    documented: ArtifactCandidate | None = None
    if baseline is not None:
        documented = _artifact_candidate(
            decision_kind="documented_native_deploy",
            context=context,
            artifact_path_kind="native_deploy",
            quantization_trait=quant_trait,
            tool_name=None,
            serving_quantization="ascend" if quant_trait else None,
            summary="documented direct deployment path for a deployable artifact",
            source_refs=baseline.source_refs,
            artifact_refs=baseline.artifact_refs,
            documented=True,
            unvalidated=False,
            confidence="high",
        )

    native_fp8_requested = (
        "weight_fp8_native" in features
        and "weight_quantized" not in features
        and "artifact_modelslim" not in features
        and "weight_fp8_origin" not in features
        and context.hw in {"A2", "A3"}
    )
    if native_fp8_requested:
        unknowns.append("native fp8 direct deployment is unsupported on A2/A3")
        unknowns.append("choose between ModelSlim conversion and fp8-origin adaptation before generating a runbook")
        if quant_trait is None:
            quant_trait = "quant_w8a8"
        if tool_source_refs:
            alternatives.append(
                _artifact_candidate(
                    decision_kind="inferred_convert_then_deploy",
                    context=context,
                    artifact_path_kind="convert_then_deploy",
                    quantization_trait=quant_trait,
                    tool_name="ModelSlim",
                    serving_quantization="ascend",
                    summary="ModelSlim conversion is the supported quantized-artifact route on Ascend, but this exact model recipe is inferred rather than documented.",
                    source_refs=tool_source_refs,
                    artifact_refs=baseline.artifact_refs if baseline else (),
                    documented=False,
                    unvalidated=True,
                    confidence="medium",
                )
            )
        selected = _artifact_candidate(
            decision_kind="unsupported_requires_choice",
            context=context,
            artifact_path_kind="native_deploy",
            quantization_trait="fp8_native",
            tool_name=None,
            serving_quantization=None,
            summary="native fp8 direct deployment is unsupported on A2/A3 and requires a route decision before any runbook can be emitted",
            source_refs=artifact_source_refs or runtime_source_refs,
            artifact_refs=(),
            documented=False,
            unvalidated=True,
            confidence="low",
        )
        return ArtifactSelection(
            selected=selected,
            documented=documented,
            alternatives=tuple(alternatives),
            warnings=tuple(warnings),
            unknowns=tuple(unknowns),
        )

    if "artifact_modelslim" in features:
        selected = _artifact_candidate(
            decision_kind="inferred_convert_then_deploy",
            context=context,
            artifact_path_kind="convert_then_deploy",
            quantization_trait=quant_trait or "quant_w8a8",
            tool_name="ModelSlim",
            serving_quantization="ascend",
            summary="selected ModelSlim conversion + deployment route",
            source_refs=tool_source_refs or artifact_source_refs,
            artifact_refs=baseline.artifact_refs if baseline else (),
            documented=False,
            unvalidated=True,
            confidence="medium",
        )
        if not tool_source_refs:
            warnings.append("ModelSlim route is selected but tool recipe facts are sparse; keeping the runbook inferred")
        return ArtifactSelection(
            selected=selected,
            documented=documented,
            alternatives=(),
            warnings=tuple(warnings),
            unknowns=tuple(unknowns),
        )

    if documented is not None:
        return ArtifactSelection(
            selected=documented,
            documented=documented,
            alternatives=(),
            warnings=tuple(warnings),
            unknowns=tuple(unknowns),
        )

    return None


def select_deployment_strategy(
    context: SelectorContext,
    baselines: tuple[BaselineValidation, ...],
    *,
    topology_multiplier: int | None = None,
) -> StrategySelection | None:
    if context.model_base is None or context.hw is None:
        return None

    matches = _matching_baselines(context, baselines)
    if not matches:
        return None

    baseline = matches[0]
    documented = _documented_candidate(context, baseline)
    warnings: list[str] = []
    unknowns: list[str] = []
    alternatives: list[StrategyCandidate] = []
    if context.logical_npus is None and context.physical_cards is not None and topology_multiplier is not None:
        logical_npus = context.physical_cards * topology_multiplier
        context = SelectorContext(
            model_base=context.model_base,
            model_traits=context.model_traits,
            features=context.features,
            configs=context.configs,
            hw=context.hw,
            physical_cards=context.physical_cards,
            logical_npus=logical_npus,
            topology_locked=context.topology_locked,
            user_priority=context.user_priority,
            requested_parallelism=context.requested_parallelism,
        )

    if not context.topology_locked:
        selected = _candidate(
            decision_kind="best_perf_default",
            context=context,
            baseline=baseline,
            physical_cards=baseline.physical_cards,
            logical_npus=baseline.logical_npus,
            tensor_parallel=baseline.tensor_parallel,
            data_parallel=baseline.data_parallel,
            expert_parallel=baseline.expert_parallel,
            parallelism_family="tp" if baseline.tensor_parallel else None,
            render_preset=baseline.render_preset,
            summary="best-performance documented baseline selected because the user did not lock topology",
            source_refs=baseline.source_refs,
            artifact_refs=baseline.artifact_refs,
            documented=True,
            unvalidated=False,
            confidence="high",
            model_traits=baseline.model_traits,
        )
        return StrategySelection(selected=selected, documented=documented, alternatives=(), warnings=(), unknowns=())

    requested_parallelism = dict(context.requested_parallelism)
    exact_parallel_match = not requested_parallelism or (
        requested_parallelism.get("tp") == baseline.tensor_parallel
        and (
            "dp" not in requested_parallelism
            or requested_parallelism.get("dp") == (baseline.data_parallel or 1)
        )
    )
    if (
        context.physical_cards is not None
        and context.physical_cards == baseline.physical_cards
        and exact_parallel_match
    ):
        selected = _candidate(
            decision_kind="documented_baseline",
            context=context,
            baseline=baseline,
            physical_cards=baseline.physical_cards,
            logical_npus=baseline.logical_npus,
            tensor_parallel=baseline.tensor_parallel,
            data_parallel=baseline.data_parallel,
            expert_parallel=baseline.expert_parallel,
            parallelism_family="tp" if baseline.tensor_parallel else None,
            render_preset=baseline.render_preset,
            summary="requested topology matches the documented baseline",
            source_refs=baseline.source_refs,
            artifact_refs=baseline.artifact_refs,
            documented=True,
            unvalidated=False,
            confidence="high",
            model_traits=baseline.model_traits,
        )
        return StrategySelection(selected=selected, documented=documented, alternatives=(), warnings=(), unknowns=())

    if context.user_priority == "validated_only":
        unknowns.append("validated-only request did not match any documented topology")
        selected = _candidate(
            decision_kind="unknown_or_reroute",
            context=context,
            baseline=baseline,
            physical_cards=context.physical_cards,
            logical_npus=context.logical_npus,
            tensor_parallel=None,
            data_parallel=None,
            expert_parallel=None,
            parallelism_family="ambiguous",
            render_preset=None,
            summary="requested topology is outside the validated matrix",
            source_refs=baseline.source_refs,
            artifact_refs=baseline.artifact_refs,
            documented=False,
            unvalidated=True,
            confidence="low",
        )
        return StrategySelection(selected=selected, documented=documented, alternatives=(), warnings=(), unknowns=tuple(unknowns))

    if requested_parallelism:
        axis_parts = [f"{axis.upper()}{value}" for axis, value in context.requested_parallelism]
        unknowns.append(
            f"explicit parallelism request {' + '.join(axis_parts)} is not documented for the current topology"
        )
        selected = _candidate(
            decision_kind="unknown_or_reroute",
            context=context,
            baseline=baseline,
            physical_cards=context.physical_cards,
            logical_npus=context.logical_npus,
            tensor_parallel=requested_parallelism.get("tp"),
            data_parallel=requested_parallelism.get("dp"),
            expert_parallel=requested_parallelism.get("ep"),
            parallelism_family="ambiguous",
            render_preset=None,
            summary="deployment strategy cannot safely converge with the requested explicit parallelism",
            source_refs=baseline.source_refs,
            artifact_refs=baseline.artifact_refs,
            documented=False,
            unvalidated=True,
            confidence="low",
        )
        return StrategySelection(selected=selected, documented=documented, alternatives=(), warnings=(), unknowns=tuple(unknowns))

    if (
        context.physical_cards is not None
        and context.logical_npus is not None
        and baseline.physical_cards is not None
        and baseline.logical_npus is not None
        and context.logical_npus < baseline.logical_npus
    ):
        tp = context.logical_npus
        warnings.append("requested topology is smaller than the documented baseline; returning an inferred conservative strategy")
        unknowns.append("inferred strategy is unvalidated for the requested topology")
        selected = _candidate(
            decision_kind="inferred_preserve_topology",
            context=context,
            baseline=baseline,
            physical_cards=context.physical_cards,
            logical_npus=context.logical_npus,
            tensor_parallel=tp,
            data_parallel=1,
            expert_parallel=1,
            parallelism_family="tp",
            render_preset=baseline.render_preset,
            summary="preserving the requested smaller topology with a conservative TP-only inference",
            source_refs=baseline.source_refs,
            artifact_refs=baseline.artifact_refs,
            documented=False,
            unvalidated=True,
            confidence="medium",
        )
        return StrategySelection(
            selected=selected,
            documented=documented,
            alternatives=(),
            warnings=tuple(warnings),
            unknowns=tuple(unknowns),
        )

    if (
        context.physical_cards is not None
        and context.logical_npus is not None
        and baseline.logical_npus is not None
        and context.logical_npus > baseline.logical_npus
    ):
        warnings.append("requested topology is larger than the documented baseline")
        alternatives.append(
            _candidate(
                decision_kind="inferred_candidate",
                context=context,
                baseline=baseline,
                physical_cards=context.physical_cards,
                logical_npus=context.logical_npus,
                tensor_parallel=context.logical_npus,
                data_parallel=1,
                expert_parallel=1,
                parallelism_family="tp",
                render_preset=baseline.render_preset,
                summary=f"candidate TP{context.logical_npus} by scaling tensor parallelism to all logical NPUs",
                source_refs=baseline.source_refs,
                artifact_refs=baseline.artifact_refs,
                documented=False,
                unvalidated=True,
                confidence="low",
            )
        )
        if baseline.tensor_parallel and context.logical_npus % baseline.logical_npus == 0:
            alternatives.append(
                _candidate(
                    decision_kind="inferred_candidate",
                    context=context,
                    baseline=baseline,
                    physical_cards=context.physical_cards,
                    logical_npus=context.logical_npus,
                    tensor_parallel=baseline.tensor_parallel,
                    data_parallel=max(1, context.logical_npus // baseline.logical_npus),
                    expert_parallel=1,
                    parallelism_family="dp_tp",
                    render_preset=None,
                    summary=(
                        f"candidate DP{max(1, context.logical_npus // baseline.logical_npus)} + "
                        f"TP{baseline.tensor_parallel} by replicating the documented TP baseline"
                    ),
                    source_refs=baseline.source_refs,
                    artifact_refs=baseline.artifact_refs,
                    documented=False,
                    unvalidated=True,
                    confidence="low",
                )
            )
        unknowns.append(
            "requested topology has multiple plausible strategies (for example TP and DP/TP), so deployment cannot safely choose one"
        )
        selected = _candidate(
            decision_kind="unknown_or_reroute",
            context=context,
            baseline=baseline,
            physical_cards=context.physical_cards,
            logical_npus=context.logical_npus,
            tensor_parallel=None,
            data_parallel=None,
            expert_parallel=None,
            parallelism_family="ambiguous",
            render_preset=None,
            summary="requested topology is larger than the documented baseline and has multiple plausible parallelism strategies",
            source_refs=baseline.source_refs,
            artifact_refs=baseline.artifact_refs,
            documented=False,
            unvalidated=True,
            confidence="low",
        )
        return StrategySelection(
            selected=selected,
            documented=documented,
            alternatives=tuple(alternatives),
            warnings=tuple(warnings),
            unknowns=tuple(unknowns),
        )

    selected = _candidate(
        decision_kind="unknown_or_reroute",
        context=context,
        baseline=baseline,
        physical_cards=context.physical_cards,
        logical_npus=context.logical_npus,
        tensor_parallel=None,
        data_parallel=None,
        expert_parallel=None,
        parallelism_family="ambiguous",
        render_preset=None,
        summary="requested topology cannot be safely projected from the documented baseline",
        source_refs=baseline.source_refs,
        artifact_refs=baseline.artifact_refs,
        documented=False,
        unvalidated=True,
        confidence="low",
    )
    unknowns.append("strategy projection could not converge")
    return StrategySelection(selected=selected, documented=documented, alternatives=(), warnings=(), unknowns=tuple(unknowns))


def _join_traits(traits: Iterable[str]) -> str:
    values = [value for value in traits if value]
    return "+".join(sorted(values)) if values else "none"


def _serialize_value(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value).replace("|", "_").replace("=", "_").replace(",", "_")


def build_strategy_atom(role: str, candidate: StrategyCandidate) -> dict[str, Any]:
    parts = [
        f"kind={candidate.decision_kind}",
        f"model={_serialize_value(candidate.model_base)}",
        f"traits={_join_traits(candidate.model_traits)}",
        f"hw={_serialize_value(candidate.hw)}",
        f"cards={_serialize_value(candidate.physical_cards)}",
        f"logical={_serialize_value(candidate.logical_npus)}",
        f"tp={_serialize_value(candidate.tensor_parallel)}",
        f"dp={_serialize_value(candidate.data_parallel)}",
        f"ep={_serialize_value(candidate.expert_parallel)}",
        f"family={_serialize_value(candidate.parallelism_family)}",
        f"preset={_serialize_value(candidate.render_preset)}",
        f"confidence={candidate.confidence}",
        f"documented={_serialize_value(candidate.documented)}",
        f"unvalidated={_serialize_value(candidate.unvalidated)}",
        f"comm={_serialize_value(candidate.comm_profile)}",
        f"env={_serialize_value(candidate.env_profile)}",
    ]
    return {
        "atom_id": f"{STRATEGY_ATOM_PREFIX}{role}|" + "|".join(parts),
        "atom_kind": "constraint",
        "summary": candidate.summary,
        "source_refs": list(candidate.source_refs or candidate.artifact_refs),
    }


def build_artifact_atom(role: str, candidate: ArtifactCandidate) -> dict[str, Any]:
    parts = [
        f"kind={candidate.decision_kind}",
        f"model={_serialize_value(candidate.model_base)}",
        f"traits={_join_traits(candidate.model_traits)}",
        f"hw={_serialize_value(candidate.hw)}",
        f"path={_serialize_value(candidate.artifact_path_kind)}",
        f"quant={_serialize_value(candidate.quantization_trait)}",
        f"tool={_serialize_value(candidate.tool_name)}",
        f"serve_quant={_serialize_value(candidate.serving_quantization)}",
        f"confidence={candidate.confidence}",
        f"documented={_serialize_value(candidate.documented)}",
        f"unvalidated={_serialize_value(candidate.unvalidated)}",
    ]
    return {
        "atom_id": f"{ARTIFACT_ATOM_PREFIX}{role}|" + "|".join(parts),
        "atom_kind": "constraint",
        "summary": candidate.summary,
        "source_refs": list(candidate.source_refs or candidate.artifact_refs),
    }


def parse_strategy_atom_id(atom_id: str) -> dict[str, str]:
    if not atom_id.startswith(STRATEGY_ATOM_PREFIX):
        return {}
    prefix, _, payload = atom_id.partition("|")
    values = {"role": prefix.split(":", 1)[1]}
    for item in payload.split("|"):
        key, _, value = item.partition("=")
        if key:
            values[key] = value
    return values


def parse_artifact_atom_id(atom_id: str) -> dict[str, str]:
    if not atom_id.startswith(ARTIFACT_ATOM_PREFIX):
        return {}
    prefix, _, payload = atom_id.partition("|")
    values = {"role": prefix.split(":", 1)[1]}
    for item in payload.split("|"):
        key, _, value = item.partition("=")
        if key:
            values[key] = value
    return values


def selected_strategy_from_atoms(atoms: Iterable[dict[str, Any]]) -> dict[str, str] | None:
    for atom in atoms:
        atom_id = atom.get("atom_id", "")
        if atom_id.startswith("strategy:selected|"):
            return parse_strategy_atom_id(atom_id)
    return None


def documented_strategy_from_atoms(atoms: Iterable[dict[str, Any]]) -> dict[str, str] | None:
    for atom in atoms:
        atom_id = atom.get("atom_id", "")
        if atom_id.startswith("strategy:documented|"):
            return parse_strategy_atom_id(atom_id)
    return None


def alternative_strategies_from_atoms(atoms: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    alternatives: list[dict[str, str]] = []
    for atom in atoms:
        atom_id = atom.get("atom_id", "")
        if atom_id.startswith("strategy:alternative|"):
            alternatives.append(parse_strategy_atom_id(atom_id))
    return alternatives


def selected_artifact_from_atoms(atoms: Iterable[dict[str, Any]]) -> dict[str, str] | None:
    for atom in atoms:
        atom_id = atom.get("atom_id", "")
        if atom_id.startswith("artifact:selected|"):
            return parse_artifact_atom_id(atom_id)
    return None


def documented_artifact_from_atoms(atoms: Iterable[dict[str, Any]]) -> dict[str, str] | None:
    for atom in atoms:
        atom_id = atom.get("atom_id", "")
        if atom_id.startswith("artifact:documented|"):
            return parse_artifact_atom_id(atom_id)
    return None


def alternative_artifacts_from_atoms(atoms: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    alternatives: list[dict[str, str]] = []
    for atom in atoms:
        atom_id = atom.get("atom_id", "")
        if atom_id.startswith("artifact:alternative|"):
            alternatives.append(parse_artifact_atom_id(atom_id))
    return alternatives
