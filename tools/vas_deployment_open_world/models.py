from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class RequestFacts:
    raw_text: str
    intent: str = 'deployment'
    model_family: str | None = None
    model_variant: str | None = None
    model_size_b: float | None = None
    hardware: str | None = None
    cards: int | None = None
    deployment_form: str = 'single_instance'
    weight_path: str | None = None
    quantization: str | None = None
    avg_input_tokens: int | None = None
    avg_output_tokens: int | None = None
    max_context_tokens: int | None = None
    tpot_ms: float | None = None
    explicit_pref: str | None = None
    existing_quantized_weights: bool | None = None
    accepts_candidate: bool = False
    alias_suspect: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceAtom:
    evidence_id: str
    subject: str
    predicate: str
    value: str
    source_ref: str
    source_tier: str
    polarity: str
    note: str = ''

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Recipe:
    recipe_id: str
    subject: str
    hardware: list[str]
    scenario_kind: str
    topology: dict[str, Any]
    feature_policy: dict[str, Any]
    source_ref: str
    note: str = ''

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DeploymentResult:
    result_class: str
    scenario: dict[str, Any]
    resolved_subject: dict[str, Any]
    evidence_summary: list[dict[str, Any]] = field(default_factory=list)
    required_questions: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    why: list[str] = field(default_factory=list)
    launch_candidates: list[dict[str, Any]] = field(default_factory=list)
    validation_checklist: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
