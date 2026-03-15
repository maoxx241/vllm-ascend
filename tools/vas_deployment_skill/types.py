from __future__ import annotations

import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

DATACLASS_KWARGS = {'slots': True} if sys.version_info >= (3, 10) else {}

ResultClass = Literal[
    'exact_verified',
    'exact_derived',
    'compatible',
    'candidate',
    'blocked.identity',
    'blocked.resource',
    'blocked.hard_negative',
    'blocked.user_only_fact',
    'blocked.scope_mismatch',
    'blocked.conflict',
]

SourceTier = Literal['local_source', 'local_docs', 'upstream_repo_mirror', 'user_asserted', 'derived']
Polarity = Literal['positive', 'negative', 'constraint', 'hint']


@dataclass(**DATACLASS_KWARGS)
class EvidenceAtom:
    evidence_id: str
    subject: str
    predicate: str
    value: str
    source_tier: SourceTier
    polarity: Polarity
    source_ref: str
    note: str = ''
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(**DATACLASS_KWARGS)
class Recipe:
    recipe_id: str
    subject: str
    scenario: str
    evidence_refs: list[str]
    command_template: str
    env: dict[str, str] = field(default_factory=dict)
    flags: dict[str, Any] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    note: str = ''

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(**DATACLASS_KWARGS)
class ParsedRequest:
    raw_text: str
    intent: str
    model_name: str | None = None
    model_size_b: float | None = None
    hardware: str | None = None
    cards: int | None = None
    quantization: str | None = None
    weight_path: str | None = None
    wants_script: bool = False
    wants_command: bool = False
    objective: str = 'unknown'
    average_input_len: int | None = None
    average_output_len: int | None = None
    max_context: int | None = None
    accepts_experimental: bool = False
    has_existing_quantized_weights: bool | None = None
    single_instance: bool = True
    tpot_limit_ms: int | None = None
    local_weights: bool | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(**DATACLASS_KWARGS)
class LaunchCandidate:
    name: str
    script_kind: str
    command: str
    env: dict[str, str] = field(default_factory=dict)
    risk_level: str = 'medium'
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(**DATACLASS_KWARGS)
class DeploymentResult:
    result_class: ResultClass
    resolved_subject: dict[str, Any]
    evidence_summary: list[dict[str, Any]] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    required_questions: list[str] = field(default_factory=list)
    why_not_exact: list[str] = field(default_factory=list)
    launch_candidates: list[LaunchCandidate] = field(default_factory=list)
    validation_checklist: list[str] = field(default_factory=list)
    report_sections: list[dict[str, str]] = field(default_factory=list)
    derived_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data['launch_candidates'] = [c.to_dict() for c in self.launch_candidates]
        return data
