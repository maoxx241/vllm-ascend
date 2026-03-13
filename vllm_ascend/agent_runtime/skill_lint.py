from __future__ import annotations

from pathlib import Path

from .paths import repo_root

CANONICAL_SKILLS = {
    "vllm-ascend-assistant": {
        "required": [
            "Use `runtime.py` first",
            "Do not grep raw docs first",
        ],
        "forbidden": [
            "1 card = 2 logical NPUs",
            "native fp8 is supported",
        ],
    },
    "deployment_execution": {
        "required": [
            "runtime",
            "capsule",
            "selected artifact",
            "selected strategy",
            "design_analysis",
        ],
        "forbidden": [
            "1 card = 2 logical NPUs",
            "grep raw docs first",
        ],
    },
    "deployment-config-synthesizer": {
        "required": [
            "capsule",
            "selected artifact",
            "selected strategy",
            "runtime",
        ],
        "forbidden": [
            "1 card = 2 logical NPUs",
            "raw docs",
        ],
    },
    "model-expected-performance-estimator": {
        "required": [
            "capsule",
            "runtime",
            "envelope",
        ],
        "forbidden": [
            "raw docs first",
        ],
    },
    "design_analysis": {
        "required": [
            "runtime.py",
            "route selection",
        ],
        "forbidden": [
            "raw docs first",
        ],
    },
}


def lint_runtime_first_skills(root: Path | None = None) -> list[str]:
    root = root or repo_root()
    findings: list[str] = []
    skills_root = root / ".agents" / "skills"
    for skill_name, rules in CANONICAL_SKILLS.items():
        skill_path = skills_root / skill_name / "SKILL.md"
        runtime_path = skills_root / skill_name / "runtime.py"
        if not skill_path.exists():
            findings.append(f"missing skill doc: {skill_name}")
            continue
        if not runtime_path.exists():
            findings.append(f"missing runtime.py: {skill_name}")
        text = skill_path.read_text(encoding="utf-8")
        lowered = text.lower()
        for phrase in rules["required"]:
            if phrase.lower() not in lowered:
                findings.append(f"{skill_name} missing required guardrail: {phrase}")
        for phrase in rules["forbidden"]:
            if phrase.lower() in lowered:
                findings.append(f"{skill_name} embeds forbidden truth source: {phrase}")
    return findings


__all__ = ["lint_runtime_first_skills"]
