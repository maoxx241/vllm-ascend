from __future__ import annotations

import json
from pathlib import Path
from .models import DeploymentResult


def render_decision_report(result: DeploymentResult) -> str:
    lines = ['# Deployment decision report', '']
    lines.append(f'- result_class: `{result.result_class}`')
    lines.append(f'- scenario: `{json.dumps(result.scenario, ensure_ascii=False)}`')
    lines.append('')
    if result.why:
        lines.append('## Why')
        for item in result.why:
            lines.append(f'- {item}')
        lines.append('')
    if result.assumptions:
        lines.append('## Assumptions')
        for item in result.assumptions:
            lines.append(f'- {item}')
        lines.append('')
    if result.blockers:
        lines.append('## Blockers')
        for item in result.blockers:
            lines.append(f'- {item}')
        lines.append('')
    if result.required_questions:
        lines.append('## Required questions')
        for item in result.required_questions:
            lines.append(f'- {item}')
        lines.append('')
    if result.evidence_summary:
        lines.append('## Evidence summary')
        for item in result.evidence_summary:
            lines.append(f"- {item['subject']} :: {item['predicate']} :: {item['value']} ({item['source_ref']})")
        lines.append('')
    if result.launch_candidates:
        lines.append('## Launch candidates')
        for item in result.launch_candidates:
            lines.append(f"- {item['name']}: topology={item['topology']}")
            lines.append('```bash')
            lines.append(item['command'])
            lines.append('```')
        lines.append('')
    return '\n'.join(lines).strip() + '\n'


def render_validation_checklist(result: DeploymentResult) -> str:
    lines = ['# Validation checklist', '']
    items = result.validation_checklist or ['Confirm startup succeeds.']
    for item in items:
        lines.append(f'- [ ] {item}')
    return '\n'.join(lines).strip() + '\n'


def write_bundle(out_dir: str | Path, result: DeploymentResult) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'result.json').write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'decision_report.md').write_text(render_decision_report(result), encoding='utf-8')
    (out / 'validation_checklist.md').write_text(render_validation_checklist(result), encoding='utf-8')
    if not result.result_class.startswith('blocked') and result.result_class != 'needs_alignment':
        scripts = out / 'scripts'
        scripts.mkdir(exist_ok=True)
        for idx, cand in enumerate(result.launch_candidates):
            name = 'launch_primary.sh' if idx == 0 else f'launch_{idx+1}.sh'
            scripts.joinpath(name).write_text('#!/usr/bin/env bash\nset -euo pipefail\n\n' + cand['command'] + '\n', encoding='utf-8')
    return out
