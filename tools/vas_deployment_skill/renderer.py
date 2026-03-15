from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import DeploymentResult


def write_bundle(bundle_dir: Path, result: DeploymentResult, *, request_text: str, case_id: str | None = None) -> dict[str, str]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir = bundle_dir / 'scripts'
    scripts_dir.mkdir(exist_ok=True)

    result_json = bundle_dir / 'result.json'
    result_json.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')

    report_md = bundle_dir / 'decision_report.md'
    report_md.write_text(_render_report(result, request_text, case_id=case_id), encoding='utf-8')

    validation_md = bundle_dir / 'validation_checklist.md'
    validation_md.write_text(_render_validation(result), encoding='utf-8')

    scripts: list[str] = []
    for idx, cand in enumerate(result.launch_candidates, start=1):
        path = scripts_dir / f'{idx:02d}_{cand.name}.sh'
        path.write_text(_render_script(cand.command, cand.env, cand.name), encoding='utf-8')
        scripts.append(str(path))

    manifest = {
        'case_id': case_id or 'adhoc',
        'result_json': str(result_json),
        'decision_report_md': str(report_md),
        'validation_checklist_md': str(validation_md),
        'scripts': scripts,
    }
    (bundle_dir / 'bundle_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


def _render_report(result: DeploymentResult, request_text: str, *, case_id: str | None = None) -> str:
    lines = ['# Deployment decision report', '']
    if case_id:
        lines += [f'- case_id: `{case_id}`']
    lines += [f'- result_class: `{result.result_class}`', '']
    lines += ['## User request', '', request_text, '']
    lines += ['## Resolved subject', '', '```json', json.dumps(result.resolved_subject, ensure_ascii=False, indent=2), '```', '']

    if result.assumptions:
        lines += ['## Assumptions', ''] + [f'- {x}' for x in result.assumptions] + ['']
    if result.why_not_exact:
        lines += ['## Why not exact', ''] + [f'- {x}' for x in result.why_not_exact] + ['']
    if result.required_questions:
        lines += ['## Required questions', ''] + [f'- {x}' for x in result.required_questions] + ['']
    if result.derived_metrics:
        lines += ['## Derived metrics', '', '```json', json.dumps(result.derived_metrics, ensure_ascii=False, indent=2), '```', '']
    if result.evidence_summary:
        lines += ['## Evidence summary', '']
        for row in result.evidence_summary:
            lines.append(f"- `{row['evidence_id']}` | `{row['source_tier']}` | `{row['polarity']}` | {row['predicate']} | {row['source_ref']}")
            if row.get('note'):
                lines.append(f"  - note: {row['note']}")
        lines.append('')
    if result.report_sections:
        lines += ['## Reasoning sections', '']
        for sec in result.report_sections:
            lines += [f"### {sec['title']}", '', sec['content'], '']
    if result.launch_candidates:
        lines += ['## Launch candidates', '']
        for cand in result.launch_candidates:
            lines += [f"### {cand.name}", '', f"- risk_level: `{cand.risk_level}`", '', '```bash', cand.command, '```', '']
            if cand.env:
                lines += ['Environment:'] + [f"- `{k}={v}`" for k, v in cand.env.items()] + ['']
            if cand.rationale:
                lines += ['Rationale:'] + [f"- {x}" for x in cand.rationale] + ['']
    return '\n'.join(lines) + '\n'


def _render_validation(result: DeploymentResult) -> str:
    lines = ['# Validation checklist', '']
    if result.validation_checklist:
        lines += [f'- {x}' for x in result.validation_checklist]
    else:
        lines += ['- No validation checklist emitted for this result class.']
    lines.append('')
    return '\n'.join(lines)


def _render_script(command: str, env: dict[str, str], name: str) -> str:
    lines = ['#!/usr/bin/env bash', 'set -euo pipefail', '', f'# candidate: {name}', '']
    for k, v in env.items():
        lines.append(f'export {k}={json.dumps(v)}')
    if env:
        lines.append('')
    lines += [command, '']
    return '\n'.join(lines)
