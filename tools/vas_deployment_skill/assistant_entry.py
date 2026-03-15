from __future__ import annotations

from pathlib import Path
from typing import Any

from .engine import EvidenceStore, OpenWorldDeploymentEngine
from .parser import parse_request
from .renderer import write_bundle
from .workspace import CaseWorkspace


def vllm_ascend_assistant(*, workspace_root: Path, build_dir: Path, text: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Default entry. Bootstrap + self-acquire-first route for deployment.

    This entry deliberately does not own the full session. It performs:
    1. request normalization
    2. light routing
    3. deployment evaluation when the request is deployment-like
    4. checkpoint + bundle writeback
    """
    parsed = parse_request(text, overrides=overrides)
    ws = CaseWorkspace(workspace_root)
    case_id = ws.create_case(parsed.intent, text)
    ws.add_checkpoint(case_id, 'entry', {'parsed_request': parsed.to_dict()})

    likely_deployment = (
        parsed.intent == 'deployment'
        or parsed.model_name is not None
        or parsed.hardware is not None
        or parsed.weight_path is not None
        or parsed.quantization is not None
        or parsed.objective != 'unknown'
    )
    if not likely_deployment:
        ws.set_uncertainties(case_id, [{'statement': '当前实现只完整覆盖 deployment 主线。', 'impact': 'medium'}])
        return {
            'case_id': case_id,
            'route': 'needs_alignment',
            'result': {
                'result_class': 'needs_alignment',
                'message': '当前入口只完整路由 deployment 场景。',
            },
        }

    engine = OpenWorldDeploymentEngine(EvidenceStore(build_dir))
    result = engine.evaluate(parsed)
    ws.add_checkpoint(case_id, 'deployment_result', result.to_dict())
    ws.set_uncertainties(case_id, [{'statement': q, 'impact': 'blocker'} for q in result.required_questions])
    bundle_dir = workspace_root / case_id / 'bundles' / 'deployment'
    manifest = write_bundle(bundle_dir, result, request_text=text, case_id=case_id)
    return {
        'case_id': case_id,
        'route': 'deployment-synthesis',
        'result': result.to_dict(),
        'bundle_manifest': manifest,
    }
