from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.vas_deployment_skill.engine import evaluate_text
from tools.vas_deployment_skill.renderer import write_bundle


def evaluate_deployment(build_dir: str | Path, text: str, *, bundle_dir: str | Path, overrides: dict[str, Any] | None = None, case_id: str = 'adhoc') -> dict[str, Any]:
    result = evaluate_text(Path(build_dir), text, overrides=overrides)
    manifest = write_bundle(Path(bundle_dir), result, request_text=text, case_id=case_id)
    return {'result': result.to_dict(), 'bundle_manifest': manifest}


__all__ = ['evaluate_deployment']
