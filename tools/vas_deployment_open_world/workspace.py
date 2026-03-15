from __future__ import annotations

import json
from pathlib import Path
from .models import RequestFacts, DeploymentResult


def write_workspace(out_dir: str | Path, request: RequestFacts, result: DeploymentResult) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'request.json').write_text(json.dumps(request.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'checkpoint.json').write_text(json.dumps({'result_class': result.result_class, 'scenario': result.scenario}, ensure_ascii=False, indent=2), encoding='utf-8')
    return out
