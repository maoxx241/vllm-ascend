from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any


class CaseWorkspace:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def create_case(self, goal_type: str, user_request: str) -> str:
        case_id = f'case_{uuid.uuid4().hex[:10]}'
        case_dir = self.root / case_id
        (case_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        (case_dir / 'bundles').mkdir(parents=True, exist_ok=True)
        spec = {
            'case_id': case_id,
            'goal_type': goal_type,
            'user_request': user_request,
            'state': 'open',
            'checkpoints': [],
            'open_uncertainties': [],
        }
        (case_dir / 'case.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
        return case_id

    def add_checkpoint(self, case_id: str, phase: str, payload: dict[str, Any]) -> Path:
        case_dir = self.root / case_id
        checkpoint_id = f'ckpt_{uuid.uuid4().hex[:8]}'
        path = case_dir / 'checkpoints' / f'{checkpoint_id}.json'
        path.write_text(json.dumps({'checkpoint_id': checkpoint_id, 'phase': phase, 'payload': payload}, ensure_ascii=False, indent=2), encoding='utf-8')
        spec = json.loads((case_dir / 'case.json').read_text(encoding='utf-8'))
        spec['checkpoints'].append(str(path.relative_to(case_dir)))
        (case_dir / 'case.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
        return path

    def set_uncertainties(self, case_id: str, uncertainties: list[dict[str, Any]]) -> None:
        case_dir = self.root / case_id
        spec = json.loads((case_dir / 'case.json').read_text(encoding='utf-8'))
        spec['open_uncertainties'] = uncertainties
        (case_dir / 'case.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
