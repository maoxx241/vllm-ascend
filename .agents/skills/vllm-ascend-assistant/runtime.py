from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.vas_deployment_skill.assistant_entry import vllm_ascend_assistant as _entry


def vllm_ascend_assistant(workspace_root: str | Path, build_dir: str | Path, text: str, *, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    return _entry(workspace_root=Path(workspace_root), build_dir=Path(build_dir), text=text, overrides=overrides)


__all__ = ['vllm_ascend_assistant']
