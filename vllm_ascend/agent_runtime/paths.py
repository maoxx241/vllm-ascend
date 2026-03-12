from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def workspace_root(root: Path | None = None) -> Path:
    return (root or repo_root()).parent


def kb_root(root: Path | None = None) -> Path:
    return (root or repo_root()) / ".agents" / "kb"


def design_root(root: Path | None = None) -> Path:
    return (root or repo_root()) / ".agents" / "design" / "v3_3-final"


def tasks_root(root: Path | None = None) -> Path:
    return (root or repo_root()) / ".agents" / "tasks"
