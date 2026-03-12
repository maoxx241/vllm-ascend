from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .contracts import now_utc
from .paths import repo_root, workspace_root


def _run(command: list[str], cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(command, cwd=str(cwd) if cwd else None, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _git_value(root: Path, *args: str) -> str:
    return _run(["git", *args], cwd=root)


def _module_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except Exception:
        return "unknown"
    return getattr(module, "__version__", "unknown")


def _detect_cann_version() -> str:
    candidates = [
        Path("/usr/local/Ascend/ascend-toolkit/latest/version.cfg"),
        Path("/usr/local/Ascend/cann-8.5.0/aarch64-linux/ascend_toolkit_install.info"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        content = candidate.read_text(encoding="utf-8", errors="ignore")
        for line in content.splitlines():
            if line.startswith("version="):
                return line.split("=", 1)[1].strip()
    return "unknown"


def _parse_board_info() -> tuple[str, str]:
    board_info = _run(["npu-smi", "info", "-t", "board", "-i", "0", "-c", "0"])
    chip_name = ""
    chip_type = ""
    for line in board_info.splitlines():
        if "Chip Name" in line:
            chip_name = line.split(":", 1)[1].strip()
        if "Chip Type" in line:
            chip_type = line.split(":", 1)[1].strip()
    if "910" in chip_name and chip_type:
        return "A2", chip_name
    if "910" in chip_name:
        return "A3", chip_name
    return "unknown", chip_name or "unknown"


def collect_runtime_context(root: Path | None = None, overrides: dict[str, str] | None = None) -> dict[str, Any]:
    root = root or repo_root()
    overrides = overrides or {}
    workspace = workspace_root(root)
    paired_vllm_root = workspace / "vllm"
    soc, raw_soc = _parse_board_info()
    if overrides.get("soc"):
        soc = overrides["soc"]
    if overrides.get("raw_soc"):
        raw_soc = overrides["raw_soc"]

    runtime_tuple = {
        "soc": soc,
        "cann": overrides.get("cann") or _detect_cann_version(),
        "torch": overrides.get("torch") or _module_version("torch"),
        "torch_npu": overrides.get("torch_npu") or _module_version("torch_npu"),
        "python": overrides.get("python") or f"{sys.version_info.major}.{sys.version_info.minor}",
    }
    repo_sha = overrides.get("repo_sha") or _git_value(root, "rev-parse", "HEAD")
    repo_branch = overrides.get("repo_branch") or _git_value(root, "branch", "--show-current") or "detached"
    paired_vllm_ref = overrides.get("paired_vllm_ref")
    if not paired_vllm_ref and paired_vllm_root.exists():
        paired_vllm_ref = _git_value(paired_vllm_root, "rev-parse", "HEAD")
    if not paired_vllm_ref:
        paired_vllm_ref = "unknown"

    return {
        "created_at": now_utc(),
        "repo_branch": repo_branch,
        "repo_sha": repo_sha,
        "paired_vllm_ref": paired_vllm_ref,
        "runtime_tuple": runtime_tuple,
        "raw_soc": raw_soc,
        "platform": platform.platform(),
    }


def serialize_runtime_context(root: Path | None = None, overrides: dict[str, str] | None = None) -> str:
    return json.dumps(collect_runtime_context(root=root, overrides=overrides), ensure_ascii=False, indent=2)
