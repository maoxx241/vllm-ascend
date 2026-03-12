from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vllm_ascend.agent_runtime.kb import build_local, resolve


@pytest.fixture(scope="session")
def agent_repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def exact_resolve_result(agent_repo_root: Path) -> dict:
    return resolve(
        agent_repo_root,
        request_id="req-test-exact",
        overrides={
            "soc": "A2",
            "cann": "8.5.0",
            "torch": "2.9.0",
            "torch_npu": "2.9.0",
            "python": "3.10",
            "repo_sha": "fe4cad24e9efa97235a5ebff10b62d8a4d981ddc",
            "paired_vllm_ref": "e39257a552d18ae9abb6ba1bbe65865d385ea764",
        },
    )


@pytest.fixture()
def built_sqlite(tmp_path: Path, agent_repo_root: Path, exact_resolve_result: dict) -> Path:
    emit_sqlite = tmp_path / "current.sqlite"
    build_local(agent_repo_root, resolve_result=exact_resolve_result, emit_sqlite=emit_sqlite)
    return emit_sqlite
