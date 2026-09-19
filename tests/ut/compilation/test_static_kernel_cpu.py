# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/compilation/static_kernel_cpu.py"
SPEC = importlib.util.spec_from_file_location("static_kernel_cpu", SOURCE)
cpu = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cpu)


@pytest.mark.parametrize("value, expected", [("", set()), ("0,2-4,8\n", {0, 2, 3, 4, 8})])
def test_parse_cpu_list(value, expected):
    assert cpu._parse_cpu_list(value) == expected


@pytest.mark.parametrize("isolated, expected", [("1-3", {0, 4}), ("", {0, 1, 2, 3, 4}), ("0-4", {0, 1, 2, 3, 4})])
def test_compiler_respects_affinity(monkeypatch, isolated, expected):
    monkeypatch.setattr(cpu.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3, 4}, raising=False)
    monkeypatch.setattr(cpu.Path, "read_text", lambda self: isolated)
    assert cpu._compiler_cpus() == expected


@pytest.mark.parametrize("requested, expected", [(384, "3"), (2, "2")])
def test_only_compiler_command_and_jobs_change(monkeypatch, requested, expected):
    monkeypatch.setattr(cpu, "_compiler_cpus", lambda: {0, 48, 49})
    original = ["op_compiler", "-p", "input with spaces", "-j", str(requested), "--enable_super_kernel"]
    command = cpu._compiler_command(original)
    assert command[:3] == [sys.executable, str(SOURCE), "0,48,49"]
    assert command[3:] == ["op_compiler", "-p", "input with spaces", "-j", expected, "--enable_super_kernel"]
    assert original[4] == str(requested)
    other = ["bash", "operator.run", "--install"]
    assert cpu._compiler_command(other) is other


def test_backend_patch_is_local_and_idempotent(monkeypatch):
    calls = []
    original = SimpleNamespace(run=lambda *args, **kwargs: calls.append((args, kwargs)) or 7, PIPE=-1)
    backend = SimpleNamespace(__name__="test_backend")
    static_kernel = SimpleNamespace(subprocess=original)
    monkeypatch.setitem(sys.modules, "test_backend._acl_concrete_graph.static_kernel", static_kernel)
    cpu.configure_static_kernel_cpu(backend)
    proxy = static_kernel.subprocess
    cpu.configure_static_kernel_cpu(backend)
    assert static_kernel.subprocess is proxy
    assert proxy.original is original and proxy.PIPE == -1
    assert proxy.run(["bash", "install.run"], check=True, capture_output=True) == 7
    assert calls == [((["bash", "install.run"],), {"check": True, "capture_output": True})]


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="Linux CPU affinity API")
def test_launcher_changes_child_affinity_only():
    parent = set(os.sched_getaffinity(0))
    selected = sorted(parent)[:2]
    code = "import json,os; print(json.dumps(sorted(os.sched_getaffinity(0))))"
    result = subprocess.run(
        [sys.executable, str(SOURCE), ",".join(map(str, selected)), sys.executable, "-c", code],
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout) == selected
    assert set(os.sched_getaffinity(0)) == parent
