# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import importlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import regex as re

ROOT = Path(__file__).resolve().parents[3]


def _compiler_options(adapter, monkeypatch):
    tree = ast.parse(adapter.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "relocation_probe")
    body = []
    for statement in function.body:
        body.append(statement)
        if (
            isinstance(statement, ast.AugAssign)
            and isinstance(statement.value, ast.Name)
            and statement.value.id == "custom_compile_options_soc"
        ):
            break
    else:
        raise AssertionError("generated adapter did not assemble compiler options")
    function.body = body + [ast.Return(value=ast.Name(id="options", ctx=ast.Load()))]
    function.decorator_list = []
    selector = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_shortsoc_compile_option"
    )
    namespace = {
        "os": os,
        "shutil": shutil,
        "re": re,
        "PYF_PATH": str(adapter.parent),
        "get_current_build_config": lambda key: False,
        "get_soc_spec": lambda key: "ascend950",
        "_build_args": lambda *args: ([], [], []),
        "get_dtype_fmt_options": lambda inputs, outputs: [],
    }
    monkeypatch.setenv("BISHENG_REAL_PATH", str(adapter.parent / "unused_toolkit/bin/bisheng"))
    monkeypatch.setenv("ASCEND_HOME_PATH", str(adapter.parent / "unused_toolkit"))
    module = ast.fix_missing_locations(ast.Module(body=[selector, function], type_ignores=[]))
    exec(compile(module, str(adapter), "exec"), namespace)
    return namespace["relocation_probe"](None)


@pytest.mark.parametrize("install_source", ["csrc/CMakeLists.txt", "csrc/cmake/custom_build.cmake"])
def test_cann_compat_survives_package_relocation(tmp_path, monkeypatch, install_source):
    """Exercise real CMake packaging rules and generated options without CANN/NPU.

    Delete the build source after installing, then preprocess using the moved
    adapter's options. An absolute build-path include must fail this regression.
    """
    cmake = shutil.which("cmake")
    compiler = shutil.which("c++") or shutil.which("g++")
    if cmake is None or compiler is None:
        pytest.skip("CMake and a host C++ preprocessor are required")
    cmake_source = (ROOT / "csrc/CMakeLists.txt").read_text()
    options_block = cmake_source.split("# Suppress warnings from", 1)[1].split("add_compile_options(", 1)[0]
    options_block = "# Suppress warnings from" + options_block
    copy_rule = re.search(r"configure_file\(\$\{VLLM_ASCEND_CANN_COMPAT_HEADER\}.*?COPYONLY\)", cmake_source, re.S)
    install_rule = re.search(
        r"install\(FILES \$\{VLLM_ASCEND_CANN_COMPAT_HEADER\}.*?\)", (ROOT / install_source).read_text(), re.S
    )
    assert copy_rule is not None and install_rule is not None
    source = tmp_path / "original_source"
    header = source / "csrc/common/include/cann_compat.h"
    header.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "csrc/common/include/cann_compat.h", header)
    build = tmp_path / "build"
    stage = tmp_path / "stage"
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\nproject(CannCompat NONE)\n"
        'set(OPS_TRANSFORMER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/csrc")\n'
        'set(ASCEND_IMPL_OUT_DIR "${CMAKE_BINARY_DIR}/impl")\n'
        'set(IMPL_INSTALL_DIR "vendor_impl")\n'
        + options_block
        + copy_rule.group(0)
        + "\n"
        + install_rule.group(0)
        + '\nfile(WRITE "${CMAKE_BINARY_DIR}/options.txt" "${OPS_COMPILE_OPTIONS}")\n'
        'install(FILES "${ASCEND_IMPL_OUT_DIR}/dynamic/relocation_probe.py" DESTINATION vendor_impl/dynamic)\n'
    )
    subprocess.run([cmake, "-S", str(source), "-B", str(build), f"-DCMAKE_INSTALL_PREFIX={stage}"], check=True)
    assert (build / "impl/ascendc/common/cann_compat.h").read_bytes() == header.read_bytes()
    monkeypatch.syspath_prepend(str(ROOT / "csrc/cmake/scripts/util"))
    builder_module = importlib.import_module("ascendc_impl_build")
    builder = builder_module.AdpBuilder("RelocationProbe")
    builder.op_file = builder.op_intf = "relocation_probe"
    builder.kern_name = "relocation_probe"
    builder.dynamic_shape = True
    for setting in ["input0.name=x", "input0.paramType=required", "input0.dtype=float32", "input0.format=ND"]:
        builder.parse_input(setting)
    builder.custom_all_compile_options = {"__ALLSOC__": (build / "options.txt").read_text().split(";")}
    builder.write_adapt("", str(build / "impl"))
    subprocess.run([cmake, "--install", str(build)], check=True)
    relocated = tmp_path / "relocated package"
    stage.rename(relocated)
    shutil.rmtree(source)
    shutil.rmtree(build)
    adapter = relocated / "vendor_impl/dynamic/relocation_probe.py"
    options = _compiler_options(adapter, monkeypatch)
    include_options = [option for option in options if option.startswith(("-I", "-include"))]
    command = [compiler, "-E", "-P", "-x", "c++", *include_options, "-"]
    result = subprocess.run(command, input="int module_id = OP;\n", text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "int module_id = 63;" in result.stdout
    (relocated / "vendor_impl/ascendc/common/cann_compat.h").unlink()
    missing = subprocess.run(command, input="", text=True, capture_output=True)
    assert missing.returncode != 0 and "cann_compat.h" in missing.stderr
