# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Launch static-kernel compilers outside isolated inference CPUs."""

import importlib
import os
import sys
from pathlib import Path


def _parse_cpu_list(value: str) -> set[int]:
    cpus: set[int] = set()
    for part in value.strip().split(","):
        if part:
            bounds = [int(item) for item in part.split("-")]
            cpus.update(range(bounds[0], bounds[-1] + 1))
    return cpus


def _compiler_cpus() -> set[int]:
    allowed = set(os.sched_getaffinity(0))
    try:
        isolated = _parse_cpu_list(Path("/sys/devices/system/cpu/isolated").read_text())
    except FileNotFoundError:
        isolated = set()
    # Respect an explicitly restricted worker affinity; never widen it.
    return allowed - isolated or allowed


def _compiler_command(args):
    if not isinstance(args, (list, tuple)) or not args or os.path.basename(args[0]) != "op_compiler":
        return args
    cpus = _compiler_cpus()
    command = list(args)
    jobs_index = command.index("-j") + 1
    command[jobs_index] = str(min(int(command[jobs_index]), len(cpus)))
    # Set affinity in a fresh interpreter, before exec/fork of the compiler
    # pool. preexec_fn is unsafe in the multithreaded inference worker.
    return [sys.executable, str(Path(__file__).resolve()), ",".join(map(str, sorted(cpus))), *command]


class _CompilerSubprocess:
    """Only intercept this backend module's op_compiler subprocess calls."""

    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def run(self, args, *popenargs, **kwargs):
        return self.original.run(_compiler_command(args), *popenargs, **kwargs)


def configure_static_kernel_cpu(backend) -> None:
    static_kernel = importlib.import_module(f"{backend.__name__}._acl_concrete_graph.static_kernel")
    if not isinstance(static_kernel.subprocess, _CompilerSubprocess):
        # Do not replace subprocess.run globally: model workers and other
        # subprocess users retain their own affinity and launch arguments.
        static_kernel.subprocess = _CompilerSubprocess(static_kernel.subprocess)


if __name__ == "__main__":
    os.sched_setaffinity(0, _parse_cpu_list(sys.argv[1]))
    os.execvp(sys.argv[2], sys.argv[2:])
