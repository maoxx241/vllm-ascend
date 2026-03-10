#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

import json
import multiprocessing
import os
import socket
import threading
import time
from collections.abc import Iterable
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}
_HOSTNAME = socket.gethostname()


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in _TRUE_VALUES


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def monotonic_us() -> int:
    return time.monotonic_ns() // 1000


def format_cpu_list(cpus: Iterable[int]) -> str:
    ordered = sorted(set(cpus))
    if not ordered:
        return ""

    ranges: list[str] = []
    start = prev = ordered[0]
    for cpu in ordered[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = cpu
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def emit_diag(channel: str, event: str, enabled: bool, **fields: Any) -> None:
    if not enabled:
        return

    payload: dict[str, Any] = {
        "event": event,
        "host": _HOSTNAME,
        "pid": os.getpid(),
        "process": multiprocessing.current_process().name,
        "thread": threading.current_thread().name,
        "ts_us": monotonic_us(),
        "wall_ms": round(time.time() * 1000, 3),
    }
    for key, value in fields.items():
        if value is not None:
            payload[key] = _normalize(value)

    logger.info("[%s]%s", channel, json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")))


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, set):
        return [_normalize(item) for item in sorted(value, key=str)]
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    return str(value)
