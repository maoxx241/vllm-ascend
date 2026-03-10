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

import threading
import time
from contextlib import contextmanager

from vllm.distributed.device_communicators import shm_broadcast
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

from vllm_ascend.diagnostics import emit_diag, env_flag, env_int

_SHM_BROADCAST_DIAG_ENABLED = env_flag("VLLM_ASCEND_SHM_BROADCAST_DIAG")
_SHM_BROADCAST_DIAG_THRESHOLD_MS = env_int("VLLM_ASCEND_SHM_BROADCAST_DIAG_THRESHOLD_MS", 50)


def _emit_wait_event(
    mq: MessageQueue,
    *,
    op: str,
    line: int,
    wait_s: float,
    wait_loops: int,
    outcome: str,
    timeout: float | None,
    read_count: int | None = None,
    indefinite: bool | None = None,
) -> None:
    wait_ms = wait_s * 1000
    if wait_ms < _SHM_BROADCAST_DIAG_THRESHOLD_MS:
        return

    emit_diag(
        "shm_broadcast_diag",
        "message_queue_wait",
        _SHM_BROADCAST_DIAG_ENABLED,
        op=op,
        line=line,
        wait_ms=wait_ms,
        wait_loops=wait_loops,
        outcome=outcome,
        timeout_ms=None if timeout is None else timeout * 1000,
        indefinite=indefinite,
        current_idx=mq.current_idx,
        n_reader=mq.buffer.n_reader,
        n_local_reader=mq.n_local_reader,
        n_remote_reader=mq.n_remote_reader,
        writer_rank=getattr(mq, "writer_rank", None),
        local_reader_rank=getattr(mq, "local_reader_rank", None),
        process_thread=f"{threading.current_thread().name}",
        blocked_readers=(
            None
            if read_count is None
            else max(mq.buffer.n_reader - read_count, 0)
        ),
        observed_read_count=read_count,
    )


if _SHM_BROADCAST_DIAG_ENABLED and not getattr(MessageQueue, "_ascend_shm_diag_patched", False):
    MessageQueue._ascend_original_acquire_write = MessageQueue.acquire_write
    MessageQueue._ascend_original_acquire_read = MessageQueue.acquire_read

    @contextmanager
    def acquire_write(self: MessageQueue, timeout: float | None = None):
        assert self._is_writer, "Only writers can acquire write"
        start_time = time.monotonic()
        n_warning = 1
        wait_loops = 0
        last_read_count: int | None = None

        while True:
            with self.buffer.get_metadata(self.current_idx) as metadata_buffer:
                shm_broadcast.memory_fence()
                read_count = sum(metadata_buffer[1:])
                written_flag = metadata_buffer[0]
                if written_flag and read_count != self.buffer.n_reader:
                    wait_loops += 1
                    last_read_count = read_count
                    shm_broadcast.sched_yield()

                    elapsed = time.monotonic() - start_time
                    if timeout is not None and elapsed > timeout:
                        _emit_wait_event(
                            self,
                            op="acquire_write",
                            line=458,
                            wait_s=elapsed,
                            wait_loops=wait_loops,
                            outcome="timeout",
                            timeout=timeout,
                            read_count=last_read_count,
                        )
                        raise TimeoutError

                    if elapsed > shm_broadcast.VLLM_RINGBUFFER_WARNING_INTERVAL * n_warning:
                        shm_broadcast.logger.info(
                            shm_broadcast.long_wait_time_msg(shm_broadcast.VLLM_RINGBUFFER_WARNING_INTERVAL)
                        )
                        n_warning += 1
                    continue

                elapsed = time.monotonic() - start_time
                _emit_wait_event(
                    self,
                    op="acquire_write",
                    line=458,
                    wait_s=elapsed,
                    wait_loops=wait_loops,
                    outcome="acquired",
                    timeout=timeout,
                    read_count=last_read_count,
                )

                metadata_buffer[0] = 0
                with self.buffer.get_data(self.current_idx) as buf:
                    yield buf

                for i in range(1, self.buffer.n_reader + 1):
                    metadata_buffer[i] = 0
                metadata_buffer[0] = 1
                shm_broadcast.memory_fence()
                self.current_idx = (self.current_idx + 1) % self.buffer.max_chunks
                break

    @contextmanager
    def acquire_read(
        self: MessageQueue,
        timeout: float | None = None,
        cancel: threading.Event | None = None,
        indefinite: bool = False,
    ):
        assert self._is_local_reader, "Only readers can acquire read"
        start_time = time.monotonic()
        n_warning = 1
        wait_loops = 0

        while True:
            with self.buffer.get_metadata(self.current_idx) as metadata_buffer:
                shm_broadcast.memory_fence()
                read_flag = metadata_buffer[self.local_reader_rank + 1]
                written_flag = metadata_buffer[0]
                if not written_flag or read_flag:
                    wait_loops += 1
                    self._read_spin_timer.spin()

                    if cancel is not None and cancel.is_set():
                        elapsed = time.monotonic() - start_time
                        _emit_wait_event(
                            self,
                            op="acquire_read",
                            line=528,
                            wait_s=elapsed,
                            wait_loops=wait_loops,
                            outcome="cancelled",
                            timeout=timeout,
                            indefinite=indefinite,
                            read_count=None,
                        )
                        raise RuntimeError("cancelled")

                    elapsed = time.monotonic() - start_time
                    if timeout is not None and elapsed > timeout:
                        _emit_wait_event(
                            self,
                            op="acquire_read",
                            line=528,
                            wait_s=elapsed,
                            wait_loops=wait_loops,
                            outcome="timeout",
                            timeout=timeout,
                            indefinite=indefinite,
                            read_count=None,
                        )
                        raise TimeoutError

                    if not indefinite and elapsed > shm_broadcast.VLLM_RINGBUFFER_WARNING_INTERVAL * n_warning:
                        shm_broadcast.logger.info(
                            shm_broadcast.long_wait_time_msg(shm_broadcast.VLLM_RINGBUFFER_WARNING_INTERVAL)
                        )
                        n_warning += 1
                    continue

                elapsed = time.monotonic() - start_time
                _emit_wait_event(
                    self,
                    op="acquire_read",
                    line=528,
                    wait_s=elapsed,
                    wait_loops=wait_loops,
                    outcome="acquired",
                    timeout=timeout,
                    indefinite=indefinite,
                    read_count=None,
                )

                with self.buffer.get_data(self.current_idx) as buf:
                    yield buf

                metadata_buffer[self.local_reader_rank + 1] = 1
                shm_broadcast.memory_fence()
                self.current_idx = (self.current_idx + 1) % self.buffer.max_chunks
                self._read_spin_timer.record_activity()
                break

    MessageQueue.acquire_write = acquire_write
    MessageQueue.acquire_read = acquire_read
    MessageQueue._ascend_shm_diag_patched = True
