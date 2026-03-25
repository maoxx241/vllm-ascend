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
import contextlib

import torch
import vllm
from torch.distributed import Backend
from vllm.distributed.parallel_state import GroupCoordinator, _get_unique_name, _register_group

from vllm_ascend.distributed.device_communicators.npu_communicator import NPUCommunicator
from vllm_ascend.utils import create_hccl_pg_options


def _describe_hccl_pg_options(hccl_pg_options) -> dict[str, object]:
    snapshot: dict[str, object] = {
        "type": hccl_pg_options.__class__.__name__,
    }
    with contextlib.suppress(Exception):
        snapshot["hccl_config"] = hccl_pg_options.hccl_config
    return snapshot


def _get_npu_backend(device_group):
    backend = None
    with contextlib.suppress(Exception):
        backend = device_group._get_backend(torch.device("npu"))
    if backend is None:
        with contextlib.suppress(Exception):
            backend = getattr(device_group, "_backend", None)
    return backend


def _get_hccl_comm_name(device_group, global_rank: int) -> str | None:
    backend = _get_npu_backend(device_group)
    if backend is None:
        return None
    getter = getattr(backend, "get_hccl_comm_name", None)
    if getter is None:
        return None
    with contextlib.suppress(TypeError):
        return str(getter(global_rank, True))
    with contextlib.suppress(TypeError):
        return str(getter(global_rank))
    with contextlib.suppress(Exception):
        return str(getter())
    return None


def _log_group_details(
    group_name: str,
    unique_name: str,
    ranks: list[int],
    global_rank: int,
    device_group,
    hccl_pg_options,
) -> None:
    backend = "unknown"
    with contextlib.suppress(Exception):
        backend = str(torch.distributed.get_backend(device_group))
    rank_in_group = None
    world_size = None
    with contextlib.suppress(Exception):
        rank_in_group = torch.distributed.get_rank(device_group)
    with contextlib.suppress(Exception):
        world_size = torch.distributed.get_world_size(device_group)
    hccl_pg_options_snapshot = _describe_hccl_pg_options(hccl_pg_options)
    comm_name = _get_hccl_comm_name(device_group, global_rank)
    print(
        f"HCCL_PG_OPTIONS: group_name={group_name}"
        f" unique_name={unique_name}"
        f" global_rank={global_rank}"
        f" ranks={ranks}"
        f" hccl_pg_options={hccl_pg_options_snapshot}",
        flush=True,
    )
    print(
        f"HCCL_COMM_NAME: group_name={group_name}"
        f" unique_name={unique_name}"
        f" backend={backend}"
        f" global_rank={global_rank}"
        f" rank_in_group={rank_in_group}"
        f" world_size={world_size}"
        f" type={device_group.__class__.__name__}"
        f" get_hccl_comm_name={comm_name}",
        flush=True,
    )


class GroupCoordinatorPatch(GroupCoordinator):
    def __init__(
        self,
        group_ranks: list[list[int]],
        local_rank: int,
        torch_distributed_backend: str | Backend,
        use_device_communicator: bool,  # whether to use device communicator
        use_message_queue_broadcaster: bool = False,
        group_name: str | None = None,
    ):
        group_name = group_name or "anonymous"
        self.unique_name = _get_unique_name(group_name)
        _register_group(self)

        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank

        self_device_group = None
        self_cpu_group = None
        hccl_pg_options = create_hccl_pg_options(group_name)

        for ranks in group_ranks:
            device_group = torch.distributed.new_group(
                ranks, backend=torch_distributed_backend, pg_options=hccl_pg_options
            )

            # a group with `gloo` backend, to allow direct coordination between
            # processes through the CPU.
            cpu_group = torch.distributed.new_group(ranks, backend="gloo")
            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self_device_group = device_group
                self_cpu_group = cpu_group

        assert self_cpu_group is not None
        assert self_device_group is not None

        self.cpu_group = self_cpu_group
        self.device_group = self_device_group
        self.device = torch.npu.current_device()

        self.use_device_communicator = use_device_communicator
        self.device_communicator = None
        if use_device_communicator and self.world_size > 1:
            self.device_communicator = NPUCommunicator(
                cpu_group=self.cpu_group,
                device=self.device,
                device_group=self.device_group,
                unique_name=self.unique_name,
            )

        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

        self.mq_broadcaster: MessageQueue | None = None
        if use_message_queue_broadcaster and self.world_size > 1:
            self.mq_broadcaster = MessageQueue.create_from_process_group(self.cpu_group, 1 << 22, 6)

        self.use_custom_op_call = True
        self.use_cpu_custom_send_recv = False
        _log_group_details(
            group_name=group_name,
            unique_name=self.unique_name,
            ranks=self.ranks,
            global_rank=self.rank,
            device_group=self.device_group,
            hccl_pg_options=hccl_pg_options,
        )

    def all_to_all(
        self,
        input_: torch.Tensor,
        scatter_dim: int = 0,
        gather_dim: int = -1,
        scatter_sizes: list[int] | None = None,
        gather_sizes: list[int] | None = None,
    ) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        assert -input_.dim() <= scatter_dim < input_.dim(), (
            f"Invalid scatter dim ({scatter_dim}) for input tensor with shape {input_.size()}"
        )
        assert -input_.dim() <= gather_dim < input_.dim(), (
            f"Invalid gather dim ({gather_dim}) for input tensor with shape {input_.size()}"
        )
        assert self.device_communicator is not None, "device_communicator should be initialized when world_size > 1"
        return self.device_communicator.all_to_all(input_, scatter_dim, gather_dim, scatter_sizes, gather_sizes)


vllm.distributed.parallel_state.GroupCoordinator = GroupCoordinatorPatch
