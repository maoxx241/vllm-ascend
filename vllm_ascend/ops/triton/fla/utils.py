# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
import contextlib
import functools
from dataclasses import dataclass
from collections.abc import Callable

import torch
from vllm.triton_utils import tl, triton

GDN_CHUNK_SIZE = 64
GDN_SOLVE_TRIL_LARGE_BLOCK_T = 608 * 2


@dataclass(slots=True)
class GDNPrefillIndexPlan:
    chunk_size: int
    total_chunks: int
    block_indices: dict[int, torch.Tensor]
    chunk_offsets: torch.Tensor
    update_chunk_offsets: torch.Tensor
    final_chunk_indices: torch.Tensor

    def get_num_blocks(self, block_size: int) -> int:
        return int(self.block_indices[block_size].shape[0])

    def get_block_indices(self, block_size: int) -> torch.Tensor:
        return self.block_indices[block_size]

    def get_chunk_indices(self) -> torch.Tensor:
        return self.block_indices[self.chunk_size]

    def get_chunk_offsets(self) -> torch.Tensor:
        return self.chunk_offsets

    def get_update_chunk_offsets(self) -> torch.Tensor:
        return self.update_chunk_offsets

    def get_final_chunk_indices(self) -> torch.Tensor:
        return self.final_chunk_indices


def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_chunk_counts(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    return triton.cdiv(prepare_lens(cu_seqlens), chunk_size)


def prepare_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in prepare_chunk_counts(cu_seqlens, chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def prepare_final_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    indices = prepare_chunk_counts(cu_seqlens, chunk_size) + 1
    return torch.cumsum(indices, 0) - 1


def prepare_chunk_offsets(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    return torch.cat([cu_seqlens.new_tensor([0]), prepare_chunk_counts(cu_seqlens, chunk_size)]).cumsum(-1)


def prepare_update_chunk_offsets(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    return torch.cat([cu_seqlens.new_tensor([0]), prepare_chunk_counts(cu_seqlens, chunk_size) + 1]).cumsum(-1)


def _maybe_pin_and_move(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if device.type == "cpu":
        return tensor
    return tensor.pin_memory().to(device=device, non_blocking=True)


def get_chunk_local_cumsum_block_size(num_heads: int, chunk_size: int = GDN_CHUNK_SIZE) -> int:
    return max(
        chunk_size,
        triton.next_power_of_2(max(1, (2**18) // max(1, num_heads * chunk_size))),
    )


def prepare_gdn_prefill_index_plan(
    cu_seqlens: torch.LongTensor,
    required_block_sizes: tuple[int, ...],
    chunk_size: int = GDN_CHUNK_SIZE,
) -> GDNPrefillIndexPlan:
    device = cu_seqlens.device
    cu_seqlens_cpu = cu_seqlens.cpu() if cu_seqlens.device.type != "cpu" else cu_seqlens
    block_sizes = tuple(sorted({chunk_size, *required_block_sizes}))
    cpu_block_indices = {
        block_size: prepare_chunk_indices(cu_seqlens_cpu, block_size)
        for block_size in block_sizes
    }
    return GDNPrefillIndexPlan(
        chunk_size=chunk_size,
        total_chunks=int(cpu_block_indices[chunk_size].shape[0]),
        block_indices={
            block_size: _maybe_pin_and_move(indices, device)
            for block_size, indices in cpu_block_indices.items()
        },
        chunk_offsets=_maybe_pin_and_move(prepare_chunk_offsets(cu_seqlens_cpu, chunk_size), device),
        update_chunk_offsets=_maybe_pin_and_move(prepare_update_chunk_offsets(cu_seqlens_cpu, chunk_size), device),
        final_chunk_indices=_maybe_pin_and_move(prepare_final_chunk_indices(cu_seqlens_cpu, chunk_size), device),
    )


def input_guard(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args)
        contiguous_kwargs = {k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()}

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None:
            ctx = torch.npu.device(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


@triton.jit
def safe_exp(x):
    return tl.exp(tl.where(x <= 0, x, float("-inf")))
