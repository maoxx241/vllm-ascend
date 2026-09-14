# SPDX-License-Identifier: Apache-2.0
"""Raw-bit O/LSE packing for the measured K3 DCP8 decode shapes.

Adapted from PR #16350's bit-preserving exchange. Communication and deferred
combine stay in sfa_cp; no sparse-index or query-layout behavior is changed.
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


@triton.jit
def _pack(
    o,
    lse,
    send,
    os0: tl.constexpr,
    os1: tl.constexpr,
    ls0: tl.constexpr,
    ls1: tl.constexpr,
    T: tl.constexpr,
    CORES: tl.constexpr,
):
    words = tl.arange(0, 256)
    for row in range(tl.program_id(0), T * 96, CORES):
        token = row // 96
        head = row % 96
        peer = head // 12
        local_head = head % 12
        base = ((peer * 12 + local_head) * T + token) * 257
        value = tl.load(o + token * os0 + head * os1 + words)
        tl.store(send + base + words, value)
        stat = tl.load(lse + token * ls0 + head * ls1)
        tl.store(send + base + 256, stat.to(tl.int32, bitcast=True))


def can_use_raw_dcp_exchange(
    output: torch.Tensor,
    lse: torch.Tensor,
    scatter_size: int,
    scatter_dim: int,
    *,
    has_pcp: bool = False,
    return_lse: bool = False,
) -> bool:
    """Shape/dtype-only dispatch: ranks must select the same wire protocol."""
    return (
        not has_pcp
        and not return_lse
        and scatter_size == 8
        and scatter_dim == 1
        and output.device.type == "npu"
        and output.dtype == torch.bfloat16
        and output.ndim == 3
        and output.shape[0] in (4, 8, 16)
        and output.shape[1:] == (96, 512)
        and lse.device == output.device
        and lse.dtype == torch.float32
        and lse.shape == (*output.shape[:2], 1)
    )


def pack_raw_dcp_output_lse(output: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    """Pack 256 BF16 pairs and one FP32 LSE into each INT32 receive row."""
    if not can_use_raw_dcp_exchange(output, lse, 8, 1):
        raise ValueError("Raw DCP packing requires BF16 T4/8/16 H96 D512 and FP32 LSE.")
    # Rank-local strides only affect normalization, never collective selection.
    if output.stride(-1) != 1 or any(stride % 2 for stride in output.stride()[:2]):
        output = output.contiguous()
    if output.storage_offset() % 2:
        output = output.clone()
    words = output.view(torch.int32)
    tokens = output.shape[0]
    send = torch.empty((8, 12, tokens, 257), dtype=torch.int32, device=output.device)
    init_device_properties_triton()
    programs = min(tokens * 96, get_vectorcore_num())
    _pack[(programs,)](words, lse, send, *words.stride()[:2], *lse.stride()[:2], tokens, programs)
    return send
