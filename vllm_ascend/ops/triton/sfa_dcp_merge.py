# SPDX-License-Identifier: Apache-2.0
"""Merge raw DCP8 history and an optional current chunk in one FP32 pass."""

import torch
from triton.language.extra.cann import extension
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

get_element = extension.get_element


@triton.jit
def _merge(
    recv,
    out_bits,
    out,
    local_o,
    local_lse,
    lo0: tl.constexpr,
    lo1: tl.constexpr,
    lo2: tl.constexpr,
    ll0: tl.constexpr,
    ll1: tl.constexpr,
    T: tl.constexpr,
    CORES: tl.constexpr,
    HAS_LOCAL: tl.constexpr,
):
    rank_ids = tl.arange(0, 8)
    dims = tl.arange(0, 512)
    for row in range(tl.program_id(0), T * 12, CORES):
        token = row // 12
        head = row % 12
        base = (rank_ids * 12 * T + head * T + token) * 257
        raw = tl.load(recv + base + 256).to(tl.float32, bitcast=True)
        finite = (raw == raw) & (raw > -float("inf")) & (raw < float("inf"))
        maximum = tl.max(tl.where(finite, raw, -float("inf")), 0)
        if HAS_LOCAL:
            local_stat = tl.load(local_lse + token * ll0 + head * ll1).to(tl.float32)
            local_valid = (local_stat == local_stat) & (local_stat > -float("inf")) & (local_stat < float("inf"))
            maximum = tl.maximum(maximum, tl.where(local_valid, local_stat, -float("inf")))
        maximum = tl.where(maximum > -float("inf"), maximum, 0.0)
        weights = tl.where(finite, tl.exp(raw - maximum), 0.0)
        denom = 0.0
        if HAS_LOCAL:
            local_weight = tl.where(local_valid, tl.exp(local_stat - maximum), 0.0)
        result = tl.zeros((512,), tl.float32)
        # Read O through a BF16 alias of the same raw buffer; no receive copy.
        for rank in tl.static_range(8):
            offset = ((rank * 12 + head) * T + token) * 514
            value = tl.load(out_bits + offset + dims).to(tl.float32)
            value = tl.where(get_element(finite, (rank,)), value, 0.0)
            weight = get_element(weights, (rank,))
            result += value * weight
            denom += weight
        if HAS_LOCAL:
            value = tl.load(local_o + token * lo0 + head * lo1 + dims * lo2).to(tl.float32)
            result += tl.where(local_valid, value, 0.0) * local_weight
            denom += local_weight
        result /= tl.where(denom > 0.0, denom, 1.0)
        tl.store(out + row * 512 + dims, result)


def merge_raw_dcp_output_lse(
    recv: torch.Tensor,
    head_dim: int,
    scatter_dim: int,
    local_output: torch.Tensor | None = None,
    local_lse: torch.Tensor | None = None,
) -> torch.Tensor:
    """Consume [rank, local head, token, 257] INT32 without a receive copy."""
    if (
        recv.device.type != "npu"
        or recv.dtype != torch.int32
        or recv.ndim != 4
        or recv.shape[0:2] != (8, 12)
        or recv.shape[2] not in (4, 8, 16)
        or recv.shape[3] != 257
        or not recv.is_contiguous()
        or head_dim != 512
        or scatter_dim != 1
    ):
        raise ValueError("Invalid raw DCP8 receive buffer or head-scatter geometry.")
    tokens = recv.shape[2]
    if (local_output is None) != (local_lse is None):
        raise ValueError("Local output and LSE must be supplied together.")
    if local_output is not None and (
        local_output.shape != (tokens, 12, 512)
        or local_lse.shape != (tokens, 12, 1)
        or local_output.device != recv.device
        or local_lse.device != recv.device
        or local_output.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or local_lse.dtype != torch.float32
    ):
        raise ValueError("Local contribution must match raw DCP tokens/heads with FP32 LSE.")
    out = torch.empty((tokens, 12, 512), dtype=torch.bfloat16, device=recv.device)
    init_device_properties_triton()
    programs = min(tokens * 12, get_vectorcore_num())
    _merge[(programs,)](
        recv,
        recv.view(torch.bfloat16),
        out,
        local_output if local_output is not None else recv,
        local_lse if local_lse is not None else recv,
        *(local_output.stride() if local_output is not None else (0, 0, 0)),
        *(local_lse.stride()[:2] if local_lse is not None else (0, 0)),
        tokens,
        programs,
        local_output is not None,
    )
    return out
