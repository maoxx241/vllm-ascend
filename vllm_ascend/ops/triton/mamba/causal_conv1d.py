# adapted from vllm/model_executor/layers/mamba/ops/causal_conv1d.py
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py
# and https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py
# mypy: ignore-errors

from collections import OrderedDict
from typing import Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from vllm.distributed import get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backends.utils import PAD_SLOT_ID  # type: ignore
from vllm_ascend.ops.triton.triton_utils import (
    get_vectorcore_num,
    init_device_properties_triton,
)


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    return_final_states: bool = False,
    final_states_out: torch.Tensor | None = None,
    activation: str | None = "silu",
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)
    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape

    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]

    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(dtype_in)  # (batch, dim, width - 1)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
    conv_states: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    cache_indices: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    metadata: Any | None = None,
    pad_slot_id: int = PAD_SLOT_ID,
):
    """
    x: (batch, dim, seqlen) or (dim,cu_seq_len) for varlen
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    bias: (dim,)
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
    conv_states: (...,dim,width - 1) itype
        updated inplace if provided
    activation: either None or "silu" or "swish"
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim, seqlen)
    """
    forward_context = get_forward_context()
    num_decodes = 0
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is not None and isinstance(attn_metadata, dict):
        attn_metadata = next(iter(attn_metadata.values()), None)
    if attn_metadata is not None:
        num_decodes = attn_metadata.num_decodes

    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    if x.stride(-1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None

    out_ref = []
    out_ref_b = []
    seqlens = query_start_loc[1:] - query_start_loc[:-1]
    seqlens = seqlens.tolist()
    splits = torch.split(x, seqlens, dim=-1)
    width = weight.shape[1]
    last_width_prefill_x = extract_last_width(x, query_start_loc[num_decodes:], conv_states.shape[-1])

    if get_pcp_group().world_size > 1:
        all_last_width_prefill_x = get_pcp_group().all_gather(last_width_prefill_x.unsqueeze(0).contiguous(), 0)
        pcp_rank = get_pcp_group().rank_in_group
        if pcp_rank > 0:
            conv_states[cache_indices[num_decodes:]] = all_last_width_prefill_x[pcp_rank - 1, ...]

    for i in range(len(seqlens)):
        x_s = splits[i]
        if cache_indices[i] == PAD_SLOT_ID:
            continue
        out_ref_b.append(
            causal_conv1d_ref(
                x_s,
                weight,
                bias,
                activation=activation,
                return_final_states=True,
                final_states_out=conv_states[cache_indices[i]][..., : (width - 1)].unsqueeze(0),
                initial_states=conv_states[cache_indices[i]][..., : (width - 1)],
            )
        )

    if get_pcp_group().world_size > 1:
        conv_states[cache_indices[num_decodes:]] = all_last_width_prefill_x[-1, ...]
    out_ref.append(torch.cat([t[0] for t in out_ref_b], dim=-1))
    out_ref_tensor = torch.cat(out_ref, dim=0)
    return out_ref_tensor


def extract_last_width(x, start_loc, width):
    end_loc = start_loc[1:]
    offsets = torch.arange(width, device=x.device)
    indices = end_loc.unsqueeze(1) - width + offsets.unsqueeze(0)  # (num_seqs, width)

    return x[:, indices].permute(1, 0, 2)


def _get_causal_conv1d_vectorcore_num() -> int:
    try:
        init_device_properties_triton()
        return get_vectorcore_num()
    except Exception:
        return 40


def _pick_causal_conv1d_update_launch_params(
    batch: int,
    dim: int,
    vectorcore_num: int | None = None,
    dtype: torch.dtype | None = None,
    width: int | None = None,
    seqlen: int | None = None,
    general_stride: bool = False,
) -> tuple[int, int, int]:
    if vectorcore_num is None:
        vectorcore_num = _get_causal_conv1d_vectorcore_num()

    # Keep total programs near ~2x vector cores while allowing larger dim
    # cases to reduce scheduling overhead with a wider channel tile.
    use_small_channel_tile = (
        (dtype == torch.float32 and ((width is not None and width >= 4) or (seqlen is not None and seqlen > 1)))
        or (general_stride and width is not None and width >= 4 and seqlen is not None and seqlen > 1)
    )
    if use_small_channel_tile:
        block_n = 256 if dim >= 256 else 128
    else:
        block_n = 512 if dim >= 512 else 256
    grid_c = triton.cdiv(dim, block_n)
    target_programs = max(2 * vectorcore_num, 1)
    b_tile_raw = max(1, triton.cdiv(batch * grid_c, target_programs))

    if b_tile_raw <= 1:
        b_tile = 1
    elif b_tile_raw <= 2:
        b_tile = 2
    elif b_tile_raw <= 4:
        b_tile = 4
    else:
        b_tile = 8

    if general_stride and width is not None and width >= 4 and seqlen is not None and seqlen > 1:
        b_tile = min(b_tile, 2)

    t_chunk = 1 if block_n == 512 else 48
    return block_n, b_tile, t_chunk


_SLA_FAST_PATH_WIDTH = 4
_SLA_FAST_PATH_SPEC_SEQLEN = 2
_SLA_FAST_PATH_DISPATCH_TABLE: dict[tuple[str, int, str, str], tuple[int, int, int]] = {
    ("decode_s1_bf16_w4", 40, "le64", "ge2048"): (512, 4, 1),
    ("spec_mtp1_s2_bf16_w4", 40, "le64", "ge2048"): (512, 4, 1),
    ("decode_s1_bf16_w4", 40, "le64", "ge4096"): (512, 8, 1),
    ("spec_mtp1_s2_bf16_w4", 40, "le64", "ge4096"): (512, 8, 1),
}


def _is_cache_line_stride_conv_state_layout(conv_state: torch.Tensor) -> bool:
    if conv_state.dim() != 3:
        return False
    stride_cache, stride_dim, stride_state = conv_state.stride()
    state_len = conv_state.shape[2]
    return stride_cache > 0 and stride_dim == state_len and stride_state == 1


def _select_causal_conv1d_update_sla_fast_path(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    conv_state_indices: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    query_start_loc: torch.Tensor | None,
    max_query_len: int,
    block_idx_last_scheduled_token: torch.Tensor | None,
    initial_state_idx: torch.Tensor | None,
) -> str | None:
    if x.device.type != "npu":
        return None
    if conv_state_indices is None:
        return None
    if block_idx_last_scheduled_token is not None or initial_state_idx is not None:
        return None
    if x.dtype != torch.bfloat16 or conv_state.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return None
    if bias is not None and bias.dtype != torch.bfloat16:
        return None
    if activation not in [None, "silu", "swish"]:
        return None
    if x.dim() != 2 or not x.is_contiguous():
        return None
    if weight.shape[1] != _SLA_FAST_PATH_WIDTH or not weight.is_contiguous():
        return None
    if not _is_cache_line_stride_conv_state_layout(conv_state):
        return None
    if conv_state.shape[1] != weight.shape[0]:
        return None
    if not all(
        _is_non_overlapping_positive_view(t)
        for t in (conv_state, conv_state_indices)
    ):
        return None
    if query_start_loc is not None and not _is_non_overlapping_positive_view(query_start_loc):
        return None
    if num_accepted_tokens is not None and not _is_non_overlapping_positive_view(num_accepted_tokens):
        return None

    if query_start_loc is None and num_accepted_tokens is None:
        if conv_state.shape[2] < _SLA_FAST_PATH_WIDTH - 1:
            return None
        return "decode_s1_bf16_w4"

    if (
        query_start_loc is not None
        and num_accepted_tokens is not None
        and max_query_len == _SLA_FAST_PATH_SPEC_SEQLEN
        and conv_state.shape[2] >= _SLA_FAST_PATH_WIDTH
        and _all_query_lengths_equal(query_start_loc, _SLA_FAST_PATH_SPEC_SEQLEN)
    ):
        return "spec_mtp1_s2_bf16_w4"
    return None


def _pick_causal_conv1d_update_sla_launch_params(
    path: str,
    batch: int,
    dim: int,
    vectorcore_num: int | None = None,
) -> tuple[int, int, int]:
    if vectorcore_num is None:
        vectorcore_num = _get_causal_conv1d_vectorcore_num()

    key = (path, vectorcore_num, _bucket_batch(batch), _bucket_dim(dim))
    config = _SLA_FAST_PATH_DISPATCH_TABLE.get(key)
    if config is not None:
        return config

    block_n = 512 if dim >= 512 else 256
    grid_c = triton.cdiv(dim, block_n)
    target_programs = max(2 * vectorcore_num, 1)
    b_tile_raw = max(1, triton.cdiv(batch * grid_c, target_programs))
    if b_tile_raw <= 1:
        b_tile = 1
    elif b_tile_raw <= 2:
        b_tile = 2
    elif b_tile_raw <= 4:
        b_tile = 4
    else:
        b_tile = 8
    t_chunk = 1 if block_n == 512 else 48
    return block_n, b_tile, t_chunk


_FAST_PATH_WIDTH = 4
_FAST_PATH_MTP_SEQLEN = 4
_WEIGHT_PREPACK_CACHE_MAXSIZE = 16
_WEIGHT_PREPACK_CACHE: "OrderedDict[tuple[Any, ...], torch.Tensor]" = OrderedDict()

_FAST_PATH_DISPATCH_TABLE: dict[tuple[str, int, str, str], tuple[int, int, int]] = {
    ("decode_contig_s1_bf16_w4", 20, "le64", "ge4096"): (512, 4, 0),
    ("decode_contig_s1_bf16_w4", 24, "le64", "ge4096"): (512, 4, 0),
    ("decode_contig_s1_bf16_w4", 40, "le64", "ge4096"): (256, 2, 0),
    ("decode_stride_s1_bf16_w4", 20, "le64", "ge4096"): (512, 2, 0),
    ("decode_stride_s1_bf16_w4", 24, "le64", "ge4096"): (512, 2, 0),
    ("decode_stride_s1_bf16_w4", 40, "le64", "ge4096"): (256, 1, 0),
    ("update_contig_s3_bf16_w4", 20, "le64", "ge4096"): (256, 2, 0),
    ("update_contig_s3_bf16_w4", 24, "le64", "ge4096"): (256, 2, 0),
    ("update_contig_s3_bf16_w4", 40, "le64", "ge4096"): (128, 2, 0),
    ("update_stride_s3_bf16_w4", 20, "le64", "ge4096"): (256, 1, 0),
    ("update_stride_s3_bf16_w4", 24, "le64", "ge4096"): (256, 1, 0),
    ("update_stride_s3_bf16_w4", 40, "le64", "ge4096"): (128, 1, 0),
    ("mtp_contig_k3_bf16_w4", 20, "le64", "ge4096"): (256, 1, 0),
    ("mtp_contig_k3_bf16_w4", 24, "le64", "ge4096"): (256, 1, 0),
    ("mtp_contig_k3_bf16_w4", 40, "le64", "ge4096"): (128, 1, 0),
    ("mtp_stride_k3_bf16_w4", 20, "le64", "ge4096"): (128, 1, 0),
    ("mtp_stride_k3_bf16_w4", 24, "le64", "ge4096"): (128, 1, 0),
    ("mtp_stride_k3_bf16_w4", 40, "le64", "ge4096"): (128, 1, 0),
}


def _make_weight_prepack_key(weight: torch.Tensor) -> tuple[Any, ...]:
    return (
        weight.device.type,
        weight.device.index,
        str(weight.dtype),
        weight.data_ptr(),
        weight.storage_offset(),
        tuple(weight.shape),
        tuple(weight.stride()),
        int(getattr(weight, "_version", 0)),
    )


def _get_weight_prepack_cache_size() -> int:
    return len(_WEIGHT_PREPACK_CACHE)


def _clear_weight_prepack_cache() -> None:
    _WEIGHT_PREPACK_CACHE.clear()


def _prepack_causal_conv1d_weight(weight: torch.Tensor) -> torch.Tensor:
    key = _make_weight_prepack_key(weight)
    packed = _WEIGHT_PREPACK_CACHE.get(key)
    if packed is not None:
        _WEIGHT_PREPACK_CACHE.move_to_end(key)
        return packed

    packed = weight.transpose(0, 1).contiguous()
    _WEIGHT_PREPACK_CACHE[key] = packed
    _WEIGHT_PREPACK_CACHE.move_to_end(key)
    while len(_WEIGHT_PREPACK_CACHE) > _WEIGHT_PREPACK_CACHE_MAXSIZE:
        _WEIGHT_PREPACK_CACHE.popitem(last=False)
    return packed


def _is_non_overlapping_positive_view(tensor: torch.Tensor) -> bool:
    if any(stride <= 0 for stride in tensor.stride()):
        return False
    overlap_check = getattr(torch, "_debug_has_internal_overlap", None)
    if overlap_check is None:
        return True
    overlap_state = int(overlap_check(tensor))
    return overlap_state != 1


def _is_fast_path_regular_layout(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    query_start_loc: torch.Tensor | None,
) -> bool:
    if query_start_loc is None:
        x_plain = x.is_contiguous() or x.stride(1) == 1
    else:
        x_plain = x.is_contiguous() or x.stride(1) == 1
    conv_state_plain = conv_state.is_contiguous() or conv_state.stride(1) == 1
    return x_plain and conv_state_plain


def _all_query_lengths_equal(
    query_start_loc: torch.Tensor,
    expected_len: int,
) -> bool:
    if query_start_loc.numel() < 2:
        return False
    return bool(
        torch.all((query_start_loc[1:] - query_start_loc[:-1]) == expected_len).item()
    )


def _select_causal_conv1d_update_fast_path(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    conv_state_indices: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    query_start_loc: torch.Tensor | None,
    max_query_len: int,
    block_idx_last_scheduled_token: torch.Tensor | None,
    initial_state_idx: torch.Tensor | None,
) -> str | None:
    if x.device.type != "npu":
        return None
    if conv_state_indices is None:
        return None
    if block_idx_last_scheduled_token is not None or initial_state_idx is not None:
        return None
    if x.dtype != torch.bfloat16 or conv_state.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return None
    if bias is not None and bias.dtype != torch.bfloat16:
        return None
    if activation not in [None, "silu", "swish"]:
        return None
    if weight.shape[1] != _FAST_PATH_WIDTH:
        return None
    if not all(
        _is_non_overlapping_positive_view(t)
        for t in (x, conv_state, weight, conv_state_indices)
    ):
        return None
    if query_start_loc is not None and not _is_non_overlapping_positive_view(query_start_loc):
        return None
    if num_accepted_tokens is not None and not _is_non_overlapping_positive_view(num_accepted_tokens):
        return None

    layout_kind = (
        "contig" if _is_fast_path_regular_layout(x, conv_state, query_start_loc) else "stride"
    )

    if query_start_loc is None and num_accepted_tokens is None:
        seqlen = 1 if x.dim() == 2 else x.shape[-1]
        if seqlen == 1:
            return f"decode_{layout_kind}_s1_bf16_w4"
        if seqlen == 3:
            return f"update_{layout_kind}_s3_bf16_w4"
        return None

    if (
        query_start_loc is not None
        and num_accepted_tokens is not None
        and x.dim() == 2
        and max_query_len == _FAST_PATH_MTP_SEQLEN
        and _all_query_lengths_equal(query_start_loc, _FAST_PATH_MTP_SEQLEN)
    ):
        return f"mtp_{layout_kind}_k3_bf16_w4"
    return None


def _bucket_batch(batch: int) -> str:
    if batch <= 32:
        return "le32"
    if batch <= 64:
        return "le64"
    return "ge128"


def _bucket_dim(dim: int) -> str:
    if dim >= 4096:
        return "ge4096"
    if dim >= 2048:
        return "ge2048"
    return "lt2048"


def _pick_causal_conv1d_update_fast_path_launch_params(
    path: str,
    batch: int,
    dim: int,
    vectorcore_num: int | None = None,
) -> tuple[int, int, int]:
    if vectorcore_num is None:
        vectorcore_num = _get_causal_conv1d_vectorcore_num()

    key = (path, vectorcore_num, _bucket_batch(batch), _bucket_dim(dim))
    config = _FAST_PATH_DISPATCH_TABLE.get(key)
    if config is not None:
        return config

    if path.startswith("decode_"):
        block_n = 512 if dim >= 512 else 256
        target_programs = max(2 * vectorcore_num, 1)
    elif path.startswith("update_"):
        block_n = 256 if dim >= 256 else 128
        target_programs = max(3 * vectorcore_num, 1)
    else:
        block_n = 256 if "contig" in path else 128
        target_programs = max(4 * vectorcore_num, 1)

    grid_c = triton.cdiv(dim, block_n)
    b_tile_raw = max(1, triton.cdiv(batch * grid_c, target_programs))
    if b_tile_raw <= 1:
        b_tile = 1
    elif b_tile_raw <= 2:
        b_tile = 2
    elif b_tile_raw <= 4:
        b_tile = 4
    else:
        b_tile = 8

    if "stride" in path:
        b_tile = min(b_tile, 2)
    return block_n, b_tile, 0


@triton.jit
def _causal_conv1d_update_kernel_npu_tiled(
    # Pointers
    x_ptr,  # (batch, dim, seqlen) OR (num_tokens, dim) for varlen
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,  # (num_cache_lines, dim, state_len)
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    query_start_loc_ptr,  # (batch + 1)
    block_idx_last_scheduled_token,  # (batch,)
    initial_state_idx,  # (batch,)
    o_ptr,  # same shape as x_ptr
    batch: tl.int32,
    dim: tl.constexpr,
    seqlen: tl.constexpr,  # max seqlen for varlen, or exact seqlen
    state_len: tl.constexpr,  # effective state_len computed in wrapper
    num_cache_lines: tl.constexpr,
    # Strides
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,  # <= 6
    SILU_ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    # tiling
    BLOCK_N: tl.constexpr,  # channel tile (C_TILE)
    B_TILE: tl.constexpr,  # batch tile
    T_CHUNK: tl.constexpr,  # token chunk for state update
):
    # program ids
    pid_b = tl.program_id(0)  # batch-tile id
    pid_c = tl.program_id(1)  # channel-tile id

    # channel indices for this program
    idx_feats = pid_c * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_w = idx_feats < dim

    # preload weights once per program (shared by B_TILE sequences)
    w_base = w_ptr + idx_feats * stride_w_dim
    # define to avoid "undefined" in branches
    w_col0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col3 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col4 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col5 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if KERNEL_WIDTH >= 1:
        w_col0 = tl.load(w_base + 0 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 2:
        w_col1 = tl.load(w_base + 1 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 3:
        w_col2 = tl.load(w_base + 2 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 4:
        w_col3 = tl.load(w_base + 3 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 5:
        w_col4 = tl.load(w_base + 4 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 6:
        w_col5 = tl.load(w_base + 5 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)

    # bias vector once per program
    if HAS_BIAS:
        acc_bias = tl.load(bias_ptr + idx_feats, mask=mask_w, other=0.0).to(tl.float32)
    else:
        acc_bias = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # process B_TILE sequences inside the same program instance
    for bi in tl.static_range(0, B_TILE):
        b = pid_b * B_TILE + bi  # scalar tl.int32
        lane_active = b < batch  # scalar predicate

        # -------------------------
        # APC mapping (optional)
        # -------------------------
        if IS_APC_ENABLED:
            conv_state_init = tl.load(initial_state_idx + b, mask=lane_active, other=0).to(tl.int32)
            current_last_index = tl.load(block_idx_last_scheduled_token + b, mask=lane_active, other=0).to(tl.int32)
        else:
            conv_state_init = tl.full((), 0, tl.int32)
            current_last_index = tl.full((), 0, tl.int32)

        # input cache line
        conv_states_input_coord = tl.load(
            conv_state_indices_ptr + b * stride_state_indices + conv_state_init, mask=lane_active, other=0
        ).to(tl.int64)

        if USE_PAD_SLOT:
            lane_active = lane_active & (conv_states_input_coord != pad_slot_id)

        # -------------------------
        # varlen (optional): revise seqlen_run and state_len_run like original kernel does
        # -------------------------
        if IS_VARLEN:
            qs = tl.load(query_start_loc_ptr + b, mask=lane_active, other=0).to(tl.int64)
            qe = tl.load(query_start_loc_ptr + (b + 1), mask=lane_active, other=0).to(tl.int64)
            seqlen_run = (qe - qs).to(tl.int32)
            # revise effective state_len for shorter sequences (same formula as original)
            state_len_run = (state_len - (seqlen - seqlen_run)).to(tl.int32)
            x_offset = (qs * stride_x_token).to(tl.int64)
            o_offset = (qs * stride_o_token).to(tl.int64)
        else:
            seqlen_run = tl.full((), seqlen, tl.int32)
            state_len_run = tl.full((), state_len, tl.int32)
            x_offset = (b * stride_x_seq).to(tl.int64)
            o_offset = (b * stride_o_seq).to(tl.int64)

        # empty sequence -> skip (avoid early return because other lanes in tile)
        lane_active = lane_active & (seqlen_run > 0)

        # -------------------------
        # spec decoding offset (optional)
        # -------------------------
        if IS_SPEC_DECODING:
            conv_state_token_offset = tl.load(num_accepted_tokens_ptr + b, mask=lane_active, other=1).to(tl.int64) - 1
            shift = tl.full((), 1, tl.int32)  # sliding by 1 in spec mode
        else:
            conv_state_token_offset = tl.full((), 0, tl.int64)
            shift = seqlen_run  # normal mode shift by seqlen

        # -------------------------
        # STEP 1: read initial history cols BEFORE state update (out==x safe)
        # -------------------------
        conv_states_base = (
            conv_state_ptr + conv_states_input_coord * stride_conv_state_seq + idx_feats * stride_conv_state_dim
        )
        prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok

        # define history vectors as zeros then load conditionally
        col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
        col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
        col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
        col3 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
        col4 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
        if KERNEL_WIDTH >= 2:
            col0 = tl.load(prior_tokens + 0 * stride_conv_state_tok, mask=lane_active & mask_w, other=0.0)
        if KERNEL_WIDTH >= 3:
            col1 = tl.load(prior_tokens + 1 * stride_conv_state_tok, mask=lane_active & mask_w, other=0.0)
        if KERNEL_WIDTH >= 4:
            col2 = tl.load(prior_tokens + 2 * stride_conv_state_tok, mask=lane_active & mask_w, other=0.0)
        if KERNEL_WIDTH >= 5:
            col3 = tl.load(prior_tokens + 3 * stride_conv_state_tok, mask=lane_active & mask_w, other=0.0)
        if KERNEL_WIDTH >= 6:
            col4 = tl.load(prior_tokens + 4 * stride_conv_state_tok, mask=lane_active & mask_w, other=0.0)

        # -------------------------
        # STEP 2: chunked state update (replaces original NP2_STATELEN x BLOCK_N big block)
        # Semantics: conv_state <- concat(old_state, x)[-state_len_run:].
        # - If seqlen_run >= state_len_run: dst[:] = x[seqlen_run - state_len_run : seqlen_run]
        # - Else: keep = state_len_run - seqlen_run,
        #         dst[0:keep] = src[shift : shift+keep], dst[keep:keep+seqlen_run] = x[0:seqlen_run]
        # -------------------------
        # output cache line
        conv_states_offset = tl.load(
            conv_state_indices_ptr + b * stride_state_indices + current_last_index, mask=lane_active, other=0
        ).to(tl.int64)

        use_shift = seqlen_run < state_len_run
        use_tail = seqlen_run >= state_len_run

        zero_i32 = tl.full((), 0, tl.int32)
        keep_shift = tl.where(use_shift, (state_len_run - seqlen_run), zero_i32).to(tl.int32)
        tail_start = tl.where(use_tail, (seqlen_run - state_len_run), zero_i32).to(tl.int32)

        # base pointers
        state_src_base = (
            conv_state_ptr
            + conv_states_input_coord * stride_conv_state_seq
            + conv_state_token_offset * stride_conv_state_tok
            + idx_feats * stride_conv_state_dim
        )
        state_dst_base = conv_state_ptr + conv_states_offset * stride_conv_state_seq + idx_feats * stride_conv_state_dim

        x_base = x_ptr + x_offset + idx_feats * stride_x_dim

        # Use 1D vector loads/stores here. Ascend Triton is more reliable with
        # strided state tensors when we avoid 2D irregular pointer matrices.
        # A) shift old state into dst[0:keep_shift)  (only when seqlen_run < state_len_run)
        for dst_tok in tl.static_range(0, NP2_STATELEN):
            src_tok = (dst_tok + shift).to(tl.int32)
            m = (
                lane_active
                & use_shift
                & (dst_tok < keep_shift)
                & (src_tok < state_len_run)
                & mask_w
                & (conv_states_input_coord < num_cache_lines)
                & (conv_states_offset < num_cache_lines)
            )

            vals = tl.load(
                state_src_base + src_tok * stride_conv_state_tok,
                mask=m,
                other=0.0,
            )
            tl.store(
                state_dst_base + dst_tok * stride_conv_state_tok,
                vals,
                mask=m,
            )

        # B) append x into dst[keep_shift : keep_shift+seqlen_run) (only when seqlen_run < state_len_run)
        for x_tok in tl.static_range(0, seqlen):
            dst_tok = (keep_shift + x_tok).to(tl.int32)
            m = (
                lane_active
                & use_shift
                & (x_tok < seqlen_run)
                & (dst_tok < state_len_run)
                & mask_w
                & (conv_states_offset < num_cache_lines)
            )

            x_vals = tl.load(
                x_base + x_tok * stride_x_token,
                mask=m,
                other=0.0,
            )
            tl.store(
                state_dst_base + dst_tok * stride_conv_state_tok,
                x_vals,
                mask=m,
            )

        # C) if seqlen_run >= state_len_run, overwrite dst with the tail of x
        for dst_tok in tl.static_range(0, NP2_STATELEN):
            x_tok = (tail_start + dst_tok).to(tl.int32)
            m = (
                lane_active
                & use_tail
                & (dst_tok < state_len_run)
                & (x_tok < seqlen_run)
                & mask_w
                & (conv_states_offset < num_cache_lines)
            )

            x_vals = tl.load(
                x_base + x_tok * stride_x_token,
                mask=m,
                other=0.0,
            )
            tl.store(
                state_dst_base + dst_tok * stride_conv_state_tok,
                x_vals,
                mask=m,
            )

        # -------------------------
        # STEP 3/4/5: causal conv1d (+ optional SiLU) and store output
        # This is original STEP3~5, but per-lane and without debug_barrier.
        # -------------------------
        x_base_1d = x_base
        o_base_1d = o_ptr + o_offset + idx_feats * stride_o_dim

        # accumulator preload (bias)
        acc_preload = acc_bias

        # compute each token; keep tl.range so varlen can use seqlen_run as runtime trip count (like original)
        for idx_token in tl.range(seqlen_run):
            acc = acc_preload

            # same selection logic as original (unrolled by KERNEL_WIDTH)
            matrix_w = w_col0
            matrix_x = col0
            for j in tl.static_range(KERNEL_WIDTH):
                if KERNEL_WIDTH == 1:
                    # only x[t] * w0
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=lane_active & mask_w, other=0.0)
                    matrix_w = w_col0
                elif KERNEL_WIDTH == 2:
                    if j == 1:
                        matrix_w = w_col1
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(x_ptrs_1d, mask=lane_active & mask_w, other=0.0)
                elif KERNEL_WIDTH == 3:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(x_ptrs_1d, mask=lane_active & mask_w, other=0.0)
                elif KERNEL_WIDTH == 4:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        matrix_x = col2
                    elif j == 3:
                        matrix_w = w_col3
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(x_ptrs_1d, mask=lane_active & mask_w, other=0.0)
                elif KERNEL_WIDTH == 5:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        matrix_x = col2
                    elif j == 3:
                        matrix_w = w_col3
                        matrix_x = col3
                    elif j == 4:
                        matrix_w = w_col4
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(x_ptrs_1d, mask=lane_active & mask_w, other=0.0)
                elif KERNEL_WIDTH == 6:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        matrix_x = col2
                    elif j == 3:
                        matrix_w = w_col3
                        matrix_x = col3
                    elif j == 4:
                        matrix_w = w_col4
                        matrix_x = col4
                    elif j == 5:
                        matrix_w = w_col5
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(x_ptrs_1d, mask=lane_active & mask_w, other=0.0)

                acc += matrix_x.to(tl.float32) * matrix_w  # [BLOCK_N]

            # roll history window
            if KERNEL_WIDTH == 2:
                col0 = matrix_x
            elif KERNEL_WIDTH == 3:
                col0 = col1
                col1 = matrix_x
            elif KERNEL_WIDTH == 4:
                col0 = col1
                col1 = col2
                col2 = matrix_x
            elif KERNEL_WIDTH == 5:
                col0 = col1
                col1 = col2
                col2 = col3
                col3 = matrix_x
            elif KERNEL_WIDTH == 6:
                col0 = col1
                col1 = col2
                col2 = col3
                col3 = col4
                col4 = matrix_x

            if SILU_ACTIVATION:
                acc = acc / (1.0 + tl.exp(-acc))

            # store output
            o_ptrs = o_base_1d + idx_token * stride_o_token
            tl.store(o_ptrs, acc, mask=lane_active & mask_w)


@triton.jit
def _causal_conv1d_update_kernel_npu_sla_tiled(
    x_ptr,
    w_ptr,
    bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    query_start_loc_ptr,
    o_ptr,
    batch: tl.int32,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_query_start_loc: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    B_TILE: tl.constexpr,
    T_CHUNK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    idx_feats = pid_c * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_w = idx_feats < dim

    w_base = w_ptr + idx_feats * stride_w_dim
    w_col0 = tl.load(w_base + 0 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    w_col1 = tl.load(w_base + 1 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    w_col2 = tl.load(w_base + 2 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    w_col3 = tl.load(w_base + 3 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)

    if HAS_BIAS:
        acc_bias = tl.load(bias_ptr + idx_feats, mask=mask_w, other=0.0).to(tl.float32)
    else:
        acc_bias = tl.zeros((BLOCK_N,), dtype=tl.float32)

    tok_vec = tl.arange(0, T_CHUNK)

    for bi in tl.static_range(0, B_TILE):
        b = pid_b * B_TILE + bi
        lane_active = b < batch

        state_index = tl.load(
            conv_state_indices_ptr + b * stride_state_indices,
            mask=lane_active,
            other=0,
        ).to(tl.int64)
        lane_active = lane_active & (state_index < num_cache_lines)
        if USE_PAD_SLOT:
            lane_active = lane_active & (state_index != pad_slot_id)
        state_index = tl.where(lane_active, state_index, 0)

        if IS_SPEC_DECODING:
            query_start = tl.load(
                query_start_loc_ptr + b * stride_query_start_loc,
                mask=lane_active,
                other=0,
            ).to(tl.int64)
            x_offset = (query_start * stride_x_token).to(tl.int64)
            o_offset = (query_start * stride_o_token).to(tl.int64)
            accepted_tokens = tl.load(
                num_accepted_tokens_ptr + b,
                mask=lane_active,
                other=1,
            ).to(tl.int32)
            conv_state_token_offset = (tl.minimum(
                tl.maximum(accepted_tokens, 1), seqlen
            ) - 1).to(tl.int64)
            shift = tl.full((), 1, tl.int32)
        else:
            x_offset = (b * stride_x_seq).to(tl.int64)
            o_offset = (b * stride_o_seq).to(tl.int64)
            conv_state_token_offset = tl.full((), 0, tl.int64)
            shift = tl.full((), seqlen, tl.int32)

        state_len_run = tl.full((), state_len, tl.int32)
        seqlen_run = tl.full((), seqlen, tl.int32)

        conv_states_base = conv_state_ptr + state_index * stride_conv_state_seq + idx_feats * stride_conv_state_dim
        prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok

        col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
        col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
        col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
        col0 = tl.load(prior_tokens + 0 * stride_conv_state_tok, mask=lane_active & mask_w, other=0.0)
        col1 = tl.load(prior_tokens + 1 * stride_conv_state_tok, mask=lane_active & mask_w, other=0.0)
        col2 = tl.load(prior_tokens + 2 * stride_conv_state_tok, mask=lane_active & mask_w, other=0.0)

        state_src_base = (
            conv_state_ptr
            + state_index * stride_conv_state_seq
            + conv_state_token_offset * stride_conv_state_tok
            + idx_feats * stride_conv_state_dim
        )
        state_dst_base = conv_state_ptr + state_index * stride_conv_state_seq + idx_feats * stride_conv_state_dim
        x_base = x_ptr + x_offset + idx_feats * stride_x_dim

        keep_shift = (state_len_run - seqlen_run).to(tl.int32)
        for t0 in tl.static_range(0, NP2_STATELEN, T_CHUNK):
            dst_tok = (t0 + tok_vec).to(tl.int32)
            src_tok = (dst_tok + shift).to(tl.int32)
            m_tok = (dst_tok < keep_shift) & (src_tok < state_len_run)
            m = (lane_active & m_tok)[:, None] & mask_w[None, :]
            src_ptrs = state_src_base[None, :] + src_tok[:, None] * stride_conv_state_tok
            dst_ptrs = state_dst_base[None, :] + dst_tok[:, None] * stride_conv_state_tok
            vals = tl.load(src_ptrs, mask=m, other=0.0)
            tl.store(dst_ptrs, vals, mask=m)

        for t0 in tl.static_range(0, seqlen, T_CHUNK):
            x_tok = (t0 + tok_vec).to(tl.int32)
            dst_tok = (keep_shift + x_tok).to(tl.int32)
            m_tok = (x_tok < seqlen_run) & (dst_tok < state_len_run)
            m = (lane_active & m_tok)[:, None] & mask_w[None, :]
            x_ptrs = x_base[None, :] + x_tok[:, None] * stride_x_token
            dst_ptrs = state_dst_base[None, :] + dst_tok[:, None] * stride_conv_state_tok
            x_vals = tl.load(x_ptrs, mask=m, other=0.0)
            tl.store(dst_ptrs, x_vals, mask=m)

        x_base_1d = x_base
        o_base_1d = o_ptr + o_offset + idx_feats * stride_o_dim
        acc_preload = acc_bias

        for idx_token in tl.static_range(0, seqlen):
            acc = acc_preload
            matrix_w = w_col0
            matrix_x = col0
            for j in tl.static_range(4):
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = tl.load(
                        x_base_1d + idx_token * stride_x_token,
                        mask=lane_active & mask_w,
                        other=0.0,
                    )
                acc += matrix_x.to(tl.float32) * matrix_w

            col0 = col1
            col1 = col2
            col2 = matrix_x

            if SILU_ACTIVATION:
                acc = acc / (1.0 + tl.exp(-acc))

            tl.store(
                o_base_1d + idx_token * stride_o_token,
                acc,
                mask=lane_active & mask_w,
            )


def _launch_causal_conv1d_update_sla_fast_path(
    path: str,
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    conv_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None,
    query_start_loc: torch.Tensor | None,
    pad_slot_id: int,
) -> None:
    weight_t = weight.transpose(0, 1)
    conv_state_t = conv_state.transpose(1, 2)
    batch = conv_state_indices.size(0)
    dim = weight.shape[0]
    num_cache_lines = conv_state_t.size(0)
    block_n, b_tile, t_chunk = _pick_causal_conv1d_update_sla_launch_params(path, batch, dim)

    def grid(meta):
        return (
            triton.cdiv(batch, meta["B_TILE"]),
            triton.cdiv(dim, meta["BLOCK_N"]),
        )

    if path == "decode_s1_bf16_w4":
        x_t = x.unsqueeze(1)
        out = x_t
        seqlen = 1
        state_len = weight.shape[1] - 1
        stride_x_seq, stride_x_token, stride_x_dim = x_t.stride()
        stride_o_seq, stride_o_token, stride_o_dim = out.stride()
        stride_query_start_loc = 0
    else:
        assert num_accepted_tokens is not None
        assert query_start_loc is not None
        out = x
        seqlen = _SLA_FAST_PATH_SPEC_SEQLEN
        state_len = weight.shape[1]
        stride_x_token, stride_x_dim = x.stride()
        stride_x_seq = 0
        stride_o_token, stride_o_dim = out.stride()
        stride_o_seq = 0
        stride_query_start_loc = query_start_loc.stride(0)
        x_t = x

    stride_w_width, stride_w_dim = weight_t.stride()
    stride_state_seq, stride_state_token, stride_state_dim = conv_state_t.stride()
    stride_state_indices = conv_state_indices.stride(0)
    np2_statelen = triton.next_power_of_2(state_len)

    _causal_conv1d_update_kernel_npu_sla_tiled[grid](
        x_t,
        weight_t,
        bias,
        conv_state_t,
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        out,
        batch,
        dim,
        seqlen,
        state_len,
        num_cache_lines,
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_state_seq,
        stride_state_dim,
        stride_state_token,
        stride_state_indices,
        stride_query_start_loc,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        pad_slot_id,
        HAS_BIAS=bias is not None,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_SPEC_DECODING=path == "spec_mtp1_s2_bf16_w4",
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_N=block_n,
        B_TILE=b_tile,
        T_CHUNK=t_chunk,
    )


@triton.jit
def _causal_conv1d_update_w4_small_bf16_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,
    o_ptr,
    batch: tl.int32,
    dim: tl.constexpr,
    num_cache_lines: tl.int32,
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_row: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    SEQLEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    B_TILE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    idx_feats = pid_c * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_w = idx_feats < dim

    w_row0 = tl.load(w_ptr + 0 * stride_w_row + idx_feats * stride_w_dim, mask=mask_w, other=0.0).to(tl.float32)
    w_row1 = tl.load(w_ptr + 1 * stride_w_row + idx_feats * stride_w_dim, mask=mask_w, other=0.0).to(tl.float32)
    w_row2 = tl.load(w_ptr + 2 * stride_w_row + idx_feats * stride_w_dim, mask=mask_w, other=0.0).to(tl.float32)
    w_row3 = tl.load(w_ptr + 3 * stride_w_row + idx_feats * stride_w_dim, mask=mask_w, other=0.0).to(tl.float32)

    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + idx_feats, mask=mask_w, other=0.0).to(tl.float32)
    else:
        bias_vals = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for bi in tl.static_range(0, B_TILE):
        b = pid_b * B_TILE + bi
        lane_active = b < batch
        state_index = tl.load(
            conv_state_indices_ptr + b * stride_state_indices,
            mask=lane_active,
            other=0,
        ).to(tl.int64)
        lane_active = lane_active & (state_index < num_cache_lines)
        if USE_PAD_SLOT:
            lane_active = lane_active & (state_index != pad_slot_id)

        x_base = x_ptr + b * stride_x_seq + idx_feats * stride_x_dim
        o_base = o_ptr + b * stride_o_seq + idx_feats * stride_o_dim
        state_base = (
            conv_state_ptr
            + state_index * stride_conv_state_seq
            + idx_feats * stride_conv_state_dim
        )

        hist0 = tl.load(
            state_base + 0 * stride_conv_state_tok,
            mask=lane_active & mask_w,
            other=0.0,
        ).to(tl.float32)
        hist1 = tl.load(
            state_base + 1 * stride_conv_state_tok,
            mask=lane_active & mask_w,
            other=0.0,
        ).to(tl.float32)
        hist2 = tl.load(
            state_base + 2 * stride_conv_state_tok,
            mask=lane_active & mask_w,
            other=0.0,
        ).to(tl.float32)

        x0 = tl.load(
            x_base + 0 * stride_x_token,
            mask=lane_active & mask_w,
            other=0.0,
        ).to(tl.float32)

        if SEQLEN == 1:
            out0 = bias_vals + hist0 * w_row0 + hist1 * w_row1 + hist2 * w_row2 + x0 * w_row3
            if SILU_ACTIVATION:
                out0 = out0 / (1.0 + tl.exp(-out0))
            tl.store(
                o_base + 0 * stride_o_token,
                out0.to(o_ptr.dtype.element_ty),
                mask=lane_active & mask_w,
            )
            tl.store(
                state_base + 0 * stride_conv_state_tok,
                hist1.to(conv_state_ptr.dtype.element_ty),
                mask=lane_active & mask_w,
            )
            tl.store(
                state_base + 1 * stride_conv_state_tok,
                hist2.to(conv_state_ptr.dtype.element_ty),
                mask=lane_active & mask_w,
            )
            tl.store(
                state_base + 2 * stride_conv_state_tok,
                x0.to(conv_state_ptr.dtype.element_ty),
                mask=lane_active & mask_w,
            )
        else:
            x1 = tl.load(
                x_base + 1 * stride_x_token,
                mask=lane_active & mask_w,
                other=0.0,
            ).to(tl.float32)
            x2 = tl.load(
                x_base + 2 * stride_x_token,
                mask=lane_active & mask_w,
                other=0.0,
            ).to(tl.float32)

            out0 = bias_vals + hist0 * w_row0 + hist1 * w_row1 + hist2 * w_row2 + x0 * w_row3
            out1 = bias_vals + hist1 * w_row0 + hist2 * w_row1 + x0 * w_row2 + x1 * w_row3
            out2 = bias_vals + hist2 * w_row0 + x0 * w_row1 + x1 * w_row2 + x2 * w_row3
            if SILU_ACTIVATION:
                out0 = out0 / (1.0 + tl.exp(-out0))
                out1 = out1 / (1.0 + tl.exp(-out1))
                out2 = out2 / (1.0 + tl.exp(-out2))
            tl.store(
                o_base + 0 * stride_o_token,
                out0.to(o_ptr.dtype.element_ty),
                mask=lane_active & mask_w,
            )
            tl.store(
                o_base + 1 * stride_o_token,
                out1.to(o_ptr.dtype.element_ty),
                mask=lane_active & mask_w,
            )
            tl.store(
                o_base + 2 * stride_o_token,
                out2.to(o_ptr.dtype.element_ty),
                mask=lane_active & mask_w,
            )
            tl.store(
                state_base + 0 * stride_conv_state_tok,
                x0.to(conv_state_ptr.dtype.element_ty),
                mask=lane_active & mask_w,
            )
            tl.store(
                state_base + 1 * stride_conv_state_tok,
                x1.to(conv_state_ptr.dtype.element_ty),
                mask=lane_active & mask_w,
            )
            tl.store(
                state_base + 2 * stride_conv_state_tok,
                x2.to(conv_state_ptr.dtype.element_ty),
                mask=lane_active & mask_w,
            )


@triton.jit
def _causal_conv1d_update_w4_mtp_bf16_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    query_start_loc_ptr,
    o_ptr,
    batch: tl.int32,
    dim: tl.constexpr,
    num_cache_lines: tl.int32,
    stride_x_token: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_w_row: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_query_start_loc: tl.constexpr,
    stride_o_token: tl.constexpr,
    stride_o_dim: tl.constexpr,
    pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    B_TILE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    idx_feats = pid_c * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_w = idx_feats < dim

    w_row0 = tl.load(w_ptr + 0 * stride_w_row + idx_feats * stride_w_dim, mask=mask_w, other=0.0).to(tl.float32)
    w_row1 = tl.load(w_ptr + 1 * stride_w_row + idx_feats * stride_w_dim, mask=mask_w, other=0.0).to(tl.float32)
    w_row2 = tl.load(w_ptr + 2 * stride_w_row + idx_feats * stride_w_dim, mask=mask_w, other=0.0).to(tl.float32)
    w_row3 = tl.load(w_ptr + 3 * stride_w_row + idx_feats * stride_w_dim, mask=mask_w, other=0.0).to(tl.float32)

    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + idx_feats, mask=mask_w, other=0.0).to(tl.float32)
    else:
        bias_vals = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for bi in tl.static_range(0, B_TILE):
        b = pid_b * B_TILE + bi
        lane_active = b < batch
        state_index = tl.load(
            conv_state_indices_ptr + b * stride_state_indices,
            mask=lane_active,
            other=0,
        ).to(tl.int64)
        lane_active = lane_active & (state_index < num_cache_lines)
        if USE_PAD_SLOT:
            lane_active = lane_active & (state_index != pad_slot_id)

        query_start = tl.load(
            query_start_loc_ptr + b * stride_query_start_loc,
            mask=lane_active,
            other=0,
        ).to(tl.int64)
        query_end = tl.load(
            query_start_loc_ptr + (b + 1) * stride_query_start_loc,
            mask=lane_active,
            other=0,
        ).to(tl.int64)
        lane_active = lane_active & ((query_end - query_start) == 4)

        accepted_tokens = tl.load(
            num_accepted_tokens_ptr + b,
            mask=lane_active,
            other=1,
        ).to(tl.int32)
        accepted_offset = tl.minimum(tl.maximum(accepted_tokens, 1), 4) - 1

        state_base = (
            conv_state_ptr
            + state_index * stride_conv_state_seq
            + idx_feats * stride_conv_state_dim
        )
        history_base = state_base + accepted_offset.to(tl.int64) * stride_conv_state_tok

        hist0 = tl.load(
            history_base + 0 * stride_conv_state_tok,
            mask=lane_active & mask_w,
            other=0.0,
        ).to(tl.float32)
        hist1 = tl.load(
            history_base + 1 * stride_conv_state_tok,
            mask=lane_active & mask_w,
            other=0.0,
        ).to(tl.float32)
        hist2 = tl.load(
            history_base + 2 * stride_conv_state_tok,
            mask=lane_active & mask_w,
            other=0.0,
        ).to(tl.float32)

        x_base = x_ptr + query_start * stride_x_token + idx_feats * stride_x_dim
        o_base = o_ptr + query_start * stride_o_token + idx_feats * stride_o_dim
        x0 = tl.load(x_base + 0 * stride_x_token, mask=lane_active & mask_w, other=0.0).to(tl.float32)
        x1 = tl.load(x_base + 1 * stride_x_token, mask=lane_active & mask_w, other=0.0).to(tl.float32)
        x2 = tl.load(x_base + 2 * stride_x_token, mask=lane_active & mask_w, other=0.0).to(tl.float32)
        x3 = tl.load(x_base + 3 * stride_x_token, mask=lane_active & mask_w, other=0.0).to(tl.float32)

        out0 = bias_vals + hist0 * w_row0 + hist1 * w_row1 + hist2 * w_row2 + x0 * w_row3
        out1 = bias_vals + hist1 * w_row0 + hist2 * w_row1 + x0 * w_row2 + x1 * w_row3
        out2 = bias_vals + hist2 * w_row0 + x0 * w_row1 + x1 * w_row2 + x2 * w_row3
        out3 = bias_vals + x0 * w_row0 + x1 * w_row1 + x2 * w_row2 + x3 * w_row3

        if SILU_ACTIVATION:
            out0 = out0 / (1.0 + tl.exp(-out0))
            out1 = out1 / (1.0 + tl.exp(-out1))
            out2 = out2 / (1.0 + tl.exp(-out2))
            out3 = out3 / (1.0 + tl.exp(-out3))

        tl.store(o_base + 0 * stride_o_token, out0.to(o_ptr.dtype.element_ty), mask=lane_active & mask_w)
        tl.store(o_base + 1 * stride_o_token, out1.to(o_ptr.dtype.element_ty), mask=lane_active & mask_w)
        tl.store(o_base + 2 * stride_o_token, out2.to(o_ptr.dtype.element_ty), mask=lane_active & mask_w)
        tl.store(o_base + 3 * stride_o_token, out3.to(o_ptr.dtype.element_ty), mask=lane_active & mask_w)

        new_hist0 = tl.load(
            history_base + 1 * stride_conv_state_tok,
            mask=lane_active & mask_w,
            other=0.0,
        )
        new_hist1 = tl.load(
            history_base + 2 * stride_conv_state_tok,
            mask=lane_active & mask_w,
            other=0.0,
        )
        tl.store(state_base + 0 * stride_conv_state_tok, new_hist0, mask=lane_active & mask_w)
        tl.store(state_base + 1 * stride_conv_state_tok, new_hist1, mask=lane_active & mask_w)
        tl.store(state_base + 2 * stride_conv_state_tok, x0.to(conv_state_ptr.dtype.element_ty), mask=lane_active & mask_w)
        tl.store(state_base + 3 * stride_conv_state_tok, x1.to(conv_state_ptr.dtype.element_ty), mask=lane_active & mask_w)
        tl.store(state_base + 4 * stride_conv_state_tok, x2.to(conv_state_ptr.dtype.element_ty), mask=lane_active & mask_w)
        tl.store(state_base + 5 * stride_conv_state_tok, x3.to(conv_state_ptr.dtype.element_ty), mask=lane_active & mask_w)


def _launch_causal_conv1d_update_fast_path(
    path: str,
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    conv_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None,
    query_start_loc: torch.Tensor | None,
    pad_slot_id: int,
) -> None:
    batch = conv_state_indices.size(0)
    dim = x.size(1) if x.dim() == 2 else x.size(1)
    num_cache_lines = conv_state.size(0)
    packed_weight = _prepack_causal_conv1d_weight(weight)
    block_n, b_tile, _ = _pick_causal_conv1d_update_fast_path_launch_params(path, batch, dim)

    def grid(meta):
        return (
            triton.cdiv(batch, meta["B_TILE"]),
            triton.cdiv(dim, meta["BLOCK_N"]),
        )

    if path.startswith("mtp_"):
        assert num_accepted_tokens is not None
        assert query_start_loc is not None
        stride_x_token, stride_x_dim = x.stride()
        stride_o_token, stride_o_dim = x.stride()
        stride_query_start_loc = query_start_loc.stride(0)
        stride_wp_row, stride_wp_dim = packed_weight.stride()
        stride_state_seq, stride_state_dim, stride_state_tok = conv_state.stride()
        stride_state_indices = conv_state_indices.stride(0)
        _causal_conv1d_update_w4_mtp_bf16_kernel[grid](
            x,
            packed_weight,
            bias,
            conv_state,
            conv_state_indices,
            num_accepted_tokens,
            query_start_loc,
            x,
            batch,
            dim,
            num_cache_lines,
            stride_x_token,
            stride_x_dim,
            stride_wp_row,
            stride_wp_dim,
            stride_state_seq,
            stride_state_dim,
            stride_state_tok,
            stride_state_indices,
            stride_query_start_loc,
            stride_o_token,
            stride_o_dim,
            pad_slot_id,
            HAS_BIAS=bias is not None,
            SILU_ACTIVATION=activation in ["silu", "swish"],
            USE_PAD_SLOT=pad_slot_id is not None,
            BLOCK_N=block_n,
            B_TILE=b_tile,
            multibuffer=False,
        )
        return

    if x.dim() == 2:
        x = x.unsqueeze(-1)
    stride_x_seq, stride_x_dim, stride_x_token = x.stride()
    stride_o_seq, stride_o_dim, stride_o_token = x.stride()
    stride_wp_row, stride_wp_dim = packed_weight.stride()
    stride_state_seq, stride_state_dim, stride_state_tok = conv_state.stride()
    stride_state_indices = conv_state_indices.stride(0)
    seqlen = 1 if path.startswith("decode_") else 3
    _causal_conv1d_update_w4_small_bf16_kernel[grid](
        x,
        packed_weight,
        bias,
        conv_state,
        conv_state_indices,
        x,
        batch,
        dim,
        num_cache_lines,
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_wp_row,
        stride_wp_dim,
        stride_state_seq,
        stride_state_dim,
        stride_state_tok,
        stride_state_indices,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        pad_slot_id,
        HAS_BIAS=bias is not None,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        USE_PAD_SLOT=pad_slot_id is not None,
        SEQLEN=seqlen,
        BLOCK_N=block_n,
        B_TILE=b_tile,
        multibuffer=False,
    )


def causal_conv1d_update_npu(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    validate_data=False,
):
    """
    x: Input tensor which can take the following shapes:

    - `[batch, dim]` - single token prediction
    - `[batch, dim, seqlen]` - single or multiple tokens prediction
    - `[num_tokens, dim]` - continuous batching, where num_tokens is
        the total tokens of all sequences in that batch

    conv_state: (..., dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    block_idx_last_scheduled_token: (batch,), dtype int32
        The pointer into conv_state_indices, where the last cache block to be filled is located.
    initial_state_idx: (batch,), dtype int32
        The pointer into conv_state_indices, where the cache block containing the initial state is located.
    num_accepted_tokens: (batch,), dtype int32
        If not None, it indicates the number of accepted tokens for each
        sequence in the batch.
        This is used in speculative decoding, where the conv_state is updated
        in a sliding window manner.
    query_start_loc: (batch + 1,) int32
        If not None, the inputs is given in a varlen fashion and this indicates
        the starting index of each sequence in the batch.
    max_query_len: int
        If query_start_loc is not None, this indicates the maximum query
        length in the batch.
    pad_slot_id: int
            if conv_state_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: conv_state_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen) or (num_tokens, dim), same shape as `x`
    """
    if validate_data:
        assert pad_slot_id is not None
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    original_x_dtype = x.dtype
    x = x.to(conv_state.dtype)

    sla_fast_path = _select_causal_conv1d_update_sla_fast_path(
        x,
        conv_state,
        weight,
        bias,
        activation,
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        max_query_len,
        block_idx_last_scheduled_token,
        initial_state_idx,
    )
    if sla_fast_path is not None:
        assert conv_state_indices is not None
        _launch_causal_conv1d_update_sla_fast_path(
            sla_fast_path,
            x,
            conv_state,
            weight,
            bias,
            activation,
            conv_state_indices,
            num_accepted_tokens,
            query_start_loc,
            pad_slot_id,
        )
        return x.to(original_x_dtype)

    unsqueeze = query_start_loc is None and x.dim() == 2
    if unsqueeze:
        # make it (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(-1)

    if query_start_loc is None:
        batch, dim, seqlen = x.shape
    else:
        assert conv_state_indices is not None
        batch = conv_state_indices.size(0)
        dim = x.size(1)
        seqlen = max_query_len

    _, width = weight.shape
    num_cache_lines, _, state_len_total = conv_state.size()

    if validate_data:
        assert dim == weight.size(0)
        assert state_len_total >= width - 1
        assert dim == conv_state.size(1)
        if conv_state_indices is None:
            assert conv_state.size(0) >= batch
        else:
            assert (batch,) == conv_state_indices.shape
        assert num_cache_lines >= batch

    # overwrite-on-x strategy same as original
    out = x

    fast_path = _select_causal_conv1d_update_fast_path(
        x,
        conv_state,
        weight,
        bias,
        activation,
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        max_query_len,
        block_idx_last_scheduled_token,
        initial_state_idx,
    )
    if fast_path is not None:
        assert conv_state_indices is not None
        _launch_causal_conv1d_update_fast_path(
            fast_path,
            x,
            conv_state,
            weight,
            bias,
            activation,
            conv_state_indices,
            num_accepted_tokens,
            query_start_loc,
            pad_slot_id,
        )
        if unsqueeze:
            out = out.squeeze(-1)
        return out.to(original_x_dtype)

    stride_w_dim, stride_w_width = weight.stride()
    if query_start_loc is None:
        stride_x_seq, stride_x_dim, stride_x_token = x.stride()
        stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    else:
        stride_x_token, stride_x_dim = x.stride()
        stride_x_seq = 0
        stride_o_token, stride_o_dim = out.stride()
        stride_o_seq = 0

    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = conv_state_indices.stride(0) if conv_state_indices is not None else 0
    general_stride = (
        stride_w_width != 1
        or stride_x_dim != 1
        or stride_o_dim != 1
        or stride_istate_dim != 1
    )

    # effective state_len exactly as original
    if num_accepted_tokens is not None:
        eff_state_len = width - 1 + (seqlen - 1)
    else:
        eff_state_len = width - 1
    np2_statelen = triton.next_power_of_2(eff_state_len)

    block_n, b_tile, t_chunk = _pick_causal_conv1d_update_launch_params(
        batch,
        dim,
        dtype=conv_state.dtype,
        width=width,
        seqlen=seqlen,
        general_stride=general_stride,
    )

    def grid(META):
        return (
            triton.cdiv(batch, META["B_TILE"]),
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    _causal_conv1d_update_kernel_npu_tiled[grid](
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        block_idx_last_scheduled_token,
        initial_state_idx,
        out,
        batch,
        dim,
        seqlen,
        eff_state_len,
        num_cache_lines,
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        pad_slot_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_VARLEN=query_start_loc is not None,
        IS_APC_ENABLED=block_idx_last_scheduled_token is not None,
        IS_SPEC_DECODING=num_accepted_tokens is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_N=block_n,
        B_TILE=b_tile,
        T_CHUNK=t_chunk,
    )

    if unsqueeze:
        out = out.squeeze(-1)
    return out.to(original_x_dtype)
