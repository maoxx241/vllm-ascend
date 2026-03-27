# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _build_chunk_counts_kernel(
    cu_seqlens_ptr,
    chunk_counts_ptr,
    update_chunk_counts_ptr,
    num_seqs,
    chunk_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_seqs

    bos = tl.load(cu_seqlens_ptr + offsets, mask=mask, other=0).to(tl.int32)
    eos = tl.load(cu_seqlens_ptr + offsets + 1, mask=mask, other=0).to(tl.int32)
    seq_lens = eos - bos
    chunk_counts = (seq_lens + chunk_size - 1) // chunk_size

    tl.store(chunk_counts_ptr + offsets, chunk_counts, mask=mask)
    tl.store(update_chunk_counts_ptr + offsets, chunk_counts + 1, mask=mask)


@triton.jit
def _build_chunk_indices_kernel(
    chunk_offsets_ptr,
    out_chunk_indices_ptr,
    total_chunks,
    num_seqs,
    SEARCH_ITERS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = rows < total_chunks

    left = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    right = tl.full([BLOCK_SIZE], num_seqs, dtype=tl.int32)
    seq_indices = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    for _ in range(SEARCH_ITERS):
        mid = (left + right) // 2
        chunk_start = tl.load(chunk_offsets_ptr + mid, mask=mask, other=0).to(tl.int32)
        chunk_end = tl.load(chunk_offsets_ptr + mid + 1, mask=mask, other=0).to(tl.int32)

        go_left = rows < chunk_start
        go_right = rows >= chunk_end
        found = ~(go_left | go_right)

        right = tl.where(go_left, mid, right)
        left = tl.where(go_right, mid + 1, left)
        seq_indices = tl.where(found, mid, seq_indices)

    seq_chunk_start = tl.load(chunk_offsets_ptr + seq_indices, mask=mask, other=0).to(tl.int32)
    flat_offsets = rows * 2

    tl.store(
        out_chunk_indices_ptr + flat_offsets,
        seq_indices.to(out_chunk_indices_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        out_chunk_indices_ptr + flat_offsets + 1,
        (rows - seq_chunk_start).to(out_chunk_indices_ptr.dtype.element_ty),
        mask=mask,
    )


def _validate_optional_output(
    name: str,
    tensor: torch.Tensor | None,
    *,
    expected_shape: tuple[int, ...] | None,
    expected_device: torch.device,
    expected_dtype: torch.dtype,
) -> None:
    if tensor is None:
        return
    if tensor.device != expected_device:
        raise ValueError(f"{name} must be on device {expected_device}, got {tensor.device}")
    if tensor.dtype != expected_dtype:
        raise ValueError(f"{name} must have dtype {expected_dtype}, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if expected_shape is not None and tuple(tensor.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}")


def build_chunk_meta_device(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    out_chunk_indices: torch.Tensor | None = None,
    out_chunk_offsets: torch.Tensor | None = None,
    out_update_chunk_offsets: torch.Tensor | None = None,
    out_final_chunk_indices: torch.Tensor | None = None,
) -> None:
    if not isinstance(cu_seqlens, torch.Tensor):
        raise TypeError("cu_seqlens must be a torch.Tensor")
    if cu_seqlens.device.type != "npu":
        raise ValueError(f"cu_seqlens must be on NPU, got {cu_seqlens.device}")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"cu_seqlens must have int32 or int64 dtype, got {cu_seqlens.dtype}")
    if cu_seqlens.ndim != 1:
        raise ValueError(f"cu_seqlens must be 1D, got shape {tuple(cu_seqlens.shape)}")
    if cu_seqlens.shape[0] < 1:
        raise ValueError("cu_seqlens must contain at least one element")
    if not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be contiguous")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    if (
        out_chunk_indices is None
        and out_chunk_offsets is None
        and out_update_chunk_offsets is None
        and out_final_chunk_indices is None
    ):
        return

    num_seqs = cu_seqlens.shape[0] - 1
    expected_prefix_shape = (num_seqs + 1,)
    expected_final_shape = (num_seqs,)

    _validate_optional_output(
        "out_chunk_indices",
        out_chunk_indices,
        expected_shape=None,
        expected_device=cu_seqlens.device,
        expected_dtype=cu_seqlens.dtype,
    )
    if out_chunk_indices is not None and (
        out_chunk_indices.ndim != 2 or out_chunk_indices.shape[1] != 2
    ):
        raise ValueError(
            f"out_chunk_indices must have shape [num_chunks, 2], got {tuple(out_chunk_indices.shape)}"
        )
    _validate_optional_output(
        "out_chunk_offsets",
        out_chunk_offsets,
        expected_shape=expected_prefix_shape,
        expected_device=cu_seqlens.device,
        expected_dtype=cu_seqlens.dtype,
    )
    _validate_optional_output(
        "out_update_chunk_offsets",
        out_update_chunk_offsets,
        expected_shape=expected_prefix_shape,
        expected_device=cu_seqlens.device,
        expected_dtype=cu_seqlens.dtype,
    )
    _validate_optional_output(
        "out_final_chunk_indices",
        out_final_chunk_indices,
        expected_shape=expected_final_shape,
        expected_device=cu_seqlens.device,
        expected_dtype=cu_seqlens.dtype,
    )

    if num_seqs == 0:
        if out_chunk_offsets is not None:
            out_chunk_offsets.zero_()
        if out_update_chunk_offsets is not None:
            out_update_chunk_offsets.zero_()
        return

    chunk_counts = torch.empty(num_seqs, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    update_chunk_counts = torch.empty_like(chunk_counts)

    block_size = 256
    grid = (triton.cdiv(num_seqs, block_size),)
    _build_chunk_counts_kernel[grid](
        cu_seqlens_ptr=cu_seqlens,
        chunk_counts_ptr=chunk_counts,
        update_chunk_counts_ptr=update_chunk_counts,
        num_seqs=num_seqs,
        chunk_size=chunk_size,
        BLOCK_SIZE=block_size,
    )

    chunk_offsets = out_chunk_offsets
    if chunk_offsets is None and out_chunk_indices is not None:
        chunk_offsets = torch.empty(
            expected_prefix_shape,
            dtype=cu_seqlens.dtype,
            device=cu_seqlens.device,
        )

    if chunk_offsets is not None:
        chunk_offsets[:1].zero_()
        torch.cumsum(chunk_counts, dim=0, out=chunk_offsets[1:])

    if out_update_chunk_offsets is not None:
        out_update_chunk_offsets[:1].zero_()
        torch.cumsum(update_chunk_counts, dim=0, out=out_update_chunk_offsets[1:])

    if out_final_chunk_indices is not None:
        if out_update_chunk_offsets is not None:
            out_final_chunk_indices.copy_(out_update_chunk_offsets[1:])
        else:
            torch.cumsum(update_chunk_counts, dim=0, out=out_final_chunk_indices)
        out_final_chunk_indices.sub_(1)

    if out_chunk_indices is not None:
        total_chunks = out_chunk_indices.shape[0]
        if total_chunks == 0:
            return

        search_iters = max(1, (num_seqs - 1).bit_length())
        grid = (triton.cdiv(total_chunks, block_size),)
        _build_chunk_indices_kernel[grid](
            chunk_offsets_ptr=chunk_offsets,
            out_chunk_indices_ptr=out_chunk_indices,
            total_chunks=total_chunks,
            num_seqs=num_seqs,
            SEARCH_ITERS=search_iters,
            BLOCK_SIZE=block_size,
        )
