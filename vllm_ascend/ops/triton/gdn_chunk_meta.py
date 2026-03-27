# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.triton_utils import tl, triton


def _cdiv(x: int, y: int) -> int:
    triton_cdiv = getattr(triton, "cdiv", None)
    if triton_cdiv is not None:
        return triton_cdiv(x, y)
    return (x + y - 1) // y


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


def _validate_optional_output(
    name: str,
    tensor: torch.Tensor | None,
    *,
    expected_shape: tuple[int, ...] | None,
    expected_device: torch.device,
) -> None:
    if tensor is None:
        return
    if tensor.device != expected_device:
        raise ValueError(f"{name} must be on device {expected_device}, got {tensor.device}")
    if tensor.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must have int32 or int64 dtype, got {tensor.dtype}")
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
    )
    _validate_optional_output(
        "out_update_chunk_offsets",
        out_update_chunk_offsets,
        expected_shape=expected_prefix_shape,
        expected_device=cu_seqlens.device,
    )
    _validate_optional_output(
        "out_final_chunk_indices",
        out_final_chunk_indices,
        expected_shape=expected_final_shape,
        expected_device=cu_seqlens.device,
    )

    if num_seqs == 0:
        if out_chunk_offsets is not None:
            out_chunk_offsets.zero_()
        if out_update_chunk_offsets is not None:
            out_update_chunk_offsets.zero_()
        return

    chunk_counts = torch.empty(num_seqs, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    update_chunk_counts = torch.empty_like(chunk_counts)

    if hasattr(_build_chunk_counts_kernel, "__getitem__"):
        block_size = 256
        grid = (_cdiv(num_seqs, block_size),)
        _build_chunk_counts_kernel[grid](
            cu_seqlens_ptr=cu_seqlens,
            chunk_counts_ptr=chunk_counts,
            update_chunk_counts_ptr=update_chunk_counts,
            num_seqs=num_seqs,
            chunk_size=chunk_size,
            BLOCK_SIZE=block_size,
        )
    else:
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        torch.div(
            seq_lens + chunk_size - 1,
            chunk_size,
            rounding_mode="floor",
            out=chunk_counts,
        )
        update_chunk_counts.copy_(chunk_counts)
        update_chunk_counts.add_(1)

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

        repeat_counts = chunk_counts.to(torch.int64)
        seq_indices = torch.repeat_interleave(
            torch.arange(num_seqs, device=cu_seqlens.device, dtype=out_chunk_indices.dtype),
            repeat_counts,
        )
        if seq_indices.shape[0] != total_chunks:
            raise ValueError(
                "out_chunk_indices has incompatible first dimension for the provided cu_seqlens"
            )
        chunk_starts = torch.repeat_interleave(
            chunk_offsets[:-1].to(dtype=out_chunk_indices.dtype),
            repeat_counts,
        )
        local_indices = (
            torch.arange(total_chunks, device=cu_seqlens.device, dtype=out_chunk_indices.dtype)
            - chunk_starts
        )
        out_chunk_indices[:, 0].copy_(seq_indices)
        out_chunk_indices[:, 1].copy_(local_indices)
