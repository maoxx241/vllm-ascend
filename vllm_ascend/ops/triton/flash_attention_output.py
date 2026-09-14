# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


@triton.jit(do_not_specialize=["num_tokens", "output_tokens"])
def _flash_attention_output_kernel(
    result,
    token_live,
    output,
    num_tokens,
    output_tokens,
    result_stride,
    output_stride,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    column_blocks = tl.cdiv(HIDDEN, BLOCK)
    for block in range(tl.program_id(0), output_tokens * column_blocks, tl.num_programs(0)):
        row = block // column_blocks
        columns = block % column_blocks * BLOCK + tl.arange(0, BLOCK)
        live = tl.load(token_live + row, mask=row < num_tokens, other=0) != 0
        # A masked load avoids propagating NaN/Inf from inactive rows.
        values = tl.load(
            result + row * result_stride + columns,
            mask=(columns < HIDDEN) & live,
            other=0,
        )
        tl.store(output + row * output_stride + columns, values, mask=columns < HIDDEN)


def flash_attention_output(
    result: torch.Tensor,
    token_live: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Write contiguous hidden rows, zeroing inactive tokens and graph padding."""
    if output.numel() == 0:
        return output
    init_device_properties_triton()
    # A fixed 1-D tile bounds UB use independently of hidden size or batch.
    block = 1024
    tasks = output.shape[0] * triton.cdiv(result.shape[1], block)
    grid = (min(tasks, get_vectorcore_num()),)
    _flash_attention_output_kernel[grid](
        result,
        token_live,
        output,
        result.shape[0],
        output.shape[0],
        result.stride(0),
        output.stride(0),
        HIDDEN=result.shape[1],
        BLOCK=block,
        multibuffer=False,
    )
    return output
