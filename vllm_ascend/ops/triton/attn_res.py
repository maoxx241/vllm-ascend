# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2


@triton.jit
def _apply_attn_res_kernel(
    block_residual_ptr,
    prefix_sum_ptr,
    norm_w_ptr,
    proj_w_ptr,
    out_ptr,
    N: tl.constexpr,
    H: tl.constexpr,
    B: tl.constexpr,
    EPS: tl.constexpr,
    NUM_CORES: tl.constexpr,
    NB: tl.constexpr,
):
    block_size = (N - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    tok0 = pid * block_size
    if tok0 >= N:
        return
    tok1 = tl.minimum(tok0 + block_size, N)

    cols = tl.arange(0, H)
    stream_indices = tl.arange(0, NB)

    norm_w = tl.load(norm_w_ptr + cols).to(tl.float32)
    proj_w = tl.load(proj_w_ptr + cols).to(tl.float32)
    score_weight = norm_w * proj_w

    block_residual_stride = B * H
    for token_idx in range(tok0, tok1):
        scores = tl.full([NB], -float("inf"), dtype=tl.float32)
        for stream_idx in range(B + 1):
            if stream_idx < B:
                values = tl.load(
                    block_residual_ptr
                    + token_idx * block_residual_stride
                    + stream_idx * H
                    + cols
                ).to(tl.float32)
            else:
                values = tl.load(prefix_sum_ptr + token_idx * H + cols).to(tl.float32)
            mean_square = tl.sum(values * values) / H
            normalized = values * tl.rsqrt(mean_square + EPS)
            score = tl.sum(normalized * score_weight)
            scores = tl.where(stream_indices == stream_idx, score, scores)

        scores_max = tl.max(scores)
        exp_scores = tl.exp(scores - scores_max)
        weights = exp_scores / tl.sum(exp_scores)

        output = tl.zeros([H], dtype=tl.float32)
        for stream_idx in range(B + 1):
            if stream_idx < B:
                values = tl.load(
                    block_residual_ptr
                    + token_idx * block_residual_stride
                    + stream_idx * H
                    + cols
                ).to(tl.float32)
            else:
                values = tl.load(prefix_sum_ptr + token_idx * H + cols).to(tl.float32)
            stream_weight = tl.sum(tl.where(stream_indices == stream_idx, weights, 0.0))
            output += stream_weight * values

        tl.store(
            out_ptr + token_idx * H + cols,
            output.to(out_ptr.dtype.element_ty),
        )


def apply_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    projection: nn.Module,
    norm: nn.Module,
) -> torch.Tensor:
    """Apply Kimi K3's learned mixture over its residual streams."""
    num_tokens, hidden_size = prefix_sum.shape
    num_block_residuals = block_residual.shape[1]
    projection_weight = projection.weight.squeeze(0)

    output = torch.empty(
        (num_tokens, hidden_size),
        dtype=prefix_sum.dtype,
        device=prefix_sum.device,
    )
    num_streams_padded = next_power_of_2(num_block_residuals + 1)
    device_properties = triton.runtime.driver.active.utils.get_device_properties(prefix_sum.device)
    num_vector_cores = device_properties.get("num_vectorcore", -1)
    if num_vector_cores <= 0:
        raise RuntimeError("Failed to detect the number of Ascend vector cores")

    _apply_attn_res_kernel[(num_vector_cores,)](
        block_residual,
        prefix_sum,
        norm.weight,
        projection_weight,
        output,
        N=num_tokens,
        H=hidden_size,
        B=num_block_residuals,
        EPS=norm.variance_epsilon,
        NUM_CORES=num_vector_cores,
        NB=num_streams_padded,
        multibuffer=True,
    )
    return output
