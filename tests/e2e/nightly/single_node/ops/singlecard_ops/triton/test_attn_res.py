# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from vllm_ascend.ops.triton.attn_res import apply_attn_res


@pytest.mark.parametrize(
    ("num_tokens", "num_block_residuals"),
    [
        pytest.param(1, 1, id="decode-min-streams"),
        pytest.param(32, 8, id="decode-max-streams"),
        pytest.param(129, 8, id="prefill-multiple-tokens-per-core"),
    ],
)
@torch.inference_mode()
def test_kimi_k3_attn_res(num_tokens: int, num_block_residuals: int):
    """Check the fused op with Kimi K3's hidden and residual-block sizes."""
    torch.manual_seed(42)
    device = "npu"
    hidden_size = 7168
    epsilon = 1e-5

    prefix_sum = torch.randn(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    block_residual = torch.randn(
        num_tokens,
        num_block_residuals,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    projection = nn.Linear(hidden_size, 1, bias=False, device=device, dtype=torch.bfloat16)
    norm = nn.Module()
    norm.register_parameter(
        "weight",
        nn.Parameter(torch.randn(hidden_size, dtype=torch.bfloat16, device=device)),
    )
    norm.variance_epsilon = epsilon

    actual = apply_attn_res(prefix_sum, block_residual, projection, norm)

    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    values_fp32 = values.float()
    normalized = values_fp32 * torch.rsqrt(
        values_fp32.square().mean(-1, keepdim=True) + epsilon
    )
    score_weight = norm.weight.float() * projection.weight.squeeze(0).float()
    probabilities = (normalized * score_weight).sum(-1).softmax(-1).unsqueeze(1)
    expected = torch.matmul(probabilities, values_fp32).squeeze(1).to(values.dtype)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
