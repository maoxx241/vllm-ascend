#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

import pytest
import torch

from vllm_ascend.ops.activation import (
    AscendSituAndMul,
    SituActivationConfig,
    situ_and_mul,
)


def _situ_and_mul_reference(
    x: torch.Tensor,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    d = x.shape[-1] // 2
    gate = x[..., :d].float()
    up = x[..., d:].float()
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (gate * up).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("linear_beta", [None, 25.0])
def test_situ_matches_kimi_fp32_intermediate_formula(dtype, linear_beta):
    x = torch.tensor(
        [[-40.0, -4.25, 0.125, 9.5, -80.0, -1.5, 3.25, 70.0]],
        dtype=dtype,
    )

    result = situ_and_mul(x, beta=4.0, linear_beta=linear_beta)

    expected = _situ_and_mul_reference(x, beta=4.0, linear_beta=linear_beta)
    assert result.dtype == dtype
    assert result.shape == (1, 4)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_situ_module_carries_runtime_parameters():
    x = torch.randn(3, 12, dtype=torch.float16)
    layer = AscendSituAndMul(beta=4.0, linear_beta=25.0)

    result = layer(x)

    expected = _situ_and_mul_reference(x, beta=4.0, linear_beta=25.0)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert layer.config == SituActivationConfig(beta=4.0, linear_beta=25.0)
