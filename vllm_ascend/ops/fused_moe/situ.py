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

import torch

from vllm_ascend.ops.activation import SituActivationConfig

# ``situ_mx_quant`` follows the Ascend custom-op ABI, where 36 identifies
# FLOAT8_E4M3FN output. Keep this ABI value local to the adapter rather than
# repeating it in MoE execution paths.
SITU_MX_QUANT_DST_TYPE_E4M3FN = 36


def dynamic_situ_quant(
    x: torch.Tensor,
    activation: SituActivationConfig,
    *,
    weight_scale: torch.Tensor | None = None,
    activation_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply SiTU and dynamically quantize its output through the Ascend ABI."""
    return torch.ops._C_ascend.dequant_situ_quant(
        x=x,
        weight_scale=weight_scale,
        activation_scale=activation_scale,
        bias=None,
        quant_scale=None,
        quant_offset=None,
        group_index=None,
        beta=activation.beta,
        linear_beta=activation.linear_beta,
        activate_left=True,
        quant_mode="dynamic",
    )


def mxfp_situ_quant(
    x: torch.Tensor,
    activation: SituActivationConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply SiTU and MXFP8-quantize its output through the Ascend ABI."""
    return torch.ops._C_ascend.situ_mx_quant(
        x=x,
        beta=activation.beta,
        linear_beta=activation.linear_beta or 0.0,
        activate_left=True,
        dst_type=SITU_MX_QUANT_DST_TYPE_E4M3FN,
    )
