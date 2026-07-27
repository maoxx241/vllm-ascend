# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm_ascend.ops.activation import SituActivationConfig
from vllm_ascend.ops.fused_moe import situ


def test_dynamic_situ_quant_owns_custom_op_abi(monkeypatch):
    quantized = torch.randn(2, 4)
    scale = torch.randn(2, 1)
    custom_op = MagicMock(return_value=(quantized, scale))
    monkeypatch.setattr(
        situ.torch.ops,
        "_C_ascend",
        SimpleNamespace(dequant_situ_quant=custom_op),
        raising=False,
    )
    x = torch.randn(2, 8)
    weight_scale = torch.randn(8)
    activation_scale = torch.randn(2, 1)
    activation = SituActivationConfig(beta=4.0, linear_beta=25.0)

    result = situ.dynamic_situ_quant(
        x,
        activation,
        weight_scale=weight_scale,
        activation_scale=activation_scale,
    )

    assert result[0] is quantized
    assert result[1] is scale
    custom_op.assert_called_once_with(
        x=x,
        weight_scale=weight_scale,
        activation_scale=activation_scale,
        bias=None,
        quant_scale=None,
        quant_offset=None,
        group_index=None,
        beta=4.0,
        linear_beta=25.0,
        activate_left=True,
        quant_mode="dynamic",
    )


def test_mxfp_situ_quant_owns_destination_type_abi(monkeypatch):
    quantized = torch.randn(2, 4)
    scale = torch.randn(2, 1)
    custom_op = MagicMock(return_value=(quantized, scale))
    monkeypatch.setattr(
        situ.torch.ops,
        "_C_ascend",
        SimpleNamespace(situ_mx_quant=custom_op),
        raising=False,
    )
    x = torch.randn(2, 8)
    activation = SituActivationConfig(beta=4.0, linear_beta=None)

    result = situ.mxfp_situ_quant(x, activation)

    assert result[0] is quantized
    assert result[1] is scale
    custom_op.assert_called_once_with(
        x=x,
        beta=4.0,
        linear_beta=0.0,
        activate_left=True,
        dst_type=situ.SITU_MX_QUANT_DST_TYPE_E4M3FN,
    )
    assert situ.SITU_MX_QUANT_DST_TYPE_E4M3FN == 36
