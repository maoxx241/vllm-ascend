from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from compressed_tensors.quantization import QuantizationType
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import RowParallelLinear, UnquantizedLinearMethod

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import COMPRESSED_TENSORS_W8A8_CONFIG
from vllm_ascend.quantization.compressed_tensors_config import AscendCompressedTensorsConfig
from vllm_ascend.quantization.method_adapters import AscendLinearMethod
from vllm_ascend.quantization.methods import AscendW8A8DynamicLinearMethod
from vllm_ascend.quantization.methods.w4a8_mxfp4 import (
    AscendW4A8MXFPCompressedTensorsFusedMoEMethod,
    AscendW4A8MXFPCompressedTensorsLinearMethod,
)
from vllm_ascend.utils import COMPRESSED_TENSORS_METHOD


class TestAscendCompressedTensorsQuanType(TestBase):
    def setUp(self):
        self.config = AscendCompressedTensorsConfig(
            target_scheme_map={"Linear": {}},
            ignore=["lm_head"],
            quant_format="",
            config={},
        )

    def _make_weight_quant(self, num_bits=8, strategy="channel", dynamic=False, symmetric=True, group_size=None):
        mock = MagicMock()
        mock.num_bits = num_bits
        mock.strategy = strategy
        mock.dynamic = dynamic
        mock.symmetric = symmetric
        mock.group_size = group_size
        return mock

    def _make_input_quant(self, num_bits=8, strategy="tensor", dynamic=False, symmetric=True):
        mock = MagicMock()
        mock.num_bits = num_bits
        mock.strategy = strategy
        mock.dynamic = dynamic
        mock.symmetric = symmetric
        return mock

    def test_detect_w8a8_static(self):
        weight = self._make_weight_quant(num_bits=8, strategy="channel", dynamic=False, symmetric=True)
        input_q = self._make_input_quant(num_bits=8, strategy="tensor", dynamic=False, symmetric=True)
        result = self.config._detect_quant_type(weight, input_q, "int-quantized")
        self.assertEqual(result, "W8A8")

    def test_detect_w8a8_dynamic(self):
        weight = self._make_weight_quant(num_bits=8, strategy="channel", dynamic=False, symmetric=True)
        input_q = self._make_input_quant(num_bits=8, strategy="token", dynamic=True, symmetric=True)
        result = self.config._detect_quant_type(weight, input_q, "int-quantized")
        self.assertEqual(result, "W8A8_DYNAMIC")

    def test_detect_w4a8_dynamic(self):
        weight = self._make_weight_quant(num_bits=4, strategy="channel", dynamic=False, symmetric=True)
        input_q = self._make_input_quant(num_bits=8, strategy="token", dynamic=True, symmetric=True)
        result = self.config._detect_quant_type(weight, input_q, "int-quantized")
        self.assertEqual(result, "W4A8_DYNAMIC")

    def test_detect_w4a16(self):
        weight = MagicMock()
        weight.num_bits = 4
        weight.strategy = "group"
        weight.dynamic = False
        weight.type = QuantizationType.INT
        result = self.config._detect_quant_type(weight, None, None)
        self.assertEqual(result, "W4A16")

    def test_packed_mxfp4_selects_explicit_compressed_tensors_schemes(self):
        self.config.quant_format = "mxfp4-pack-quantized"
        weight = self._make_weight_quant(
            num_bits=4,
            strategy="group",
            dynamic=False,
        )
        weight.type = QuantizationType.FLOAT
        weight.group_size = 32
        vllm_config = SimpleNamespace(
            quant_config=SimpleNamespace(quant_description={"group_size": 32}),
            compilation_config=SimpleNamespace(mode=None),
            model_config=SimpleNamespace(enforce_eager=True),
        )
        ascend_config = SimpleNamespace(
            eplb_config=SimpleNamespace(dynamic_eplb=False),
        )

        with (
            patch(
                "vllm_ascend.quantization.methods.w4a8_mxfp4.ensure_mxfp4_linear_available",
            ),
            patch(
                "vllm_ascend.quantization.methods.w4a8_mxfp4.get_current_vllm_config",
                return_value=vllm_config,
            ),
            patch(
                "vllm_ascend.quantization.methods.w4a8_mxfp4.get_ascend_config",
                return_value=ascend_config,
            ),
            patch(
                "vllm_ascend.quantization.methods.w4a8_mxfp4.get_ep_group",
            ),
        ):
            linear_scheme = self.config._create_scheme_for_layer_type(
                weight,
                None,
                None,
                "linear",
            )
            moe_scheme = self.config._create_scheme_for_layer_type(
                weight,
                None,
                None,
                "moe",
            )

        self.assertIsInstance(
            linear_scheme,
            AscendW4A8MXFPCompressedTensorsLinearMethod,
        )
        self.assertIsInstance(
            moe_scheme,
            AscendW4A8MXFPCompressedTensorsFusedMoEMethod,
        )
        self.assertEqual(
            set(linear_scheme.get_weight(64, 4, torch.bfloat16)),
            {"weight_packed"},
        )
        self.assertEqual(
            set(moe_scheme.get_weight(2, 8, 64, torch.bfloat16)),
            {"w13_weight_packed", "w2_weight_packed"},
        )

    @patch(
        "vllm_ascend.quantization.methods.w4a8_mxfp4.torch_npu.npu_format_cast",
        side_effect=lambda tensor, *_args, **_kwargs: tensor,
    )
    def test_packed_mxfp4_normalizes_checkpoint_names_after_loading(self, _mock_format_cast):
        linear_scheme = AscendW4A8MXFPCompressedTensorsLinearMethod.__new__(AscendW4A8MXFPCompressedTensorsLinearMethod)
        linear = torch.nn.Module()
        linear.weight_packed = torch.nn.Parameter(
            torch.zeros(2, 4, dtype=torch.uint8),
            requires_grad=False,
        )
        linear.weight_scale = torch.nn.Parameter(
            torch.zeros(2, 2, dtype=torch.uint8),
            requires_grad=False,
        )

        linear_scheme.process_weights_after_loading(linear)

        self.assertFalse(hasattr(linear, "weight_packed"))
        self.assertEqual(linear.weight.shape, (4, 2))

        moe_scheme = AscendW4A8MXFPCompressedTensorsFusedMoEMethod.__new__(
            AscendW4A8MXFPCompressedTensorsFusedMoEMethod
        )
        moe = torch.nn.Module()
        moe.w13_weight_packed = torch.nn.Parameter(
            torch.zeros(1, 4, 2, dtype=torch.uint8),
            requires_grad=False,
        )
        moe.w2_weight_packed = torch.nn.Parameter(
            torch.zeros(1, 2, 2, dtype=torch.uint8),
            requires_grad=False,
        )
        moe.w13_weight_scale = torch.nn.Parameter(
            torch.zeros(1, 4, 2, dtype=torch.uint8),
            requires_grad=False,
        )
        moe.w2_weight_scale = torch.nn.Parameter(
            torch.zeros(1, 2, 2, dtype=torch.uint8),
            requires_grad=False,
        )

        moe_scheme.process_weights_after_loading(moe)

        self.assertFalse(hasattr(moe, "w13_weight_packed"))
        self.assertFalse(hasattr(moe, "w2_weight_packed"))
        self.assertEqual(moe.w13_weight.shape, (1, 2, 4))
        self.assertEqual(moe.w2_weight.shape, (1, 2, 2))

    def test_detect_unsupported_raises(self):
        weight = self._make_weight_quant(num_bits=2, strategy="channel", dynamic=False, symmetric=True)
        input_q = self._make_input_quant(num_bits=2, strategy="tensor", dynamic=False, symmetric=True)
        with self.assertRaises(NotImplementedError):
            self.config._detect_quant_type(weight, input_q, "int_quantized")


class TestAscendCompressedTensorsConfigGetQuantMethod(TestBase):
    def setUp(self):
        self.config = AscendCompressedTensorsConfig.from_config(COMPRESSED_TENSORS_W8A8_CONFIG)

    @patch("vllm_ascend.quantization.method_adapters.AscendLinearMethod.__init__")
    def test_get_linear_quant_method(self, mock_method):
        mock_method.return_value = None
        layer = MagicMock(spec=RowParallelLinear)
        result = self.config.get_quant_method(layer, "model.layers.0.self_attn.q_proj")
        self.assertEqual(layer.ascend_quant_method, COMPRESSED_TENSORS_METHOD)
        self.assertTrue(isinstance(result, AscendLinearMethod))
        self.assertTrue(isinstance(layer.scheme, AscendW8A8DynamicLinearMethod))

    def test_get_linear_unquantized_method(self):
        layer = MagicMock(spec=RowParallelLinear)
        result = self.config.get_quant_method(layer, "lm_head")
        self.assertEqual(layer.ascend_quant_method, COMPRESSED_TENSORS_METHOD)
        self.assertTrue(isinstance(result, UnquantizedLinearMethod))

    def test_adds_routed_experts_target_for_linear_scheme(self):
        linear_scheme = self.config.target_scheme_map["Linear"]

        self.config._add_fused_moe_to_target_scheme_map()

        self.assertIs(self.config.target_scheme_map["RoutedExperts"], linear_scheme)

    @patch("vllm_ascend.quantization.compressed_tensors_config.find_matched_target", return_value=None)
    def test_get_scheme_dict_returns_none_for_unmatched_target(self, _mock_find_target):
        layer = MagicMock(spec=Attention)

        result = self.config.get_scheme_dict(layer, "model.layers.0.self_attn.attn")

        self.assertIsNone(result)

    @patch(
        "vllm_ascend.quantization.compressed_tensors_config.find_matched_target",
        return_value="Linear",
    )
    def test_get_scheme_dict_returns_none_for_none_scheme(self, _mock_find_target):
        self.config.target_scheme_map["Linear"] = None
        layer = MagicMock(spec=Attention)

        result = self.config.get_scheme_dict(layer, "model.layers.0.self_attn.attn")

        self.assertIsNone(result)

    @patch("vllm_ascend.quantization.method_adapters.AscendFusedMoEMethod")
    def test_get_routed_experts_quant_method(self, mock_method):
        layer = RoutedExperts.__new__(RoutedExperts)
        torch.nn.Module.__init__(layer)
        layer.moe_config = MagicMock()
        moe_scheme = MagicMock()

        with patch.object(self.config, "_get_moe_scheme", return_value=moe_scheme):
            result = self.config.get_quant_method(layer, "model.layers.0.mlp.experts")

        self.assertIs(result, mock_method.return_value)
        self.assertEqual(layer.ascend_quant_method, COMPRESSED_TENSORS_METHOD)
        self.assertIs(layer.scheme, moe_scheme)
        mock_method.assert_called_once_with(moe_scheme, layer.moe_config, None)

    def test_no_quant_method(self):
        layer = MagicMock(spec=Attention)
        result = self.config.get_quant_method(layer, "attn")
        self.assertIsNone(result)
