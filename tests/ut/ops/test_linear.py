import os
import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock, patch

import torch
from vllm import config

from tests.ut.base import TestBase
from vllm_ascend import ascend_config
from vllm_ascend.distributed import parallel_state
from vllm_ascend.ops.linear import (AscendMergedColumnParallelLinear,
                                    AscendQKVParallelLinear,
                                    AscendReplicatedLinear,
                                    AscendRowParallelLinear,
                                    AscendUnquantizedLinearMethod)
from vllm_ascend.ops.linear_op import SequenceColumnParallelOp


def _build_mock_vllm_config(first_layer_type="linear_attention", is_vl=True, speculative_method=None):
    hf_text_config = SimpleNamespace(
        model_type="qwen3_5_text",
        layer_types=[first_layer_type],
    )
    hf_config = SimpleNamespace(
        to_dict=lambda: {"vision_config": {}} if is_vl else {},
    )
    if not is_vl:
        hf_config = hf_text_config

    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=hf_config,
            hf_text_config=hf_text_config,
        ),
        speculative_config=SimpleNamespace(method=speculative_method),
    )


class BaseLinearTest(unittest.TestCase):

    def setUp(self):
        self.mock_group = mock.MagicMock()
        self.mock_group.world_size = 2
        self.mock_group.rank_in_group = 0

        parallel_state._MLP_TP = self.mock_group
        parallel_state._OTP = self.mock_group

        self.mock_ascend_config = MagicMock()
        self.mock_ascend_config.finegrained_tp_config.oproj_tensor_parallel_size = 2
        self.mock_ascend_config.finegrained_tp_config.mlp_tensor_parallel_size = 2

        self.patches = [
            patch("vllm_ascend.ascend_config.get_ascend_config",
                  return_value=self.mock_ascend_config),
            patch("vllm_ascend.distributed.parallel_state.get_otp_group",
                  return_value=self.mock_group),
            patch("vllm_ascend.distributed.parallel_state.get_mlp_tp_group",
                  return_value=self.mock_group),
            patch("vllm_ascend.ops.linear_op.get_tp_group",
                  return_value=self.mock_group),
            patch(
                "vllm.distributed.parallel_state.get_tp_group",
                return_value=self.mock_group,
            ),
            patch("vllm_ascend.ops.linear.enable_sp", return_value=False),
            patch("vllm_ascend.ops.linear_op.enable_dsa_cp", return_value=False),
            patch("vllm_ascend.ops.linear_op.enable_dsa_cp_with_layer_shard", return_value=False),
            patch("vllm_ascend.ops.linear_op.enable_sp", return_value=False),
            patch("vllm_ascend.utils.mlp_tp_enable", return_value=True),
            patch("vllm_ascend.utils.oproj_tp_enable", return_value=True)
        ]

        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()


class TestAscendUnquantizedLinearMethod(TestBase):

    def setUp(self):
        self.method = AscendUnquantizedLinearMethod()
        self.layer = mock.MagicMock()
        mock_dtype = mock.PropertyMock(return_value=torch.float16)
        type(self.layer.weight.data).dtype = mock_dtype

    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "0"})
    @mock.patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_with_nz0(self, mock_format_cast):
        self.method.process_weights_after_loading(self.layer)
        mock_format_cast.assert_not_called()

    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "1"})
    @mock.patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_with_nz1(self, mock_format_cast):
        self.method.process_weights_after_loading(self.layer)
        mock_format_cast.assert_not_called()

    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "2"})
    @mock.patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_with_nz2(self, mock_format_cast):
        self.method.process_weights_after_loading(self.layer)
        mock_format_cast.assert_called_once()


class TestAscendRowParallelLinear(BaseLinearTest):

    @patch("vllm_ascend.ops.linear_op.get_weight_prefetch_method",
           return_value=MagicMock())
    def test_mlp_optimize(self, mock_get_weight_prefetch_method):

        ascend_config._ASCEND_CONFIG = MagicMock()
        ascend_config._ASCEND_CONFIG.recompute_scheduler_enable = False
        ascend_config._ASCEND_CONFIG.finegrained_tp_config.mlp_tensor_parallel_size = 2
        ascend_config._ASCEND_CONFIG.ascend_scheduler_config.enabled = False

        linear = AscendRowParallelLinear(
            input_size=16,
            output_size=8,
            prefix="down_proj",
        )
        self.assertEqual(linear.custom_op.comm_group, parallel_state._MLP_TP)

        input_tensor = torch.randn(16, 8)
        linear(input_tensor)

    @patch("vllm_ascend.ops.linear_op.get_weight_prefetch_method",
           return_value=MagicMock())
    def test_oproj_tp(self, mock_get_weight_prefetch_method):

        config._current_vllm_config = MagicMock()

        ascend_config._ASCEND_CONFIG = MagicMock()
        ascend_config._ASCEND_CONFIG.recompute_scheduler_enable = False
        ascend_config._ASCEND_CONFIG.finegrained_tp_config.oproj_tensor_parallel_size = 2
        ascend_config._ASCEND_CONFIG.ascend_scheduler_config.enabled = False

        linear = AscendRowParallelLinear(
            input_size=16,
            output_size=8,
            prefix="o_proj",
        )
        self.assertEqual(linear.custom_op.comm_group, parallel_state._OTP)

        input_tensor = torch.randn(16, 8)
        linear(input_tensor)


class TestAscendMergedColumnParallelLinear(BaseLinearTest):

    def test_merged_mlp_tp_init(self):

        ascend_config._ASCEND_CONFIG = MagicMock()
        ascend_config._ASCEND_CONFIG.recompute_scheduler_enable = False
        ascend_config._ASCEND_CONFIG.finegrained_tp_config.mlp_tensor_parallel_size = 2
        ascend_config._ASCEND_CONFIG.ascend_scheduler_config.enabled = False

        linear = AscendMergedColumnParallelLinear(
            input_size=16,
            output_sizes=[8, 8],
            prefix="gate_up_proj",
        )
        self.assertEqual(linear.custom_op.comm_group, parallel_state._MLP_TP)

    @patch("vllm.config.get_current_vllm_config")
    def test_marks_qwen35_vl_linear_attention_first_projection(self, mock_get_current_vllm_config):
        mock_get_current_vllm_config.return_value = _build_mock_vllm_config()

        linear = AscendMergedColumnParallelLinear(
            input_size=16,
            output_sizes=[8, 8],
            prefix="model.language_model.model.layers.0.linear_attn.in_proj",
        )

        self.assertTrue(linear.fc1_skip_input_gather)

    @patch("vllm.config.get_current_vllm_config")
    def test_does_not_mark_non_first_qwen35_projection(self, mock_get_current_vllm_config):
        mock_get_current_vllm_config.return_value = _build_mock_vllm_config()

        linear = AscendMergedColumnParallelLinear(
            input_size=16,
            output_sizes=[8, 8],
            prefix="model.language_model.model.layers.1.linear_attn.in_proj",
        )

        self.assertFalse(linear.fc1_skip_input_gather)


class TestAscendQKVParallelLinear(BaseLinearTest):

    @patch("vllm.config.get_current_vllm_config")
    def test_marks_qwen35_vl_full_attention_first_projection(self, mock_get_current_vllm_config):
        mock_get_current_vllm_config.return_value = _build_mock_vllm_config(first_layer_type="full_attention")

        linear = AscendQKVParallelLinear(
            hidden_size=16,
            head_size=4,
            total_num_heads=4,
            total_num_kv_heads=4,
            prefix="model.language_model.model.layers.0.self_attn.qkv_proj",
        )

        self.assertTrue(linear.fc1_skip_input_gather)

class TestSequenceColumnParallelOp(unittest.TestCase):

    def _build_layer(self, skip_input_gather):
        return SimpleNamespace(
            bias=None,
            skip_bias_add=False,
            return_bias=False,
            gather_output=False,
            prefix="model.language_model.model.layers.0.linear_attn.in_proj",
            fc1_skip_input_gather=skip_input_gather,
            quant_method=SimpleNamespace(apply=lambda layer, input_, bias: input_ + 1),
        )

    @patch("torch.ops.vllm.maybe_all_gather_and_maybe_unpad")
    def test_skips_gather_for_marked_first_projection(self, mock_all_gather):
        mock_all_gather.side_effect = lambda tensor, _: tensor
        layer = self._build_layer(skip_input_gather=True)
        op = SequenceColumnParallelOp(layer)
        op.update_attrs()

        output, _ = op.apply_impl(torch.randn(4, 8))

        mock_all_gather.assert_not_called()
        self.assertEqual(output.shape, (4, 8))

    @patch("torch.ops.vllm.maybe_all_gather_and_maybe_unpad")
    def test_gathers_for_unmarked_projection(self, mock_all_gather):
        mock_all_gather.side_effect = lambda tensor, _: tensor
        layer = self._build_layer(skip_input_gather=False)
        op = SequenceColumnParallelOp(layer)
        op.update_attrs()

        output, _ = op.apply_impl(torch.randn(4, 8))

        mock_all_gather.assert_called_once()
        self.assertEqual(output.shape, (4, 8))


class TestAscendReplicatedLinear(BaseLinearTest):

    def test_init_disable_tp(self):
        linear = AscendReplicatedLinear(
            input_size=16,
            output_size=8,
        )
        self.assertTrue(
            isinstance(linear.quant_method, AscendUnquantizedLinearMethod))

    def test_init_without_disable_tp(self):
        linear = AscendReplicatedLinear(
            input_size=16,
            output_size=8,
        )
        self.assertTrue(
            isinstance(linear.quant_method, AscendUnquantizedLinearMethod))


if __name__ == '__main__':
    unittest.main()
