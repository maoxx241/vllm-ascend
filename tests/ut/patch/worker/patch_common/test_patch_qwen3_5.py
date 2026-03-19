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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F
import vllm.model_executor.layers.linear as linear_module
import vllm.model_executor.parameter as parameter_module
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.config import set_current_vllm_config
from vllm_ascend.ops import linear as ascend_linear_module

from tests.ut.base import PytestBase
from vllm_ascend.patch.worker.patch_qwen3_5 import (
    _patched_qwen3_5_model_load_weights,
    qwen35_packed_in_proj_output_sizes,
    split_qwen35_packed_in_proj_output,
)


class DummyQwen35Model:
    def __init__(self, params):
        self._params = params
        self.config = SimpleNamespace()

    def named_parameters(self):
        return self._params.items()

    def named_modules(self):
        return []

    def get_expert_mapping(self):
        return []


def _make_weight(output_size: int, hidden_size: int, offset: float) -> torch.Tensor:
    return (
        torch.arange(output_size * hidden_size, dtype=torch.float32).reshape(
            output_size,
            hidden_size,
        )
        + offset
    )


def _local_shard(weight: torch.Tensor, tp_size: int, tp_rank: int) -> torch.Tensor:
    shard_size = weight.size(0) // tp_size
    start = tp_rank * shard_size
    return weight.narrow(0, start, shard_size)


@pytest.fixture
def default_vllm_config():
    mock_config = MagicMock()
    mock_config.compilation_config.dispatch_forward_backend = "eager"
    mock_config.compilation_config.custom_ops = ["all"]
    with set_current_vllm_config(mock_config):
        yield mock_config


class TestPatchQwen35PackedInProj(PytestBase):
    @pytest.mark.parametrize("tp_size", [1, 2])
    def test_split_qwen35_packed_in_proj_matches_legacy_reference(self, tp_size):
        hidden_size = 7
        key_dim = 24
        num_v_heads = 4
        head_v_dim = 8
        value_dim = num_v_heads * head_v_dim
        num_tokens = 5

        local_q = _make_weight(key_dim // tp_size, hidden_size, 0)
        local_k = _make_weight(key_dim // tp_size, hidden_size, 1000)
        local_v = _make_weight(value_dim // tp_size, hidden_size, 2000)
        local_z = _make_weight(value_dim // tp_size, hidden_size, 3000)
        local_b = _make_weight(num_v_heads // tp_size, hidden_size, 4000)
        local_a = _make_weight(num_v_heads // tp_size, hidden_size, 5000)

        hidden_states = _make_weight(num_tokens, hidden_size, 0.5)
        legacy_qkvz_weight = torch.cat((local_q, local_k, local_v, local_z), dim=0)
        legacy_ba_weight = torch.cat((local_b, local_a), dim=0)
        packed_weight = torch.cat(
            (local_q, local_k, local_v, local_z, local_b, local_a),
            dim=0,
        )

        mixed_qkvz = F.linear(hidden_states, legacy_qkvz_weight)
        mixed_qkv_ref, z_ref = mixed_qkvz.split(
            [
                (key_dim * 2 + value_dim) // tp_size,
                value_dim // tp_size,
            ],
            dim=-1,
        )
        z_ref = z_ref.reshape(num_tokens, -1, head_v_dim)
        ba = F.linear(hidden_states, legacy_ba_weight)
        b_ref, a_ref = ba.chunk(2, dim=-1)
        b_ref = b_ref.contiguous()
        a_ref = a_ref.contiguous()

        projected_states = F.linear(hidden_states, packed_weight)
        mixed_qkv, z, b, a = split_qwen35_packed_in_proj_output(
            projected_states,
            key_dim,
            value_dim,
            num_v_heads,
            tp_size,
            head_v_dim,
        )

        torch.testing.assert_close(mixed_qkv, mixed_qkv_ref, rtol=0, atol=0)
        torch.testing.assert_close(z, z_ref, rtol=0, atol=0)
        torch.testing.assert_close(b, b_ref, rtol=0, atol=0)
        torch.testing.assert_close(a, a_ref, rtol=0, atol=0)

    @pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 0), (2, 1)])
    def test_qwen35_model_load_weights_maps_into_packed_in_proj(
        self,
        monkeypatch,
        tp_size,
        tp_rank,
        default_vllm_config,
    ):
        hidden_size = 7
        key_dim = 24
        num_v_heads = 4
        head_v_dim = 8
        value_dim = num_v_heads * head_v_dim

        monkeypatch.setenv("VLLM_ASCEND_VALIDATE_QWEN35_PACKED_INPROJ", "1")
        monkeypatch.setattr(
            linear_module,
            "get_tensor_model_parallel_world_size",
            lambda: tp_size,
        )
        monkeypatch.setattr(
            linear_module,
            "get_tensor_model_parallel_rank",
            lambda: tp_rank,
        )
        monkeypatch.setattr(
            parameter_module,
            "get_tensor_model_parallel_world_size",
            lambda: tp_size,
        )
        monkeypatch.setattr(
            parameter_module,
            "get_tensor_model_parallel_rank",
            lambda: tp_rank,
        )
        monkeypatch.setattr(
            ascend_linear_module,
            "get_parallel_op",
            lambda disable_tp, prefix, layer, direct: (None, tp_rank, tp_size),
        )

        packed_layer = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=qwen35_packed_in_proj_output_sizes(
                key_dim,
                value_dim,
                num_v_heads,
            ),
            bias=False,
            prefix="layers.0.linear_attn.in_proj",
        )
        legacy_qkvz_layer = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[key_dim, key_dim, value_dim, value_dim],
            bias=False,
            prefix="layers.0.linear_attn.in_proj_qkvz",
        )
        legacy_ba_layer = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[num_v_heads, num_v_heads],
            bias=False,
            prefix="layers.0.linear_attn.in_proj_ba",
        )

        model = DummyQwen35Model(
            {
                "layers.0.linear_attn.in_proj.weight": packed_layer.weight,
                "layers.0.linear_attn.in_proj_qkvz.weight": legacy_qkvz_layer.weight,
                "layers.0.linear_attn.in_proj_ba.weight": legacy_ba_layer.weight,
            }
        )

        q = _make_weight(key_dim, hidden_size, 0)
        k = _make_weight(key_dim, hidden_size, 1000)
        v = _make_weight(value_dim, hidden_size, 2000)
        z = _make_weight(value_dim, hidden_size, 3000)
        b = _make_weight(num_v_heads, hidden_size, 4000)
        a = _make_weight(num_v_heads, hidden_size, 5000)

        loaded_params = _patched_qwen3_5_model_load_weights(
            model,
            [
                (
                    "layers.0.linear_attn.in_proj_qkv.weight",
                    torch.cat((q, k, v), dim=0),
                ),
                ("layers.0.linear_attn.in_proj_z.weight", z),
                ("layers.0.linear_attn.in_proj_b.weight", b),
                ("layers.0.linear_attn.in_proj_a.weight", a),
            ],
        )

        expected_packed = torch.cat(
            (
                _local_shard(q, tp_size, tp_rank),
                _local_shard(k, tp_size, tp_rank),
                _local_shard(v, tp_size, tp_rank),
                _local_shard(z, tp_size, tp_rank),
                _local_shard(b, tp_size, tp_rank),
                _local_shard(a, tp_size, tp_rank),
            ),
            dim=0,
        )
        expected_qkvz = torch.cat(
            (
                _local_shard(q, tp_size, tp_rank),
                _local_shard(k, tp_size, tp_rank),
                _local_shard(v, tp_size, tp_rank),
                _local_shard(z, tp_size, tp_rank),
            ),
            dim=0,
        )
        expected_ba = torch.cat(
            (
                _local_shard(b, tp_size, tp_rank),
                _local_shard(a, tp_size, tp_rank),
            ),
            dim=0,
        )

        torch.testing.assert_close(
            packed_layer.weight.data,
            expected_packed,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            legacy_qkvz_layer.weight.data,
            expected_qkvz,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            legacy_ba_layer.weight.data,
            expected_ba,
            rtol=0,
            atol=0,
        )
        assert loaded_params == {
            "layers.0.linear_attn.in_proj.weight",
            "layers.0.linear_attn.in_proj_qkvz.weight",
            "layers.0.linear_attn.in_proj_ba.weight",
        }


class _PassthroughLayerNorm:
    def __call__(self, hidden_states, residual=None):
        if residual is None:
            return hidden_states
        return hidden_states, residual

