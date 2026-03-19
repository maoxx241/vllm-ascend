#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
# mypy: ignore-errors

import json
import os
from collections.abc import Iterable
from itertools import islice

import torch
from einops import rearrange
from vllm.distributed import get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLMBase,
    Qwen3_5DecoderLayer,
    Qwen3_5GatedDeltaNet,
    Qwen3_5Model,
    default_weight_loader,
    is_pp_missing_parameter,
    logger,
    maybe_remap_kv_scale_name,
)
from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.attention.utils import maybe_save_kv_layer_to_connector
from vllm_ascend.ops.linear import AscendMergedColumnParallelLinear
from vllm_ascend.ops.triton.fla.prefill_precompute import (
    build_gdn_prefill_precomputed,
)
from vllm_ascend.ops.triton.fla.sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)
from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch
from vllm_ascend.utils import enable_sp


_VALIDATE_PACKED_IN_PROJ_ENV = "VLLM_ASCEND_VALIDATE_QWEN35_PACKED_INPROJ"
_QWEN35_ALIAS_DEBUG_JSONL = os.environ.get("QWEN35_ALIAS_DEBUG_JSONL")
_QWEN35_DECODER_DUMP_DIR = os.environ.get("QWEN35_DECODER_DUMP_DIR")
_QWEN35_DECODER_DUMP_LIMIT = int(os.environ.get("QWEN35_DECODER_DUMP_LIMIT", "4"))
_QWEN35_DECODER_DUMP_SKIP = int(os.environ.get("QWEN35_DECODER_DUMP_SKIP", "0"))
_QWEN35_DECODER_DUMP_LAYER_TYPE = os.environ.get(
    "QWEN35_DECODER_DUMP_LAYER_TYPE",
    "full_attention",
)
_ORIGINAL_QWEN3_5_MODEL_FORWARD = Qwen3_5Model.forward
_ORIGINAL_QWEN3_5_MODEL_LOAD_WEIGHTS = Qwen3_5Model.load_weights
_ORIGINAL_QWEN3_5_GATED_DELTA_NET_INIT = Qwen3_5GatedDeltaNet.__init__
_ORIGINAL_QWEN3_5_GATED_DELTA_NET_FORWARD = Qwen3_5GatedDeltaNet.forward
_QWEN35_DECODER_DUMP_COUNTERS: dict[str, int] = {}

_QWEN35_PACKED_MODULES_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj": ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"],
}

_QWEN35_STACKED_PARAMS_MAPPING = [
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
    ("in_proj", "in_proj_qkv", (0, 1, 2)),
    ("in_proj", "in_proj_z", 3),
    ("in_proj", "in_proj_b", 4),
    ("in_proj", "in_proj_a", 5),
]


def _should_write_alias_debug() -> bool:
    return bool(_QWEN35_ALIAS_DEBUG_JSONL) and not _is_compiling_debug()


def _tensor_debug_ptr(tensor: torch.Tensor | None) -> dict[str, int | list[int] | str] | None:
    if not _should_write_alias_debug():
        return None
    if tensor is None:
        return None
    return {
        "data_ptr": int(tensor.data_ptr()),
        "storage_ptr": int(tensor.untyped_storage().data_ptr()),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }


def _same_storage_debug(
    lhs: torch.Tensor | None,
    rhs: torch.Tensor | None,
) -> bool | None:
    if not _should_write_alias_debug():
        return None
    if lhs is None or rhs is None:
        return None
    return lhs.untyped_storage().data_ptr() == rhs.untyped_storage().data_ptr()


def _is_compiling_debug() -> bool:
    try:
        if torch.compiler.is_compiling():
            return True
    except Exception:
        pass
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None:
        try:
            return bool(dynamo.is_compiling())
        except Exception:
            pass
    return False


def _write_alias_debug(stage: str, **fields) -> None:
    if not _should_write_alias_debug():
        return
    record = {"stage": stage, "pid": os.getpid(), **fields}
    with open(_QWEN35_ALIAS_DEBUG_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _get_debug_rank() -> int:
    try:
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()
    except Exception:
        pass
    return -1


def _dump_decoder_tensors(stage: str, layer_type: str, **tensors: torch.Tensor | None) -> None:
    if not _QWEN35_DECODER_DUMP_DIR or _is_compiling_debug():
        return
    if _QWEN35_DECODER_DUMP_LAYER_TYPE and layer_type != _QWEN35_DECODER_DUMP_LAYER_TYPE:
        return

    idx = _QWEN35_DECODER_DUMP_COUNTERS.get(stage, 0)
    _QWEN35_DECODER_DUMP_COUNTERS[stage] = idx + 1
    if idx < _QWEN35_DECODER_DUMP_SKIP:
        return
    if idx >= _QWEN35_DECODER_DUMP_SKIP + _QWEN35_DECODER_DUMP_LIMIT:
        return

    os.makedirs(_QWEN35_DECODER_DUMP_DIR, exist_ok=True)
    rank = _get_debug_rank()
    for name, tensor in tensors.items():
        if tensor is None:
            continue
        torch.save(
            {
                "stage": stage,
                "layer_type": layer_type,
                "idx": idx,
                "rank": rank,
                "pid": os.getpid(),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "tensor": tensor.detach().cpu(),
            },
            os.path.join(
                _QWEN35_DECODER_DUMP_DIR,
                f"{stage}_idx{idx}_rank{rank}_{name}.pt",
            ),
        )

_QWEN35_LEGACY_IN_PROJ_MAPPING = {
    "in_proj_qkv": ("in_proj_qkvz", (0, 1, 2)),
    "in_proj_z": ("in_proj_qkvz", 3),
    "in_proj_b": ("in_proj_ba", 0),
    "in_proj_a": ("in_proj_ba", 1),
}


def _should_validate_packed_in_proj() -> bool:
    return os.getenv(_VALIDATE_PACKED_IN_PROJ_ENV, "0") == "1"


def qwen35_packed_in_proj_output_sizes(
    key_dim: int,
    value_dim: int,
    num_v_heads: int,
) -> list[int]:
    return [key_dim, key_dim, value_dim, value_dim, num_v_heads, num_v_heads]


def qwen35_packed_in_proj_split_sizes(
    key_dim: int,
    value_dim: int,
    num_v_heads: int,
    tp_size: int,
) -> tuple[int, int, int, int]:
    return (
        (key_dim * 2 + value_dim) // tp_size,
        value_dim // tp_size,
        num_v_heads // tp_size,
        num_v_heads // tp_size,
    )


def split_qwen35_packed_in_proj_output(
    projected_states: torch.Tensor,
    key_dim: int,
    value_dim: int,
    num_v_heads: int,
    tp_size: int,
    head_v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mixed_qkv_size, z_size, b_size, a_size = qwen35_packed_in_proj_split_sizes(
        key_dim,
        value_dim,
        num_v_heads,
        tp_size,
    )
    mixed_qkv, z, b, a = projected_states.split(
        [mixed_qkv_size, z_size, b_size, a_size],
        dim=-1,
    )
    z = z.reshape(z.size(0), -1, head_v_dim)
    return mixed_qkv, z, b.contiguous(), a.contiguous()


@torch.compiler.disable
def _validate_qwen35_packed_in_proj(
    model: Qwen3_5GatedDeltaNet,
    hidden_states: torch.Tensor,
    mixed_qkv: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
) -> None:
    if not _should_validate_packed_in_proj():
        return
    if not hasattr(model, "in_proj_qkvz") or not hasattr(model, "in_proj_ba"):
        return

    mixed_qkvz_ref, _ = model.in_proj_qkvz(hidden_states)
    mixed_qkv_ref, z_ref = mixed_qkvz_ref.split(
        [
            (model.key_dim * 2 + model.value_dim) // model.tp_size,
            model.value_dim // model.tp_size,
        ],
        dim=-1,
    )
    z_ref = z_ref.reshape(z_ref.size(0), -1, model.head_v_dim)
    ba_ref, _ = model.in_proj_ba(hidden_states)
    b_ref, a_ref = ba_ref.chunk(2, dim=-1)
    b_ref = b_ref.contiguous()
    a_ref = a_ref.contiguous()

    torch.testing.assert_close(mixed_qkv, mixed_qkv_ref, rtol=0, atol=0)
    torch.testing.assert_close(z, z_ref, rtol=0, atol=0)
    torch.testing.assert_close(b, b_ref, rtol=0, atol=0)
    torch.testing.assert_close(a, a_ref, rtol=0, atol=0)


def _load_qwen35_legacy_in_proj_for_validation(
    params_dict: dict[str, torch.nn.Parameter],
    packed_name: str,
    weight_name: str,
    loaded_weight: torch.Tensor,
) -> str | None:
    if not _should_validate_packed_in_proj():
        return None
    if weight_name not in _QWEN35_LEGACY_IN_PROJ_MAPPING:
        return None

    legacy_param_name, legacy_shard_id = _QWEN35_LEGACY_IN_PROJ_MAPPING[weight_name]
    legacy_name = packed_name.replace("in_proj", legacy_param_name)
    legacy_param = params_dict.get(legacy_name)
    if legacy_param is None:
        return None
    legacy_weight_loader = getattr(
        legacy_param,
        "weight_loader",
        default_weight_loader,
    )
    legacy_weight_loader(legacy_param, loaded_weight, legacy_shard_id)
    return legacy_name


def _patched_qwen3_5_gated_delta_net_init(self, *args, **kwargs) -> None:
    _ORIGINAL_QWEN3_5_GATED_DELTA_NET_INIT(self, *args, **kwargs)

    self.in_proj = AscendMergedColumnParallelLinear(
        input_size=self.hidden_size,
        output_sizes=qwen35_packed_in_proj_output_sizes(
            self.key_dim,
            self.value_dim,
            self.num_v_heads,
        ),
        bias=False,
        quant_config=self.quant_config,
        prefix=f"{self.prefix}.in_proj",
    )
    self.in_proj_split_sizes = qwen35_packed_in_proj_split_sizes(
        self.key_dim,
        self.value_dim,
        self.num_v_heads,
        self.tp_size,
    )

    if not _should_validate_packed_in_proj():
        delattr(self, "in_proj_qkvz")
        delattr(self, "in_proj_ba")


def _patched_qwen3_5_model_load_weights(
    self,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    params_dict = dict(self.named_parameters())
    if not any(name.endswith("in_proj.weight") for name in params_dict):
        return _ORIGINAL_QWEN3_5_MODEL_LOAD_WEIGHTS(self, weights)

    loaded_params: set[str] = set()
    expert_params_mapping = self.get_expert_mapping()
    is_fused_expert = False
    fused_expert_params_mapping = [
        ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
        ("experts.w2_weight", "experts.down_proj", 0, "w2"),
    ]
    num_experts = self.config.num_experts if hasattr(self.config, "num_experts") else 0

    for name, loaded_weight in weights:
        if "rotary_emb.inv_freq" in name:
            continue

        if name.startswith("mtp."):
            continue

        if name.endswith("scale"):
            name = maybe_remap_kv_scale_name(name, params_dict)
            if name is None:
                continue

        for param_name, weight_name, shard_id in _QWEN35_STACKED_PARAMS_MAPPING:
            if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                is_fused_expert = True
                expert_params_mapping = fused_expert_params_mapping

            if weight_name not in name:
                continue

            if "mlp.experts" in name:
                continue

            name_mapped = name.replace(weight_name, param_name)
            if name_mapped.endswith(".bias") and name_mapped not in params_dict:
                continue
            if is_pp_missing_parameter(name_mapped, self):
                continue
            if name_mapped not in params_dict:
                continue

            param = params_dict[name_mapped]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight, shard_id)
            legacy_name = _load_qwen35_legacy_in_proj_for_validation(
                params_dict,
                name_mapped,
                weight_name,
                loaded_weight,
            )
            if legacy_name is not None:
                loaded_params.add(legacy_name)
            name = name_mapped
            break
        else:
            is_expert_weight = False
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue
                is_expert_weight = True
                name_mapped = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(name_mapped, self):
                    continue
                if is_fused_expert:
                    if "experts.gate_up_proj" in name:
                        gate_weight, up_weight = loaded_weight.chunk(2, dim=-2)
                        success_w1 = self.load_fused_expert_weights(
                            name_mapped,
                            params_dict,
                            gate_weight,
                            "w1",
                            num_experts,
                        )
                        success_w3 = self.load_fused_expert_weights(
                            name_mapped,
                            params_dict,
                            up_weight,
                            "w3",
                            num_experts,
                        )
                        success = success_w1 and success_w3
                    else:
                        success = self.load_fused_expert_weights(
                            name_mapped,
                            params_dict,
                            loaded_weight,
                            shard_id,
                            num_experts,
                        )
                    if success:
                        name = name_mapped
                        break
                else:
                    if (
                        name_mapped.endswith(".bias")
                        or name_mapped.endswith("_bias")
                    ) and name_mapped not in params_dict:
                        continue
                    param = params_dict[name_mapped]
                    weight_loader = param.weight_loader
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                if success:
                    name = name_mapped
                    break
            else:
                if is_expert_weight:
                    continue
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    logger.warning_once(
                        f"Parameter {name} not found in params_dict, skip loading"
                    )
                    continue
                param = params_dict[name]
                weight_loader = getattr(
                    param,
                    "weight_loader",
                    default_weight_loader,
                )
                weight_loader(param, loaded_weight)
        loaded_params.add(name)
    return loaded_params


def _has_qwen35_prefill_metadata(model: Qwen3_5Model) -> bool:
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None or not isinstance(attn_metadata, dict):
        return False
    if get_pcp_group().world_size > 1:
        return False

    for layer in islice(model.layers, model.start_layer, model.end_layer):
        if getattr(layer, "layer_type", None) != "linear_attention":
            continue

        layer_attn_metadata = attn_metadata.get(layer.linear_attn.prefix)
        if (
            isinstance(layer_attn_metadata, GDNAttentionMetadata)
            and layer_attn_metadata.num_prefills > 0
        ):
            return True
    return False


@torch.compiler.disable
def _prepare_qwen35_prefill_precomputed(model: Qwen3_5Model) -> None:
    forward_context = get_forward_context()
    forward_context.qwen35_gdn_prefill_precomputed = {}

    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None or not isinstance(attn_metadata, dict):
        return
    if get_pcp_group().world_size > 1:
        return

    shared_precomputed = {}
    precomputed_by_prefix = {}
    for layer in islice(model.layers, model.start_layer, model.end_layer):
        if getattr(layer, "layer_type", None) != "linear_attention":
            continue

        linear_attn = layer.linear_attn
        layer_attn_metadata = attn_metadata.get(linear_attn.prefix)
        if not isinstance(layer_attn_metadata, GDNAttentionMetadata):
            continue
        if (
            layer_attn_metadata.num_prefills <= 0
            or layer_attn_metadata.non_spec_query_start_loc is None
        ):
            continue

        cu_seqlens = layer_attn_metadata.non_spec_query_start_loc
        num_heads = linear_attn.num_v_heads // linear_attn.tp_size
        cache_key = (cu_seqlens.data_ptr(), num_heads)
        if cache_key not in shared_precomputed:
            shared_precomputed[cache_key] = build_gdn_prefill_precomputed(
                cu_seqlens,
                num_heads,
            )
        precomputed_by_prefix[linear_attn.prefix] = shared_precomputed[cache_key]

    forward_context.qwen35_gdn_prefill_precomputed = precomputed_by_prefix


class AscendQwen3_5Model(Qwen3_5Model):
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        forward_context = get_forward_context()
        forward_context.qwen35_gdn_prefill_precomputed = {}
        # if _has_qwen35_prefill_metadata(self):
        #     _prepare_qwen35_prefill_precomputed(self)
        return _ORIGINAL_QWEN3_5_MODEL_FORWARD(
            self,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )


class AscendQwen3_5GatedDeltaNet(Qwen3_5GatedDeltaNet):
    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor | None:
        if not hasattr(self, "in_proj"):
            return _ORIGINAL_QWEN3_5_GATED_DELTA_NET_FORWARD(
                self,
                hidden_states,
                output,
            )

        projected_states, _ = self.in_proj(hidden_states)
        num_tokens = projected_states.size(0)
        mixed_qkv, z, b, a = split_qwen35_packed_in_proj_output(
            projected_states,
            self.key_dim,
            self.value_dim,
            self.num_v_heads,
            self.tp_size,
            self.head_v_dim,
        )
        if _should_validate_packed_in_proj() and self.tp_size == 1:
            _validate_qwen35_packed_in_proj(self, hidden_states, mixed_qkv, z, b, a)

        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            self.prefix,
        )

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        projected_output, _ = self.out_proj(core_attn_out)
        if output.shape == projected_output.shape:
            output.copy_(projected_output)
            return output
        return projected_output

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            return

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
        self_kv_cache = self.kv_cache[forward_context.virtual_engine]
        conv_state = self_kv_cache[0].transpose(-1, -2)
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens
        prefill_precomputed = None
        if attn_metadata.num_prefills > 0:
            precomputed_by_prefix = getattr(
                forward_context,
                "qwen35_gdn_prefill_precomputed",
                None,
            )
            if isinstance(precomputed_by_prefix, dict):
                prefill_precomputed = precomputed_by_prefix.get(self.prefix)

        if not enable_sp():
            mixed_qkv = mixed_qkv[:num_actual_tokens]
            b = b[:num_actual_tokens]
            a = a[:num_actual_tokens]

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0),
            self.conv1d.weight.size(2),
        )
        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        if spec_sequence_masks is not None:
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][
                    : attn_metadata.num_spec_decodes
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )

        if attn_metadata.num_prefills > 0:
            if mixed_qkv_non_spec is not None:
                conv_weights_t = conv_weights.transpose(0, 1)
                mixed_qkv_non_spec = torch.ops._C_ascend.causal_conv1d_fn(
                    mixed_qkv_non_spec,
                    conv_weights_t,
                    self.conv1d.bias,
                    activation=self.activation,
                    conv_state=self_kv_cache[0],
                    has_initial_state=has_initial_state,
                    non_spec_state_indices_tensor=non_spec_state_indices_tensor,
                    non_spec_query_start_loc=non_spec_query_start_loc,
                    pad_slot_id=PAD_SLOT_ID,
                )
        elif attn_metadata.num_decodes > 0:
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[
                    : attn_metadata.num_actual_tokens
                ],
                validate_data=True,
            )
        else:
            mixed_qkv_non_spec = None
        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
            mixed_qkv_non_spec
        )

        if attn_metadata.num_prefills > 0 or spec_sequence_masks is not None:
            g, beta = fused_gdn_gating_patch(self.A_log, a, b, self.dt_bias)
            if spec_sequence_masks is not None:
                if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                    g_spec = g
                    beta_spec = beta
                    g_non_spec = None
                    beta_non_spec = None
                else:
                    g_spec = g.index_select(1, spec_token_indx)
                    beta_spec = beta.index_select(1, spec_token_indx)
                    g_non_spec = g.index_select(1, non_spec_token_indx)
                    beta_non_spec = beta.index_select(1, non_spec_token_indx)
            else:
                g_spec = None
                beta_spec = None
                g_non_spec = g
                beta_non_spec = beta

            if spec_sequence_masks is not None:
                core_attn_out_spec, last_recurrent_state = (
                    fused_recurrent_gated_delta_rule(
                        q=query_spec,
                        k=key_spec,
                        v=value_spec,
                        g=g_spec,
                        beta=beta_spec,
                        initial_state=ssm_state,
                        inplace_final_state=True,
                        cu_seqlens=spec_query_start_loc[
                            : attn_metadata.num_spec_decodes + 1
                        ],
                        ssm_state_indices=spec_state_indices_tensor,
                        num_accepted_tokens=num_accepted_tokens,
                        use_qk_l2norm_in_kernel=True,
                    )
                )
            else:
                core_attn_out_spec, last_recurrent_state = None, None

            if attn_metadata.num_prefills > 0:
                initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
                initial_state[~has_initial_state, ...] = 0
                core_attn_out_non_spec, last_recurrent_state = chunk_gated_delta_rule(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=initial_state,
                    output_final_state=True,
                    cu_seqlens=non_spec_query_start_loc,
                    head_first=False,
                    use_qk_l2norm_in_kernel=True,
                    prefill_precomputed=prefill_precomputed,
                )
                ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(
                    ssm_state.dtype
                )
            elif attn_metadata.num_decodes > 0:
                core_attn_out_non_spec, last_recurrent_state = (
                    fused_recurrent_gated_delta_rule(
                        q=query_non_spec,
                        k=key_non_spec,
                        v=value_non_spec,
                        g=g_non_spec,
                        beta=beta_non_spec,
                        initial_state=ssm_state,
                        inplace_final_state=True,
                        cu_seqlens=non_spec_query_start_loc[
                            : attn_metadata.num_decodes + 1
                        ],
                        ssm_state_indices=non_spec_state_indices_tensor,
                        use_qk_l2norm_in_kernel=True,
                    )
                )
            else:
                core_attn_out_non_spec, last_recurrent_state = None, None

        elif attn_metadata.num_decodes > 0:
            core_attn_out_non_spec = fused_sigmoid_gating_delta_rule_update(
                A_log=self.A_log.contiguous(),
                dt_bias=self.dt_bias.contiguous(),
                q=query_non_spec.contiguous(),
                k=key_non_spec.contiguous(),
                v=value_non_spec.contiguous(),
                a=a.contiguous(),
                b=b.contiguous(),
                initial_state_source=ssm_state,
                initial_state_indices=non_spec_state_indices_tensor,
                cu_seqlens=non_spec_query_start_loc,
                use_qk_l2norm_in_kernel=True,
                softplus_beta=1.0,
                softplus_threshold=20.0,
            )

        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(
                1,
                non_spec_token_indx,
                core_attn_out_non_spec,
            )
            if not enable_sp():
                core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
            else:
                core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)[
                    :num_actual_tokens
                ]
        elif spec_sequence_masks is not None:
            if not enable_sp():
                core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
            else:
                core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)[
                    :num_actual_tokens
                ]
        else:
            if not enable_sp():
                core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)
            else:
                core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)[
                    :num_actual_tokens
                ]
        maybe_save_kv_layer_to_connector("", [])


class AscendQwen3_5DecoderLayer(Qwen3_5DecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor = None,
        **kwargs: object,
    ):
        _write_alias_debug(
            "decoder_enter",
            layer_type=self.layer_type,
            hidden=_tensor_debug_ptr(hidden_states),
            residual=_tensor_debug_ptr(residual),
        )
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        _write_alias_debug(
            "after_input_layernorm",
            layer_type=self.layer_type,
            hidden=_tensor_debug_ptr(hidden_states),
            residual=_tensor_debug_ptr(residual),
            hidden_residual_same_storage=_same_storage_debug(
                hidden_states,
                residual,
            ),
        )
        _dump_decoder_tensors(
            "after_input_layernorm",
            self.layer_type,
            hidden_states=hidden_states,
            residual=residual,
        )

        self_attention_output = torch.empty_like(hidden_states)
        if self.layer_type == "linear_attention":
            attn_output = self.linear_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
            )
        elif self.layer_type == "full_attention":
            attn_output = self.self_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")
        hidden_states = self_attention_output if attn_output is None else attn_output
        _write_alias_debug(
            "after_attention",
            layer_type=self.layer_type,
            hidden=_tensor_debug_ptr(hidden_states),
            residual=_tensor_debug_ptr(residual),
            attn_output=_tensor_debug_ptr(attn_output),
            self_attention_output=_tensor_debug_ptr(self_attention_output),
            hidden_residual_same_storage=_same_storage_debug(
                hidden_states,
                residual,
            ),
            output_residual_same_storage=_same_storage_debug(
                self_attention_output,
                residual,
            ),
        )
        _dump_decoder_tensors(
            "after_attention",
            self.layer_type,
            hidden_states=hidden_states,
            residual=residual,
            self_attention_output=self_attention_output,
        )

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype) + 1
                )

        _dump_decoder_tensors(
            "before_post_attention_layernorm",
            self.layer_type,
            hidden_states=hidden_states,
            residual=residual,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        _dump_decoder_tensors(
            "after_post_attention_layernorm",
            self.layer_type,
            hidden_states=hidden_states,
            residual=residual,
        )
        hidden_states = self.mlp(hidden_states)
        _dump_decoder_tensors(
            "after_mlp",
            self.layer_type,
            hidden_states=hidden_states,
            residual=residual,
        )

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                    f"shape must be the same {len(hidden_states.shape)}, "
                    f"{len(self.ffn_layer_scale.shape)}"
                )
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype) + 1
                )

        return hidden_states, residual


Qwen3_5ForCausalLMBase.packed_modules_mapping = _QWEN35_PACKED_MODULES_MAPPING
Qwen3_5DecoderLayer.forward = AscendQwen3_5DecoderLayer.forward
Qwen3_5GatedDeltaNet.__init__ = _patched_qwen3_5_gated_delta_net_init
Qwen3_5GatedDeltaNet.forward = AscendQwen3_5GatedDeltaNet.forward
Qwen3_5GatedDeltaNet._forward_core = AscendQwen3_5GatedDeltaNet._forward_core
Qwen3_5Model.forward = AscendQwen3_5Model.forward
Qwen3_5Model.load_weights = _patched_qwen3_5_model_load_weights
