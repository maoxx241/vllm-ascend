# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Ascend implementation of Kimi K3's gated delta attention.

K3 owns a different projection topology from Kimi Linear, most notably its
full-rank output gate. Keep that topology explicit here while reusing vLLM's
standard opaque ``kda_attention`` custom-op boundary.
"""

from functools import partial

import torch
from einops import rearrange
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import divide, get_pcp_group, get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)

# Importing this module registers ``torch.ops.vllm.kda_attention`` through
# vLLM's standard direct custom-op registry.
from vllm.model_executor.layers.mamba.gdn import kimi_gdn_linear_attn as _upstream_kimi_gdn_linear_attn  # noqa: F401
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.ops.gdn_attn_builder import (
    AscendGDNAttentionBackend,
    GDNChunkedPrefillMetadata,
)
from vllm_ascend.ops.kimi_kda_state import kimi_kda_state_shape
from vllm_ascend.ops.linear import AscendColumnParallelLinear
from vllm_ascend.ops.parallel_types import AscendLinearParallelMode, AscendTokenLayout
from vllm_ascend.ops.triton.fla.utils import clear_ssm_states
from vllm_ascend.transformers_utils.configs.kimi_k3 import KimiK3TextConfig

_KDA_CHUNK_SIZE = 64


class _KimiK3TPColumnParallelLinear(AscendColumnParallelLinear):
    ascend_parallel_mode = AscendLinearParallelMode.TENSOR_PARALLEL


def _load_kimi_k3_a_log(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    *,
    num_heads: int,
) -> None:
    """Normalize K3's padded A_log checkpoint and then TP-shard heads."""
    if loaded_weight.ndim != 1 or loaded_weight.shape[0] < num_heads:
        raise ValueError(
            "Kimi K3 A_log must use the released one-dimensional checkpoint "
            f"layout with at least {num_heads} heads, got {tuple(loaded_weight.shape)}"
        )
    # Official K3 stores 96 logical heads followed by 32 padding elements in
    # a one-dimensional [128] tensor.
    loaded_weight = loaded_weight[:num_heads].reshape(1, 1, num_heads, 1)

    local_heads = param.shape[2]
    tp_rank = get_tensor_model_parallel_rank()
    start = tp_rank * local_heads
    default_weight_loader(
        param,
        loaded_weight.narrow(2, start, local_heads),
    )


class AscendKimiGatedDeltaNetAttention(GatedDeltaNetAttention):
    """Kimi KDA with Ascend prefill/decode kernels.

    Kimi K3 adds two details that are absent from the upstream base layer:
    a full-rank output gate (``g_proj``) and a bounded sigmoid decay gate.
    """

    def __init__(
        self,
        config: KimiK3TextConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
        *,
        input_layout: AscendTokenLayout,
    ) -> None:
        super().__init__(config, vllm_config, prefix)

        kda_config = config.linear_attn_config
        if kda_config is None:
            raise ValueError("Kimi K3 requires linear_attn_config")
        if not kda_config.get("use_full_rank_gate", False):
            raise ValueError("Kimi K3 KDA requires use_full_rank_gate=true")

        self.head_dim = kda_config["head_dim"]
        self.num_heads = kda_config["num_heads"]
        if self.num_heads % self.tp_size:
            raise ValueError("Kimi K3 KDA num_heads must be divisible by tensor parallel size")
        self.local_num_heads = divide(self.num_heads, self.tp_size)
        projection_size = self.head_dim * self.num_heads
        self.conv_size = kda_config["short_conv_kernel_size"]

        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.v_proj",
        )
        self.f_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.f_a_proj",
        )
        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        self.dt_bias = nn.Parameter(
            torch.empty(
                divide(projection_size, self.tp_size),
                dtype=torch.float32,
            )
        )
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})
        self.b_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.b_proj",
        )

        self.q_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.q_conv1d",
        )
        self.k_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.k_conv1d",
        )
        self.v_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.v_conv1d",
        )
        for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d):
            # Match the checkpoint's conv1d shape while retaining vLLM's
            # ColumnParallelLinear weight loader.
            conv.weight.data = conv.weight.data.unsqueeze(1)

        self.A_log = nn.Parameter(
            torch.empty(
                1,
                1,
                self.local_num_heads,
                1,
                dtype=torch.float32,
            )
        )
        set_weight_attrs(
            self.A_log,
            {
                "weight_loader": partial(
                    _load_kimi_k3_a_log,
                    num_heads=self.num_heads,
                )
            },
        )

        self.g_proj = _KimiK3TPColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.g_proj",
        )
        self.o_norm = FusedRMSNormGated(
            self.head_dim,
            eps=config.rms_norm_eps,
            activation="sigmoid",
        )
        self.o_proj = RowParallelLinear(
            projection_size,
            self.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.o_proj",
        )

        gate_lower_bound = kda_config.get("gate_lower_bound")
        if gate_lower_bound is None:
            raise ValueError("Kimi K3 KDA requires gate_lower_bound")
        self.gate_lower_bound = float(gate_lower_bound)

        # The model owns the token-layout transition and passes this static
        # decision into the attention layer.
        self.input_layout = input_layout

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return AscendGDNAttentionBackend

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return kimi_kda_state_shape(
            self.tp_size,
            self.num_heads,
            self.head_dim,
            self.conv_size,
            self.num_spec,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        del positions
        # KDA metadata and its recurrent state describe the complete sequence.
        # Unlike Qwen GDN, Kimi's independent q/k/v/g projections do not match
        # SequenceColumnParallelOp's prefix whitelist.  Gather the token shard
        # once before all projections instead of gathering for every linear.
        # The multimodal first layer is already full-sized and must not gather.
        hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            hidden_states.contiguous(),
            self.input_layout is AscendTokenLayout.TOKEN_SHARDED,
        )
        num_tokens = hidden_states.size(0)
        q = self.q_proj(hidden_states)[0]
        k = self.k_proj(hidden_states)[0]
        v = self.v_proj(hidden_states)[0]

        beta = self.b_proj(hidden_states)[0].float().sigmoid().unsqueeze(0)
        raw_gate = self.f_b_proj(self.f_a_proj(hidden_states)[0])[0]
        raw_gate = rearrange(raw_gate, "n (h d) -> 1 n h d", d=self.head_dim)

        output_gate = self.g_proj(hidden_states)[0]
        output_gate = rearrange(output_gate, "n (h d) -> n h d", d=self.head_dim)

        core_attn_out = torch.zeros(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.vllm.kda_attention(
            q,
            k,
            v,
            raw_gate,
            beta,
            core_attn_out,
            self.prefix,
        )
        core_attn_out = self.o_norm(core_attn_out, output_gate)
        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        output[:] = self.o_proj(core_attn_out)[0]

    @staticmethod
    def _run_causal_conv1d(
        mixed_qkv: torch.Tensor,
        conv_weights_t: torch.Tensor,
        conv_state: torch.Tensor,
        query_start_loc: torch.Tensor,
        cache_indices: torch.Tensor,
        *,
        run_mode: int,
        initial_state_mode: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = torch.empty_like(mixed_qkv)
        torch.ops._C_ascend.npu_causal_conv1d_custom(
            out,
            mixed_qkv,
            conv_weights_t,
            conv_state=conv_state,
            bias_opt=None,
            query_start_loc_opt=query_start_loc,
            cache_indices_opt=cache_indices,
            initial_state_mode_opt=initial_state_mode,
            num_accepted_tokens_opt=num_accepted_tokens,
            activation_mode=1,
            pad_slot_id=PAD_SLOT_ID,
            run_mode=run_mode,
        )
        return out

    def _conv_weights_t(self, dtype: torch.dtype) -> torch.Tensor:
        weights = []
        for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d):
            weight = conv.weight.view(conv.weight.size(0), conv.weight.size(2)).transpose(0, 1)
            weights.append(weight)
        # The upstream KDA layer stores short-convolution weights in fp32,
        # while the AscendC causal-conv contract requires input/weight/cache
        # dtypes to match.
        return torch.cat(weights, dim=1).to(dtype=dtype).contiguous()

    def _run_recurrent(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        state_indices: torch.Tensor,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = torch.ops._C_ascend.recurrent_kda(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            raw_gate.contiguous(),
            beta.contiguous(),
            recurrent_state,
            cu_seqlens,
            state_indices,
            self.A_log.reshape(-1).contiguous(),
            self.dt_bias.contiguous(),
            num_accepted_tokens=num_accepted_tokens,
            scale=self.head_dim**-0.5,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=False,
            allow_neg_eigval=False,
            safe_gate=True,
            lower_bound=self.gate_lower_bound,
        )
        return out

    def _run_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        prebuilt_metadata: GDNChunkedPrefillMetadata,
    ) -> torch.Tensor:
        if get_pcp_group().world_size > 1:
            raise NotImplementedError("Kimi KDA prefill does not yet support PCP.")

        cu_seqlens_kern = prebuilt_metadata.cu_seqlens_kern
        cu_seqlens = prebuilt_metadata.cu_seqlens_host if cu_seqlens_kern is None else cu_seqlens_kern
        keep = prebuilt_metadata.keep_meta
        if keep is not None:
            state_indices = state_indices[keep]
            has_initial_state = has_initial_state[keep]

        # The recurrent cache uses [H,V,K], while the AscendC prefill operator
        # uses [H,K,V]. Transpose only at that operator boundary.
        initial_state_vk = recurrent_state[state_indices].contiguous()
        clear_ssm_states(initial_state_vk, has_initial_state)

        initial_state_kv = initial_state_vk.transpose(-1, -2).contiguous()

        q = l2norm_fwd(q.contiguous())
        k = l2norm_fwd(k.contiguous())

        gate_cumsum = torch.ops._C_ascend.kda_gate_cumsum(
            raw_gate.contiguous(),
            _KDA_CHUNK_SIZE,
            A_log=self.A_log.reshape(-1).contiguous(),
            dt_bias=self.dt_bias.contiguous(),
            cu_seqlens=cu_seqlens,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=self.gate_lower_bound,
            layout="BSND",
        )

        result = torch.ops._C_ascend.chunk_kda_fwd(
            q,
            k,
            v.contiguous(),
            gate_cumsum,
            beta.contiguous(),
            self.head_dim**-0.5,
            _KDA_CHUNK_SIZE,
            layout="BSND",
            initial_state=initial_state_kv,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            chunk_indices=prebuilt_metadata.chunk_indices_chunk64_host,
            return_intermediate=False,
        )
        recurrent_state[state_indices] = result[1].transpose(-1, -2).contiguous().to(recurrent_state.dtype)
        return result[0]

    def _forward(
        self,
        q_proj_states: torch.Tensor,
        k_proj_states: torch.Tensor,
        v_proj_states: torch.Tensor,
        g1: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw: AttentionMetadata | None = forward_context.attn_metadata
        if attn_metadata_raw is None:
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        num_actual_tokens = attn_metadata.num_actual_tokens
        q_proj_states = q_proj_states[:num_actual_tokens]
        k_proj_states = k_proj_states[:num_actual_tokens]
        v_proj_states = v_proj_states[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]

        conv_state, recurrent_state = self.kv_cache
        mixed_qkv = torch.cat((q_proj_states, k_proj_states, v_proj_states), dim=-1)
        conv_weights_t = self._conv_weights_t(mixed_qkv.dtype)

        spec_masks = attn_metadata.spec_sequence_masks
        spec_token_indices = attn_metadata.spec_token_indx
        non_spec_token_indices = attn_metadata.non_spec_token_indx

        if spec_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_spec = mixed_qkv
                raw_gate_spec = g1
                beta_spec = beta
                mixed_non_spec = raw_gate_non_spec = beta_non_spec = None
            else:
                mixed_spec = mixed_qkv.index_select(0, spec_token_indices)
                raw_gate_spec = g1.index_select(1, spec_token_indices)
                beta_spec = beta.index_select(1, spec_token_indices)
                mixed_non_spec = mixed_qkv.index_select(0, non_spec_token_indices)
                raw_gate_non_spec = g1.index_select(1, non_spec_token_indices)
                beta_non_spec = beta.index_select(1, non_spec_token_indices)
        else:
            mixed_spec = raw_gate_spec = beta_spec = None
            mixed_non_spec = mixed_qkv
            raw_gate_non_spec = g1
            beta_non_spec = beta

        core_spec = None
        if mixed_spec is not None:
            spec_meta = attn_metadata.spec_decode_metadata
            assert spec_meta is not None
            spec_conv_meta = spec_meta.spec_causal_conv1d
            mixed_spec = self._run_causal_conv1d(
                mixed_spec,
                conv_weights_t,
                conv_state,
                spec_conv_meta.query_start_loc,
                spec_conv_meta.cache_indices,
                run_mode=1,
                num_accepted_tokens=spec_conv_meta.num_accepted_tokens,
            )
            q_spec, k_spec, v_spec = mixed_spec.chunk(3, dim=-1)
            q_spec, k_spec, v_spec = (
                rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim) for x in (q_spec, k_spec, v_spec)
            )
            assert raw_gate_spec is not None and beta_spec is not None
            assert attn_metadata.spec_query_start_loc is not None
            assert attn_metadata.spec_state_indices_tensor is not None
            core_spec = self._run_recurrent(
                q_spec,
                k_spec,
                v_spec,
                raw_gate_spec,
                beta_spec,
                recurrent_state,
                attn_metadata.spec_query_start_loc,
                attn_metadata.spec_state_indices_tensor,
                num_accepted_tokens=spec_conv_meta.num_accepted_tokens,
            )

        core_non_spec = None
        if mixed_non_spec is not None and mixed_non_spec.shape[0] > 0:
            if attn_metadata.num_prefills > 0:
                prefill_meta = attn_metadata.non_spec_prefill_metadata
                assert prefill_meta is not None
                mixed_non_spec = self._run_causal_conv1d(
                    mixed_non_spec,
                    conv_weights_t,
                    conv_state,
                    prefill_meta.causal_conv1d.query_start_loc,
                    prefill_meta.causal_conv1d.cache_indices,
                    run_mode=0,
                    initial_state_mode=prefill_meta.causal_conv1d.initial_state_mode,
                )
            elif attn_metadata.num_decodes > 0:
                decode_meta = attn_metadata.non_spec_decode_metadata
                assert decode_meta is not None
                mixed_non_spec = self._run_causal_conv1d(
                    mixed_non_spec,
                    conv_weights_t,
                    conv_state,
                    decode_meta.causal_conv1d.query_start_loc,
                    decode_meta.causal_conv1d.cache_indices,
                    run_mode=1,
                    initial_state_mode=decode_meta.causal_conv1d.initial_state_mode,
                )

            q_non_spec, k_non_spec, v_non_spec = mixed_non_spec.chunk(3, dim=-1)
            q_non_spec, k_non_spec, v_non_spec = (
                rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim) for x in (q_non_spec, k_non_spec, v_non_spec)
            )
            assert raw_gate_non_spec is not None and beta_non_spec is not None

            split_non_spec = spec_masks is None and attn_metadata.num_prefills > 0 and attn_metadata.num_decodes > 0
            num_decode_tokens = attn_metadata.num_decode_tokens
            core_decode = None
            if split_non_spec:
                assert attn_metadata.non_spec_query_start_loc is not None
                assert attn_metadata.non_spec_state_indices_tensor is not None
                core_decode = self._run_recurrent(
                    q_non_spec[:, :num_decode_tokens],
                    k_non_spec[:, :num_decode_tokens],
                    v_non_spec[:, :num_decode_tokens],
                    raw_gate_non_spec[:, :num_decode_tokens],
                    beta_non_spec[:, :num_decode_tokens],
                    recurrent_state,
                    attn_metadata.non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                    attn_metadata.non_spec_state_indices_tensor[: attn_metadata.num_decodes],
                )

            if attn_metadata.num_prefills > 0:
                if split_non_spec:
                    q_non_spec = q_non_spec[:, num_decode_tokens:]
                    k_non_spec = k_non_spec[:, num_decode_tokens:]
                    v_non_spec = v_non_spec[:, num_decode_tokens:]
                    raw_gate_non_spec = raw_gate_non_spec[:, num_decode_tokens:]
                    beta_non_spec = beta_non_spec[:, num_decode_tokens:]

                assert attn_metadata.prefill_state_indices is not None
                assert attn_metadata.prefill_has_initial_state is not None
                prefill_meta = attn_metadata.non_spec_prefill_metadata
                assert prefill_meta is not None
                core_prefill = self._run_prefill(
                    q_non_spec,
                    k_non_spec,
                    v_non_spec,
                    raw_gate_non_spec,
                    beta_non_spec,
                    recurrent_state,
                    attn_metadata.prefill_state_indices,
                    attn_metadata.prefill_has_initial_state,
                    prefill_meta.chunk,
                )
                core_non_spec = (
                    torch.cat((core_decode, core_prefill), dim=1) if core_decode is not None else core_prefill
                )
            elif attn_metadata.num_decodes > 0:
                assert attn_metadata.non_spec_query_start_loc is not None
                assert attn_metadata.non_spec_state_indices_tensor is not None
                core_non_spec = self._run_recurrent(
                    q_non_spec,
                    k_non_spec,
                    v_non_spec,
                    raw_gate_non_spec,
                    beta_non_spec,
                    recurrent_state,
                    attn_metadata.non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                    attn_metadata.non_spec_state_indices_tensor,
                )

        if core_spec is not None and core_non_spec is not None:
            merged = torch.empty(
                (1, num_actual_tokens, self.local_num_heads, self.head_dim),
                dtype=core_non_spec.dtype,
                device=core_non_spec.device,
            )
            merged.index_copy_(1, spec_token_indices, core_spec)
            merged.index_copy_(1, non_spec_token_indices, core_non_spec)
            core_attn_out[:, :num_actual_tokens] = merged
        elif core_spec is not None:
            core_attn_out[:, :num_actual_tokens] = core_spec
        elif core_non_spec is not None:
            core_attn_out[:, :num_actual_tokens] = core_non_spec
