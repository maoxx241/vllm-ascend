# SPDX-License-Identifier: Apache-2.0
"""vLLM text model implementation for Kimi K3."""

from collections.abc import Iterable

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE, fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import KimiGatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
)
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, maybe_remap_kv_scale_name
from vllm.model_executor.models.deepseek_v2 import DeepSeekV2FusedQkvAProjLinear
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid, MixtureOfExperts, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from vllm_ascend.ops.activation import AscendSituAndMul
from vllm_ascend.ops.kimi_kda_state import kimi_kda_state_shape
from vllm_ascend.transformers_utils.configs.kimi_k3 import KimiK3TextConfig


def _situ_params(config: KimiK3TextConfig) -> tuple[float, float | None]:
    return config.activation_situ_beta or 1.0, config.activation_situ_linear_beta


def _routed_latent_quant_config(
    quant_config: QuantizationConfig | None,
) -> QuantizationConfig | None:
    """Quantize latent MoE projections only for native ModelSlim weights."""
    if quant_config is not None and quant_config.get_name() == "ascend":
        return quant_config
    return None


class KimiK3MLP(nn.Module):
    def __init__(
        self,
        config: KimiK3TextConfig,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size if hidden_size is None else hidden_size
        intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if config.hidden_act == "situ":
            beta, linear_beta = _situ_params(config)
            self.act_fn = AscendSituAndMul(beta=beta, linear_beta=linear_beta)
        elif config.hidden_act == "silu":
            self.act_fn = SiluAndMul()
        else:
            raise ValueError(f"Unsupported Kimi K3 activation: {config.hidden_act}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class _KimiRoutedOutputTransform(nn.Module):
    """Non-owning callable used by MoERunner after routed expert combine."""

    def __init__(self, norm: nn.Module | None, up_proj: nn.Module) -> None:
        super().__init__()
        # The owning KimiK3MoE registers these modules under checkpoint names.
        # Avoid registering aliases below experts.runner as well.
        object.__setattr__(self, "_norm", norm)
        object.__setattr__(self, "_up_proj", up_proj)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        norm = object.__getattribute__(self, "_norm")
        up_proj = object.__getattribute__(self, "_up_proj")
        if norm is not None:
            hidden_states = norm(hidden_states)
        return up_proj(hidden_states)[0]


class KimiK3MoE(nn.Module):
    def __init__(
        self,
        config: KimiK3TextConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if config.hidden_act != "situ":
            raise ValueError("Kimi K3 routed experts require the SiTU activation")
        if config.routed_expert_hidden_size is None:
            raise ValueError("Kimi K3 requires routed_expert_hidden_size")

        self.config = config
        self.hidden_size = config.hidden_size
        self.moe_hidden_size = config.routed_expert_hidden_size
        self.num_shared_experts = config.num_shared_experts
        latent_quant_config = _routed_latent_quant_config(quant_config)

        # Routing always uses the original full-width hidden state.
        self.gate = ReplicatedLinear(
            self.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = nn.Parameter(torch.empty(config.num_experts))

        self.routed_expert_down_proj = ReplicatedLinear(
            self.hidden_size,
            self.moe_hidden_size,
            bias=False,
            # These projections are BF16 in the released compressed-tensors
            # checkpoint, but ModelSlim may quantize them independently.
            quant_config=latent_quant_config,
            prefix=f"{prefix}.routed_expert_down_proj",
        )
        self.routed_expert_norm = (
            RMSNorm(self.moe_hidden_size, eps=config.rms_norm_eps) if config.latent_moe_use_norm else None
        )
        self.routed_expert_up_proj = ReplicatedLinear(
            self.moe_hidden_size,
            self.hidden_size,
            bias=False,
            quant_config=latent_quant_config,
            prefix=f"{prefix}.routed_expert_up_proj",
        )
        routed_output_transform = _KimiRoutedOutputTransform(
            self.routed_expert_norm,
            self.routed_expert_up_proj,
        )

        if self.num_shared_experts:
            self.shared_experts = KimiK3MLP(
                config,
                hidden_size=self.hidden_size,
                intermediate_size=config.moe_intermediate_size * self.num_shared_experts,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None

        beta, linear_beta = _situ_params(config)
        self.experts = FusedMoE(
            shared_experts=self.shared_experts,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_token,
            hidden_size=self.moe_hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.moe_renormalize,
            quant_config=quant_config,
            use_grouped_topk=config.use_grouped_topk,
            num_expert_group=config.num_expert_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func=config.moe_router_activation_func,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=config.routed_scaling_factor,
            n_shared_experts=self.num_shared_experts,
            routed_input_transform=self.routed_expert_down_proj,
            routed_output_transform=routed_output_transform,
            activation="situ",
            situ_beta=beta,
            situ_linear_beta=linear_beta,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        router_logits, _ = self.gate(hidden_states)
        output = self.experts(hidden_states=hidden_states, router_logits=router_logits)
        return output.view(num_tokens, hidden_size)


class KimiK3MLAAttention(nn.Module):
    """Q-LoRA MLA with a position-independent q/k slice and output gate."""

    def __init__(
        self,
        config: KimiK3TextConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs
        if not config.mla_use_nope or config.mla_use_rope:
            raise ValueError("Kimi K3 MLA must use the explicit no-RoPE path")
        if not config.mla_use_output_gate:
            raise ValueError("Kimi K3 MLA requires its output gate")

        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        if num_heads % tp_size:
            raise ValueError("num_attention_heads must be divisible by tensor parallel size")
        self.num_local_heads = num_heads // tp_size
        self.scaling = self.qk_head_dim**-0.5

        self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
            hidden_size,
            [q_lora_rank, kv_lora_rank + qk_rope_head_dim],
            quant_config=quant_config,
            prefix=f"{prefix}.fused_qkv_a_proj",
        )
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            q_lora_rank,
            num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.g_proj = ColumnParallelLinear(
            hidden_size,
            num_heads * v_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.g_proj",
        )
        self.o_proj = RowParallelLinear(
            num_heads * v_head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=None,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj,
            kv_a_proj_with_mqa=None,
            q_a_layernorm=self.q_a_layernorm,
            q_b_proj=self.q_b_proj,
            q_proj=None,
            indexer=None,
            is_sparse=False,
            topk_indices_buffer=None,
        )
        # MLAModules is an upstream dataclass without K3 fields.  Dynamic
        # attributes preserve compatibility while the Ascend OOT wrapper
        # consumes the model-specific modules.
        mla_modules.g_proj = self.g_proj
        mla_modules.use_output_gate = True
        mla_modules.use_mla_rope = False
        self.mla_attn = MultiHeadLatentAttentionWrapper(
            hidden_size,
            self.num_local_heads,
            self.scaling,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            q_lora_rank,
            kv_lora_rank,
            mla_modules,
            cache_config,
            quant_config,
            prefix,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        output[:] = self.mla_attn(positions, hidden_states)


def _apply_attention_residual(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    projection: nn.Module,
    norm: RMSNorm,
) -> torch.Tensor:
    """Apply K3's learned normalized mixture over residual block starts."""
    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    values_fp32 = values.float()
    variance = values_fp32.square().mean(-1, keepdim=True)
    normalized = values_fp32 * torch.rsqrt(variance + norm.variance_epsilon)
    score_weight = norm.weight.float() * projection.weight.squeeze(0).float()
    probabilities = (normalized * score_weight).sum(-1).softmax(-1).unsqueeze(1)
    return torch.matmul(probabilities, values_fp32).squeeze(1).to(values.dtype)


class KimiK3DecoderLayer(nn.Module):
    def __init__(self, config: KimiK3TextConfig, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = int(prefix.rsplit(".", 1)[1])
        quant_config = vllm_config.quant_config

        if config.is_kda_layer(self.layer_idx):
            # Instantiate by the upstream registered class name so vLLM's
            # PluggableLayer mechanism selects the Ascend OOT implementation.
            self.self_attn = KimiGatedDeltaNetAttention(
                config,
                vllm_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            self.self_attn = KimiK3MLAAttention(
                config=config,
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                cache_config=vllm_config.cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )

        if (
            config.num_experts is not None
            and self.layer_idx >= config.first_k_dense_replace
            and self.layer_idx % config.moe_layer_freq == 0
        ):
            # Keep the registered module name identical to the checkpoint.
            # A local ``self.mlp`` alias would move every MoE parameter under
            # ``layers.N.mlp`` and make AutoWeightsLoader miss
            # ``layers.N.block_sparse_moe`` weights.
            self.block_sparse_moe = KimiK3MoE(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.block_sparse_moe",
            )
        else:
            self.mlp = KimiK3MLP(config, quant_config=quant_config, prefix=f"{prefix}.mlp")

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_res_block_size = config.attn_res_block_size
        if self.attn_res_block_size is None:
            raise ValueError("Kimi K3 requires attn_res_block_size")
        self.self_attention_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attention_res_proj = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.self_attention_res_proj",
        )
        self.mlp_res_proj = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.mlp_res_proj",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_sum: torch.Tensor | None = hidden_states
        if block_residual.shape[1] > 0:
            hidden_states = _apply_attention_residual(
                prefix_sum,
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
            )

        if self.layer_idx % self.attn_res_block_size == 0:
            block_residual = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
            prefix_sum = None

        hidden_states = self.input_layernorm(hidden_states)
        attention_output = torch.empty_like(hidden_states)
        self.self_attn(positions=positions, hidden_states=hidden_states, output=attention_output)
        prefix_sum = attention_output if prefix_sum is None else prefix_sum + attention_output

        hidden_states = _apply_attention_residual(
            prefix_sum,
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        if hasattr(self, "block_sparse_moe"):
            hidden_states = self.block_sparse_moe(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        return prefix_sum + hidden_states, block_residual


@support_torch_compile
class KimiK3TextModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: KimiK3TextConfig = vllm_config.model_config.hf_text_config
        self.config = config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def get_layer(prefix: str):
            return KimiK3DecoderLayer(config, vllm_config, prefix)

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.output_attn_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.output_attn_res_proj = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.output_attn_res_proj",
            )
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.output_attn_res_norm = PPMissingLayer()
            self.output_attn_res_proj = PPMissingLayer()
            self.norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def initial_block_count(self) -> int:
        block_size = self.config.attn_res_block_size
        return sum(layer_idx % block_size == 0 for layer_idx in range(self.start_layer))

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        del kwargs
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            block_residual = hidden_states.new_zeros((hidden_states.shape[0], 0, hidden_states.shape[-1]))
        else:
            if intermediate_tensors is None:
                raise ValueError("intermediate_tensors are required on non-first PP ranks")
            hidden_states = intermediate_tensors["hidden_states"]
            block_residual = intermediate_tensors["block_residual"]

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, block_residual = layer(positions, hidden_states, block_residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "block_residual": block_residual})

        hidden_states = _apply_attention_residual(
            hidden_states,
            block_residual,
            self.output_attn_res_proj,
            self.output_attn_res_norm,
        )
        return self.norm(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            (".fused_qkv_a_proj", ".q_a_proj", 0),
            (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
        ]
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_experts,
        )
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for args in weights:
            name, loaded_weight = args[:2]
            loader_kwargs = args[2] if len(args) > 2 else {}
            if "rotary_emb" in name:
                continue
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if ".experts." in name and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    break
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name, self):
                        break
                    param = params_dict[name]
                    param.weight_loader(
                        param,
                        loaded_weight,
                        name,
                        expert_id=expert_id,
                        shard_id=shard_id,
                    )
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None or is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight, **loader_kwargs)
            loaded_params.add(name)
        return loaded_params


class AscendKimiK3ForCausalLM(nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid):
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.config: KimiK3TextConfig = self.model_config.hf_text_config
        self.quant_config = vllm_config.quant_config
        self.model = KimiK3TextModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size,
            scale=getattr(self.config, "logit_scale", 1.0),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros((batch_size, self.config.hidden_size), dtype=dtype, device=device),
                "block_residual": torch.zeros(
                    (batch_size, self.model.initial_block_count(), self.config.hidden_size),
                    dtype=dtype,
                    device=device,
                ),
            }
        )

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        parallel_config = vllm_config.parallel_config
        config = vllm_config.model_config.hf_text_config
        num_spec = vllm_config.speculative_config.num_speculative_tokens if vllm_config.speculative_config else 0
        return kimi_kda_state_shape(
            parallel_config.tensor_parallel_size,
            config.linear_attn_config["num_heads"],
            config.linear_attn_config["head_dim"],
            config.linear_attn_config["short_conv_kernel_size"],
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)


def get_spec_layer_idx_from_weight_name(config: KimiK3TextConfig, weight_name: str) -> int | None:
    for index in range(config.num_nextn_predict_layers):
        layer_idx = config.num_hidden_layers + index
        if weight_name.startswith(f"model.layers.{layer_idx}."):
            return layer_idx
    return None


__all__ = [
    "AscendKimiK3ForCausalLM",
    "KimiK3DecoderLayer",
    "KimiK3MLAAttention",
    "KimiK3MLP",
    "KimiK3MoE",
    "KimiK3TextModel",
]
