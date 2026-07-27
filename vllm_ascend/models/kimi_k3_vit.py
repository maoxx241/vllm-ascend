# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 MoonViT tower and PatchMergerV2 projector."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.kimi_k25_vit import (
    Learnable2DInterpPosEmbDivided_fixed,
    Rope2DPosEmbRepeated,
    apply_rope,
    tpool_patch_merger,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.models.vision import is_vit_use_data_parallel, run_dp_sharded_mrope_vision_model

from vllm_ascend.transformers_utils.configs.kimi_k3 import KimiK3VisionConfig


@dataclass(frozen=True, slots=True)
class KimiK3VisionAttentionMetadata:
    rope_freqs_cis: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: torch.Tensor
    sequence_lengths: torch.Tensor | None


class KimiK3VisionPatchEmbed(nn.Module):
    def __init__(self, config: KimiK3VisionConfig) -> None:
        super().__init__()
        if config.pos_emb_type != "divided_fixed":
            raise ValueError(f"Unsupported Kimi K3 position embedding: {config.pos_emb_type}")
        configured_patch_size: int | Sequence[int] = config.patch_size
        if isinstance(configured_patch_size, int):
            self.patch_size = (configured_patch_size, configured_patch_size)
        elif isinstance(configured_patch_size, Sequence) and len(configured_patch_size) == 2:
            self.patch_size = (configured_patch_size[0], configured_patch_size[1])
        else:
            raise ValueError(f"Invalid Kimi K3 patch size: {configured_patch_size}")
        self.proj = nn.Conv2d(
            3,
            config.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=config.patch_embed_proj_bias,
        )
        self.pos_emb = Learnable2DInterpPosEmbDivided_fixed(
            height=config.init_pos_emb_height,
            width=config.init_pos_emb_width,
            num_frames=config.init_pos_emb_time,
            dim=config.hidden_size,
            # The released K3 reference constructor leaves this at the
            # LearnablePosEmbInterp default, which is bicubic. Its config.json
            # contains a bilinear field but the reference model never consumes
            # it, so honoring that field here would change non-64x64 grids.
            interpolation_mode="bicubic",
        )

    def forward(self, pixels: torch.Tensor, grid_thws: torch.Tensor | list[list[int]]) -> torch.Tensor:
        hidden_states = self.proj(pixels).view(pixels.shape[0], -1)
        return self.pos_emb(hidden_states, grid_thws)


class KimiK3VisionMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: nn.Module,
        quant_config: QuantizationConfig | None,
        prefix: str,
        use_data_parallel: bool,
        bias: bool,
    ) -> None:
        super().__init__()
        self.fc0 = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc0",
            disable_tp=use_data_parallel,
        )
        self.fc1 = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
            disable_tp=use_data_parallel,
        )
        self.activation = activation

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc0(hidden_states)
        hidden_states = self.activation(hidden_states)
        return self.fc1(hidden_states)[0]


class KimiK3VisionEncoderLayer(nn.Module):
    def __init__(
        self,
        config: KimiK3VisionConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        if config.mlp_type != "mlp2":
            raise ValueError(f"Unsupported Kimi K3 vision MLP: {config.mlp_type}")
        self.use_data_parallel = is_vit_use_data_parallel()
        self.hidden_dim = config.hidden_size
        self.qkv_hidden_size = config.qkv_hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.qkv_hidden_size // self.num_heads
        if self.qkv_hidden_size % self.num_heads:
            raise ValueError("K3 qkv_hidden_size must be divisible by vision heads")
        self.tp_size = 1 if self.use_data_parallel else get_tensor_model_parallel_world_size()
        if self.num_heads % self.tp_size:
            raise ValueError(
                "Kimi K3 vision heads cannot be sharded across the configured "
                f"tensor parallel size ({self.num_heads} heads, TP={self.tp_size}). "
                "Use --mm-encoder-tp-mode data."
            )
        self.num_local_heads = divide(self.num_heads, self.tp_size)

        if config.norm_type != "rmsnorm":
            raise ValueError(f"K3 vision requires RMSNorm, got {config.norm_type}")
        # The released K3 implementation intentionally leaves encoder eps at
        # torch.nn.RMSNorm's dtype-dependent default.  Projector RMSNorm below
        # is the only vision norm with an explicit 1e-5 checkpoint contract.
        self.norm0 = nn.RMSNorm(self.hidden_dim)
        self.norm1 = nn.RMSNorm(self.hidden_dim)
        self.mlp = KimiK3VisionMLP(
            self.hidden_dim,
            config.intermediate_size,
            get_act_fn(config.activation_func),
            quant_config,
            f"{prefix}.mlp",
            self.use_data_parallel,
            config.linear_bias,
        )
        self.wqkv = QKVParallelLinear(
            hidden_size=self.hidden_dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_heads,
            bias=config.attn_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.wqkv",
            disable_tp=self.use_data_parallel,
        )
        self.wo = RowParallelLinear(
            self.qkv_hidden_size,
            self.hidden_dim,
            bias=config.attn_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.wo",
            disable_tp=self.use_data_parallel,
        )
        self.attn = MMEncoderAttention(
            num_heads=self.num_local_heads,
            head_size=self.head_dim,
            scale=self.head_dim**-0.5,
            prefix=f"{prefix}.attn",
        )

    def attention(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        qkv = self.wqkv(hidden_states)[0].view(
            num_tokens,
            3,
            self.num_local_heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(dim=1)
        # QKVParallelLinear shards heads, while the shared RoPE table is head
        # independent and therefore needs no TP slicing.
        query, key = apply_rope(query, key, rope_freqs_cis)
        output = self.attn(
            query.unsqueeze(0),
            key.unsqueeze(0).contiguous(),
            value.unsqueeze(0).contiguous(),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        output = output.reshape(num_tokens, self.num_local_heads * self.head_dim)
        return self.wo(output)[0]

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)
        hidden_states = residual + self.attention(
            hidden_states,
            cu_seqlens,
            rope_freqs_cis,
            max_seqlen,
            sequence_lengths,
        )
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        return residual + self.mlp(hidden_states)


class KimiK3VisionEncoder(nn.Module):
    def __init__(
        self,
        config: KimiK3VisionConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.rope_2d = Rope2DPosEmbRepeated(
            config.qkv_hidden_size // config.num_attention_heads,
            512,
            512,
        )
        self.blocks = nn.ModuleList(
            [
                KimiK3VisionEncoderLayer(config, quant_config, f"{prefix}.blocks.{layer_idx}")
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.final_layernorm = nn.RMSNorm(config.hidden_size)

    def prepare_encoder_metadata(
        self,
        grid_thw_list: list[list[int]],
        device: torch.device,
    ) -> KimiK3VisionAttentionMetadata:
        rope_freqs_cis = self.rope_2d.get_freqs_cis(grid_thw_list, device=device)
        grid = np.asarray(grid_thw_list, dtype=np.int32)
        lengths = grid[:, 0] * grid[:, 1] * grid[:, 2]
        cu_seqlens_np = np.concatenate((np.zeros(1, dtype=np.int32), lengths.cumsum(dtype=np.int32)))
        backend = self.blocks[0].attn.attn_backend
        sequence_lengths = MMEncoderAttention.maybe_compute_seq_lens(backend, cu_seqlens_np, device)
        max_seqlen = torch.tensor(
            MMEncoderAttention.compute_max_seqlen(backend, cu_seqlens_np),
            dtype=torch.int32,
        )
        cu_seqlens = MMEncoderAttention.maybe_recompute_cu_seqlens(
            backend,
            cu_seqlens_np,
            self.blocks[0].hidden_dim,
            self.blocks[0].tp_size,
            device,
        )
        if cu_seqlens is None:
            raise RuntimeError("Kimi K3 vision attention backend did not produce cu_seqlens")
        return KimiK3VisionAttentionMetadata(
            rope_freqs_cis=rope_freqs_cis,
            sequence_lengths=sequence_lengths,
            max_seqlen=max_seqlen,
            cu_seqlens=cu_seqlens,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thws: torch.Tensor | list[list[int]],
        encoder_metadata: KimiK3VisionAttentionMetadata | None = None,
    ) -> torch.Tensor:
        grid_thw_list = grid_thws if isinstance(grid_thws, list) else grid_thws.tolist()
        if encoder_metadata is None:
            encoder_metadata = self.prepare_encoder_metadata(grid_thw_list, hidden_states.device)
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                encoder_metadata.cu_seqlens,
                encoder_metadata.rope_freqs_cis,
                encoder_metadata.max_seqlen,
                encoder_metadata.sequence_lengths,
            )
        return self.final_layernorm(hidden_states)


class KimiK3VisionTower(nn.Module):
    def __init__(
        self,
        config: KimiK3VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if config.merge_type != "sd2_tpool":
            raise ValueError(f"Unsupported Kimi K3 merge type: {config.merge_type}")
        self.config = config
        self.patch_size = config.patch_size
        self.merge_kernel_size = config.merge_kernel_size
        self.patch_embed = KimiK3VisionPatchEmbed(config)
        self.encoder = KimiK3VisionEncoder(config, quant_config, maybe_prefix(prefix, "encoder"))

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thws: torch.Tensor | list[list[int]],
        encoder_metadata: KimiK3VisionAttentionMetadata | None = None,
    ) -> list[torch.Tensor]:
        grid_thw_list = grid_thws if isinstance(grid_thws, list) else grid_thws.tolist()
        if encoder_metadata is None:
            encoder_metadata = self.encoder.prepare_encoder_metadata(grid_thw_list, pixel_values.device)
        hidden_states = self.patch_embed(pixel_values, grid_thw_list)
        hidden_states = self.encoder(hidden_states, grid_thw_list, encoder_metadata)
        return tpool_patch_merger(hidden_states, grid_thw_list, self.merge_kernel_size)


class KimiK3MultiModalProjector(nn.Module):
    def __init__(
        self,
        config: KimiK3VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        enable_rotation: bool = False,
    ) -> None:
        super().__init__()
        merge_size = config.merge_kernel_size[0] * config.merge_kernel_size[1]
        self.input_size = config.mm_hidden_size * merge_size
        if config.mm_projector_type != "patchmergerv2":
            raise ValueError(f"Unsupported Kimi K3 projector: {config.mm_projector_type}")
        self.linear_1 = ReplicatedLinear(
            self.input_size,
            self.input_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_1",
        )
        self.linear_2 = ReplicatedLinear(
            self.input_size,
            config.text_hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_2",
        )
        self.act = get_act_fn(config.projector_hidden_act)
        self.post_norm = RMSNorm(config.text_hidden_size, eps=config.projector_ln_eps)
        # ModelSlim rotates K3's FP4 activations before INT4 inference. Text
        # embeddings fold this matrix into their input projection, but the
        # vision path ends in RMSNorm, so the rotation must remain explicit.
        self.rot_proj: ReplicatedLinear | None = None
        if enable_rotation:
            self.rot_proj = ReplicatedLinear(
                config.text_hidden_size,
                config.text_hidden_size,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.rot_proj",
            )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = image_features.reshape(-1, self.input_size)
        hidden_states = self.linear_1(hidden_states)[0]
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)[0]
        hidden_states = self.post_norm(hidden_states)
        if self.rot_proj is not None:
            hidden_states = self.rot_proj(hidden_states)[0]
        return hidden_states


@torch.inference_mode()
def vision_tower_forward(
    vision_tower: KimiK3VisionTower,
    pixel_values: torch.Tensor,
    grid_thws: torch.Tensor,
    mm_projector: KimiK3MultiModalProjector,
    use_data_parallel: bool,
) -> list[torch.Tensor]:
    grid_thw_list = grid_thws.tolist()
    if use_data_parallel:
        tower_outputs = run_dp_sharded_mrope_vision_model(
            vision_model=vision_tower,
            pixel_values=pixel_values,
            grid_thw_list=grid_thw_list,
            rope_type="rope_2d",
        )
    else:
        metadata = vision_tower.encoder.prepare_encoder_metadata(grid_thw_list, pixel_values.device)
        tower_outputs = vision_tower(pixel_values, grid_thw_list, metadata)

    lengths = [item.shape[0] for item in tower_outputs]
    batched = torch.cat(list(tower_outputs), dim=0)
    projected = mm_projector(batched)
    return list(projected.split(lengths, dim=0))


__all__ = [
    "KimiK3MultiModalProjector",
    "KimiK3VisionEncoderLayer",
    "KimiK3VisionTower",
    "vision_tower_forward",
]
