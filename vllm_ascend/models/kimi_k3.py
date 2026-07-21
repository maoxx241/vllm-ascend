# SPDX-License-Identifier: Apache-2.0
"""Native multimodal Kimi K3 model for vLLM-Ascend."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import torch
from torch import nn
from transformers import BatchFeature
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors import compressed_tensors
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper, init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    NestedTensors,
    VisionChunkImage,
)
from vllm.multimodal.parse import MultiModalDataItems, VisionChunkProcessorItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    InputProcessingContext,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.processor import cached_get_image_processor
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from vllm_ascend.models.kimi_k3_text import AscendKimiK3ForCausalLM
from vllm_ascend.models.kimi_k3_vit import KimiK3MultiModalProjector, KimiK3VisionTower, vision_tower_forward
from vllm_ascend.transformers_utils.configs.kimi_k3 import KimiK3Config
from vllm_ascend.transformers_utils.processors.kimi_k3 import KimiK3Processor


def _move_module_to_device(
    module: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype | None,
) -> nn.Module:
    """Move an eagerly allocated module without materializing meta tensors.

    Some quantization methods deliberately create parameters on the meta
    device.  The model loader records and materializes those parameters after
    model construction, so calling ``Module.to`` here would fail before the
    loader gets that opportunity.  Non-meta modules retain the explicit move
    used by the upstream Kimi multimodal implementation.
    """
    tensors = (*module.parameters(), *module.buffers())
    if any(tensor.is_meta for tensor in tensors):
        return module
    return module.to(device=device, dtype=dtype)


@dataclass
class MaxImageTokenMeta:
    width: int = 3000
    height: int = 3000


class KimiK3MediaPixelInputs(TensorSchema):
    type: Literal["pixel_values"] = "pixel_values"
    pixel_values: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("np", 3, "ps", "ps"),
    ]
    grid_thws: Annotated[torch.Tensor, TensorShape("nm", 3)]


class KimiK3ProcessingInfo(BaseProcessingInfo):
    def __init__(self, ctx: InputProcessingContext) -> None:
        super().__init__(ctx)
        self.hf_config = self.get_hf_config()
        tokenizer = self.get_tokenizer()
        image_processor = cached_get_image_processor(
            self.ctx.model_config.model,
            revision=self.ctx.model_config.revision,
            trust_remote_code=self.ctx.model_config.trust_remote_code,
        )
        configured_id = self.hf_config.media_placeholder_token_id
        tokenizer_id = tokenizer.convert_tokens_to_ids("<|media_pad|>")
        valid_tokenizer_id = isinstance(tokenizer_id, int) and (
            tokenizer.unk_token_id is None or tokenizer_id != tokenizer.unk_token_id
        )
        self.media_token_id = tokenizer_id if valid_tokenizer_id else configured_id
        self.hf_config.media_placeholder_token_id = self.media_token_id
        self.media_token = tokenizer.decode(self.media_token_id)
        self.image_processor = image_processor
        self.hf_processor = KimiK3Processor(image_processor, tokenizer, self.media_token_id)
        self.media_tokens_calculator = image_processor.media_tokens_calculator

    def get_hf_processor(self):
        return self.hf_processor

    def get_hf_config(self) -> KimiK3Config:
        return self.ctx.get_hf_config(KimiK3Config)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"vision_chunk": None}


class KimiK3DummyInputsBuilder(BaseDummyInputsBuilder[KimiK3ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>" * mm_counts.get("vision_chunk", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        del seq_len, mm_options
        count = mm_counts.get("vision_chunk", 0)
        images = self._get_dummy_images(
            height=MaxImageTokenMeta.height,
            width=MaxImageTokenMeta.width,
            num_images=count,
        )
        return {
            "vision_chunk": [VisionChunkImage(type="image", image=image) for image in images],
        }


class KimiK3MultiModalProcessor(BaseMultiModalProcessor[KimiK3ProcessingInfo]):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        del hf_processor_mm_kwargs
        grid_thws = hf_inputs.get("grid_thws", torch.empty((0, 3)))
        grid_sizes = grid_thws.prod(-1)
        return {
            "pixel_values": MultiModalFieldConfig.flat_from_sizes("vision_chunk", grid_sizes),
            "grid_thws": MultiModalFieldConfig.batched("vision_chunk", keep_on_cpu=True),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        del hf_processor_mm_kwargs, out_mm_kwargs
        media_token_id = self.info.media_token_id
        tokenizer = self.info.get_tokenizer()
        target = (
            tokenizer.encode(
                "<|media_begin|>image<|media_content|>",
                add_special_tokens=False,
            )
            + [media_token_id]
            + tokenizer.encode("<|media_end|>", add_special_tokens=False)
        )

        def replacement(item_idx: int) -> PromptUpdateDetails[list[int]]:
            media = mm_items.get_items("vision_chunk", (VisionChunkProcessorItems,))
            item = media.get(item_idx)
            if item["type"] != "image":
                raise ValueError("Kimi K3 currently supports image inputs only")
            image = item["image"]
            if not hasattr(image, "size"):
                raise ValueError("Kimi K3 image processor did not resolve the input to a PIL image")
            width, height = image.size
            full = (
                tokenizer.encode(
                    f"<|media_begin|>image {width}x{height}<|media_content|>",
                    add_special_tokens=False,
                )
                + [media_token_id] * self.info.media_tokens_calculator(item)
                + tokenizer.encode("<|media_end|>", add_special_tokens=False)
            )
            return PromptUpdateDetails.select_token_id(full, media_token_id)

        return [
            PromptReplacement(
                modality="vision_chunk",
                target=target,
                replacement=replacement,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    KimiK3MultiModalProcessor,
    info=KimiK3ProcessingInfo,
    dummy_inputs=KimiK3DummyInputsBuilder,
)
class AscendKimiK3ForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
):
    supports_encoder_tp_data = True
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "mm_projector.proj.0": "mm_projector.linear_1",
            "mm_projector.proj.2": "mm_projector.linear_2",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        del i
        if modality == "image":
            return "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"
        raise ValueError(f"Kimi K3 does not support modality: {modality}")

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        model_config = vllm_config.model_config
        config: KimiK3Config = model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.hidden_size = config.text_config.hidden_size
        self.device = current_platform.current_device()
        self.use_data_parallel = model_config.multimodal_config.mm_encoder_tp_mode == "data"
        vision_quant = self._maybe_ignore_quant_config(self.quant_config)

        with self._mark_tower_model(vllm_config, "vision_chunk"):
            self.vision_tower = KimiK3VisionTower(
                config.vision_config,
                quant_config=vision_quant,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            tower_dtype = model_config.dtype if vision_quant is None else None
            self.vision_tower = _move_module_to_device(
                self.vision_tower,
                device=self.device,
                dtype=tower_dtype,
            )
            self.mm_projector = KimiK3MultiModalProjector(
                config.vision_config,
                quant_config=vision_quant,
                prefix=maybe_prefix(prefix, "mm_projector"),
            )
            self.mm_projector = _move_module_to_device(
                self.mm_projector,
                device=self.device,
                dtype=model_config.dtype,
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["KimiK3ForCausalLM"],
            )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors
        self.media_placeholder = config.media_placeholder_token_id

    @staticmethod
    def _maybe_ignore_quant_config(
        quant_config: QuantizationConfig | None,
    ) -> QuantizationConfig | None:
        if quant_config is not None and (
            isinstance(quant_config, compressed_tensors.CompressedTensorsConfig)
            or quant_config.get_name() == "compressed-tensors"
        ):
            return None
        return quant_config

    def _parse_and_validate_media_input(self, **kwargs: object) -> KimiK3MediaPixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        grid_thws = kwargs.pop("grid_thws", None)
        if pixel_values is None:
            return None
        if isinstance(pixel_values, list):
            pixel_tensors: list[torch.Tensor] = []
            for pixel_value in pixel_values:
                if not isinstance(pixel_value, torch.Tensor):
                    raise TypeError(f"pixel_values entries must be tensors, got {type(pixel_value)}")
                pixel_tensors.append(pixel_value)
            pixel_values = torch.cat(pixel_tensors, dim=0)
        elif not isinstance(pixel_values, torch.Tensor):
            raise TypeError(f"pixel_values must be a tensor or list of tensors, got {type(pixel_values)}")
        if pixel_values.ndim in (3, 5):
            pixel_values = pixel_values.reshape(
                pixel_values.shape[0] * pixel_values.shape[1],
                *pixel_values.shape[2:],
            )
        target_dtype = next(self.vision_tower.parameters()).dtype
        pixel_values = pixel_values.to(target_dtype)
        if not isinstance(grid_thws, torch.Tensor):
            raise TypeError(f"grid_thws must be a tensor, got {type(grid_thws)}")
        grid_thws_tensor: torch.Tensor = grid_thws.reshape(-1, grid_thws.shape[-1])
        if grid_thws_tensor.ndim != 2 or grid_thws_tensor.shape[1] != 3:
            raise ValueError(f"Unexpected Kimi K3 grid_thws shape: {grid_thws_tensor.shape}")
        return KimiK3MediaPixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            grid_thws=grid_thws_tensor,
        )

    def embed_multimodal(self, **kwargs: object) -> NestedTensors | None:
        media_input = self._parse_and_validate_media_input(**kwargs)
        if media_input is None:
            return None
        return vision_tower_forward(
            self.vision_tower,
            media_input["pixel_values"],
            media_input["grid_thws"],
            self.mm_projector,
            self.use_data_parallel,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        del kwargs
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor | None:
        del kwargs
        return self.language_model.compute_logits(hidden_states)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return AscendKimiK3ForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        return AscendKimiK3ForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return AscendKimiK3ForCausalLM.get_mamba_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # ModelSlim emits this projector rotation matrix as a conversion
        # auxiliary.  It is not a parameter of the official Kimi K3 network;
        # keep the exception exact so all real projector weights are checked.
        loader = AutoWeightsLoader(self, skip_prefixes=["mm_projector.rot_proj."])
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


__all__ = ["AscendKimiK3ForConditionalGeneration"]
