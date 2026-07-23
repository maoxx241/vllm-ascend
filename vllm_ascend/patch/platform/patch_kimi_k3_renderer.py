#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
"""Render Kimi K3 chat prompts with its tokenizer-owned Python encoder.

Kimi K3 deliberately does not publish a Jinja chat template. Its trusted
remote ``TikTokenTokenizer.apply_chat_template`` implements the XTML protocol,
including typed tool calls, reasoning controls, and multimodal placeholders.
The regular HF renderer rejects tokenizers without a Jinja template, so K3
uses a dedicated renderer while continuing to load the tokenizer through the
standard HF ``auto`` tokenizer mode.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

from vllm.config import ModelConfig, VllmConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.renderers import registry as renderer_registry
from vllm.renderers.base import BaseRenderer
from vllm.renderers.inputs import DictPrompt
from vllm.renderers.inputs.preprocess import parse_dec_only_prompt
from vllm.renderers.params import ChatParams
from vllm.tokenizers.hf import HfTokenizer
from vllm.utils.async_utils import make_async

KIMI_K3_MODEL_TYPE = "kimi_k3"
KIMI_K3_RENDERER_MODE = "kimi_k3"
_ORIGINAL_TOKENIZER_ARGS_ATTR = "_ascend_original_kimi_k3_tokenizer_args_from_config"


def is_kimi_k3_model_config(model_config: ModelConfig) -> bool:
    hf_config = getattr(model_config, "hf_config", None)
    return getattr(hf_config, "model_type", None) == KIMI_K3_MODEL_TYPE


def _normalize_developer_messages(
    conversation: list[ConversationMessage],
) -> list[ConversationMessage]:
    """Match the HF renderer's developer-to-system compatibility behavior."""

    if not any(message["role"] == "developer" for message in conversation):
        return conversation

    converted: list[ConversationMessage] = []
    for message in conversation:
        if message["role"] == "developer":
            converted_message = dict(message)
            converted_message["role"] = "system"
            converted_message.pop("tools", None)
            converted.append(converted_message)  # type: ignore[arg-type]
        else:
            converted.append(message)

    system_contents: list[str] = []
    non_system: list[ConversationMessage] = []
    needs_consolidation = False
    for index, message in enumerate(converted):
        if message["role"] != "system":
            non_system.append(message)
            continue

        if index > 0 or system_contents:
            needs_consolidation = True
        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    parts.append(part["text"])
                elif isinstance(part, str):
                    parts.append(part)
            content = "\n".join(parts)
        if content:
            system_contents.append(content)

    if not needs_consolidation:
        return converted

    merged_system: ConversationMessage = {
        "role": "system",
        "content": "\n\n".join(system_contents),
    }
    return [merged_system, *non_system]


class KimiK3Renderer(BaseRenderer[HfTokenizer]):
    """Renderer that delegates the complete chat protocol to K3's tokenizer."""

    def __init__(
        self,
        config: VllmConfig,
        tokenizer: HfTokenizer | None,
    ) -> None:
        super().__init__(config, tokenizer)
        self._apply_chat_template_async = make_async(
            self._apply_chat_template,
            executor=self._executor,
        )

    def _apply_chat_template(
        self,
        conversation: list[ConversationMessage],
        **kwargs: Any,
    ) -> str | list[int]:
        # K3's Python encoder is the source of truth. In particular, do not let
        # an optional server/request Jinja value replace or filter its kwargs.
        kwargs.pop("chat_template", None)
        return self.get_tokenizer().apply_chat_template(
            conversation=conversation,
            **kwargs,
        )

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )
        conversation = _normalize_developer_messages(conversation)

        prompt_raw = self._apply_chat_template(
            conversation,
            **params.get_apply_chat_template_kwargs(),
        )
        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )
        conversation = _normalize_developer_messages(conversation)

        prompt_raw = await self._apply_chat_template_async(
            conversation,
            **params.get_apply_chat_template_kwargs(),
        )
        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt


if not hasattr(renderer_registry, _ORIGINAL_TOKENIZER_ARGS_ATTR):
    setattr(
        renderer_registry,
        _ORIGINAL_TOKENIZER_ARGS_ATTR,
        renderer_registry.tokenizer_args_from_config,
    )


@wraps(getattr(renderer_registry, _ORIGINAL_TOKENIZER_ARGS_ATTR))
def _tokenizer_args_with_kimi_k3_renderer(model_config: ModelConfig, **kwargs):
    original = getattr(renderer_registry, _ORIGINAL_TOKENIZER_ARGS_ATTR)
    tokenizer_args = original(model_config, **kwargs)
    renderer_mode, *remaining_args = tokenizer_args

    # Preserve the standard HF tokenizer loader and explicit non-HF modes. K3
    # only needs a different renderer when auto/slow/hf resolves to HF.
    if renderer_mode == "hf" and is_kimi_k3_model_config(model_config):
        renderer_mode = KIMI_K3_RENDERER_MODE

    return renderer_mode, *remaining_args


renderer_registry.tokenizer_args_from_config = _tokenizer_args_with_kimi_k3_renderer

if KIMI_K3_RENDERER_MODE not in renderer_registry.RENDERER_REGISTRY.renderers:
    renderer_registry.RENDERER_REGISTRY.register(
        KIMI_K3_RENDERER_MODE,
        __name__,
        "KimiK3Renderer",
    )
