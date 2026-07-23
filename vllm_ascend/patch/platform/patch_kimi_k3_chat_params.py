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
"""Pass OpenAI request controls to Kimi K3's Python chat encoder.

The mapping is activated by the serving instance's model configuration. This
keeps all other models on vLLM's original request path and avoids using a Jinja
template as a model-identification sentinel.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.exceptions import VLLMValidationError

from vllm_ascend.patch.platform.patch_kimi_k3_renderer import (
    KIMI_K3_PROMPT_TOOL_CHOICE_KEY,
    decode_kimi_k3_prompt_tool_choice,
    encode_kimi_k3_prompt_tool_choice,
    is_kimi_k3_model_config,
)

_ORIGINAL_RENDER_CHAT_ATTR = "_ascend_original_kimi_k3_render_chat"
_ORIGINAL_EFFECTIVE_KWARGS_ATTR = "_ascend_original_kimi_k3_effective_chat_template_kwargs"
_PREPARED_ATTR = "_kimi_k3_chat_params_prepared"

_RESERVED_CHAT_TEMPLATE_KWARGS = frozenset(
    {
        "add_generation_prompt",
        "chat_template",
        "continue_final_message",
        "conversation",
        "enable_thinking",
        "image_prompts",
        KIMI_K3_PROMPT_TOOL_CHOICE_KEY,
        "max_length",
        "padding",
        "response_format",
        "response_schema",
        "return_dict",
        "return_tensors",
        "thinking",
        "thinking_effort",
        "tokenize",
        "tool_choice",
        "tools",
        "truncation",
    }
)

_REASONING_EFFORT_MAP = {
    "minimal": "low",
    "low": "low",
    "medium": "high",
    "high": "high",
    "xhigh": "max",
    "max": "max",
}


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=True, exclude_none=True)
    return value


def _tool_name(tool: Any) -> str | None:
    if isinstance(tool, dict):
        function = tool.get("function")
        if isinstance(function, dict):
            return function.get("name")
        return getattr(function, "name", None)
    return getattr(getattr(tool, "function", None), "name", None)


def _named_tool_choice(request: ChatCompletionRequest) -> str | None:
    choice = request.tool_choice
    function = choice.get("function") if isinstance(choice, dict) else getattr(choice, "function", None)
    if isinstance(function, dict):
        return function.get("name")
    return getattr(function, "name", None)


def prepare_kimi_k3_chat_template_kwargs(request: ChatCompletionRequest) -> None:
    """Install typed K3 controls while rejecting conflicting free-form kwargs."""

    if getattr(request, _PREPARED_ATTR, False):
        return

    user_kwargs = request.chat_template_kwargs or {}
    reserved_overrides = sorted(_RESERVED_CHAT_TEMPLATE_KWARGS.intersection(user_kwargs))
    if reserved_overrides:
        raise VLLMValidationError(
            "Kimi K3 chat_template_kwargs cannot override typed protocol fields: " + ", ".join(reserved_overrides),
            parameter="chat_template_kwargs",
        )
    if request.chat_template is not None:
        raise VLLMValidationError(
            "Kimi K3 uses its tokenizer-owned chat encoder and does not accept a request chat_template.",
            parameter="chat_template",
        )
    if not request.add_generation_prompt or request.continue_final_message:
        raise VLLMValidationError(
            "Kimi K3 requires add_generation_prompt=true and does not yet support continue_final_message.",
            parameter="add_generation_prompt",
        )
    if request.response_format is not None:
        raise VLLMValidationError(
            "Kimi K3 does not yet support response_format with its XTML response envelope.",
            parameter="response_format",
        )
    if request.structured_outputs is not None:
        raise VLLMValidationError(
            "Kimi K3 does not yet support structured_outputs with its XTML response envelope.",
            parameter="structured_outputs",
        )
    if request.tools and request.tool_choice is None:
        raise VLLMValidationError(
            "Kimi K3 requires explicit tool_choice='auto' when tools are provided; null is not an implicit auto mode.",
            parameter="tool_choice",
        )
    template_kwargs = dict(user_kwargs)
    request_tools = [_model_dump(tool) for tool in (request.tools or [])]

    template_kwargs["thinking"] = request.reasoning_effort != "none"
    # Always materialize the typed request value so neither CLI defaults nor
    # request-independent chat-template kwargs can inject a different tool set.
    template_kwargs["tools"] = request_tools

    if template_kwargs["thinking"]:
        template_kwargs["thinking_effort"] = _REASONING_EFFORT_MAP.get(
            request.reasoning_effort,
            "max",
        )

    named_tool = _named_tool_choice(request)
    if named_tool:
        matching_tools = [tool for tool in request_tools if _tool_name(tool) == named_tool]
        if not matching_tools:
            raise VLLMValidationError(
                f"Named Kimi K3 tool choice {named_tool!r} is not declared.",
                parameter="tool_choice",
            )
        template_kwargs["tool_choice"] = "required"
        template_kwargs["tools"] = matching_tools
    elif isinstance(request.tool_choice, str):
        template_kwargs["tool_choice"] = request.tool_choice
    else:
        template_kwargs["tool_choice"] = "none"
    template_kwargs[KIMI_K3_PROMPT_TOOL_CHOICE_KEY] = encode_kimi_k3_prompt_tool_choice(template_kwargs["tool_choice"])

    request.chat_template_kwargs = template_kwargs
    request.skip_special_tokens = False
    request.spaces_between_special_tokens = False
    object.__setattr__(request, _PREPARED_ATTR, True)


if not hasattr(OpenAIServingRender, _ORIGINAL_RENDER_CHAT_ATTR):
    setattr(
        OpenAIServingRender,
        _ORIGINAL_RENDER_CHAT_ATTR,
        OpenAIServingRender.render_chat,
    )


@wraps(getattr(OpenAIServingRender, _ORIGINAL_RENDER_CHAT_ATTR))
async def _render_chat_with_kimi_k3_params(
    self: OpenAIServingRender,
    request: ChatCompletionRequest,
    *,
    skip_mm_cache: bool = False,
):
    if is_kimi_k3_model_config(self.model_config):
        if not isinstance(request, ChatCompletionRequest):
            raise VLLMValidationError(
                "Kimi K3 reasoning and tool use are currently supported through /v1/chat/completions only.",
                parameter="request",
            )
        prepare_kimi_k3_chat_template_kwargs(request)

    original = getattr(type(self), _ORIGINAL_RENDER_CHAT_ATTR)
    return await original(self, request, skip_mm_cache=skip_mm_cache)


OpenAIServingRender.render_chat = _render_chat_with_kimi_k3_params


if not hasattr(OpenAIServingChat, _ORIGINAL_EFFECTIVE_KWARGS_ATTR):
    setattr(
        OpenAIServingChat,
        _ORIGINAL_EFFECTIVE_KWARGS_ATTR,
        OpenAIServingChat._effective_chat_template_kwargs,
    )


@wraps(getattr(OpenAIServingChat, _ORIGINAL_EFFECTIVE_KWARGS_ATTR))
def _effective_chat_template_kwargs_with_kimi_k3_params(
    self: OpenAIServingChat,
    request: ChatCompletionRequest,
) -> dict[str, Any]:
    if is_kimi_k3_model_config(self.model_config):
        prepare_kimi_k3_chat_template_kwargs(request)

    original = getattr(type(self), _ORIGINAL_EFFECTIVE_KWARGS_ATTR)
    effective_kwargs = original(self, request)
    if is_kimi_k3_model_config(self.model_config):
        prompt_tool_choice = effective_kwargs.get(KIMI_K3_PROMPT_TOOL_CHOICE_KEY)
        if prompt_tool_choice is not None:
            effective_kwargs["tool_choice"] = decode_kimi_k3_prompt_tool_choice(prompt_tool_choice)
    return effective_kwargs


OpenAIServingChat._effective_chat_template_kwargs = _effective_chat_template_kwargs_with_kimi_k3_params
