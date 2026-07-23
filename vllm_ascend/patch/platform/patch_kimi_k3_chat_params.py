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

from vllm_ascend.patch.platform.patch_kimi_k3_renderer import is_kimi_k3_model_config

_ORIGINAL_RENDER_CHAT_ATTR = "_ascend_original_kimi_k3_render_chat"
_ORIGINAL_EFFECTIVE_KWARGS_ATTR = "_ascend_original_kimi_k3_effective_chat_template_kwargs"

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
    """Add K3-native controls to one request without overriding user kwargs."""

    user_kwargs = request.chat_template_kwargs or {}
    template_kwargs = dict(user_kwargs)

    if "thinking" not in user_kwargs and request.reasoning_effort is not None:
        template_kwargs["thinking"] = request.reasoning_effort != "none"

    if (
        "thinking_effort" not in user_kwargs
        and template_kwargs.get("thinking", True)
        and request.reasoning_effort in _REASONING_EFFORT_MAP
    ):
        template_kwargs["thinking_effort"] = _REASONING_EFFORT_MAP[request.reasoning_effort]

    if "tool_choice" not in user_kwargs:
        named_tool = _named_tool_choice(request)
        if named_tool:
            matching_tools = [_model_dump(tool) for tool in (request.tools or []) if _tool_name(tool) == named_tool]
            if not matching_tools:
                raise ValueError(f"Named Kimi K3 tool choice {named_tool!r} is not declared.")
            template_kwargs["tool_choice"] = "required"
            template_kwargs["tools"] = matching_tools
        elif isinstance(request.tool_choice, str):
            template_kwargs["tool_choice"] = request.tool_choice

    if "response_format" not in user_kwargs and request.response_format is not None:
        template_kwargs["response_format"] = _model_dump(request.response_format)

    request.chat_template_kwargs = template_kwargs


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
    return original(self, request)


OpenAIServingChat._effective_chat_template_kwargs = _effective_chat_template_kwargs_with_kimi_k3_params
