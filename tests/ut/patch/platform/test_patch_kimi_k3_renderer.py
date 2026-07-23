# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.renderers import registry as renderer_registry
from vllm.renderers.params import ChatParams

from vllm_ascend.patch.platform import patch_kimi_k3_chat_params as chat_params_patch
from vllm_ascend.patch.platform import patch_kimi_k3_renderer as renderer_patch
from vllm_ascend.patch.platform.patch_kimi_k3_renderer import (
    KIMI_K3_IMAGE_PROMPT,
    KIMI_K3_PROMPT_TOOL_CHOICE_KEY,
    KimiK3Renderer,
    decode_kimi_k3_prompt_tool_choice,
)


def _model_config(model_type: str):
    return SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type))


def _request(**kwargs):
    defaults = {
        "model": "kimi-k3",
        "messages": [{"role": "user", "content": "help"}],
        "reasoning_effort": "high",
    }
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


def test_kimi_k3_renderer_is_selected_from_model_type(monkeypatch):
    original_attr = renderer_patch._ORIGINAL_TOKENIZER_ARGS_ATTR
    monkeypatch.setattr(
        renderer_registry,
        original_attr,
        lambda model_config, **kwargs: ("hf", model_config, kwargs),
    )

    kimi_args = renderer_registry.tokenizer_args_from_config(_model_config("kimi_k3"))
    other_args = renderer_registry.tokenizer_args_from_config(_model_config("other"))

    assert kimi_args[0] == "kimi_k3"
    assert other_args[0] == "hf"
    assert renderer_registry.RENDERER_REGISTRY.load_renderer_cls("kimi_k3") is KimiK3Renderer


def test_explicit_non_hf_tokenizer_mode_is_not_rewritten(monkeypatch):
    original_attr = renderer_patch._ORIGINAL_TOKENIZER_ARGS_ATTR
    monkeypatch.setattr(
        renderer_registry,
        original_attr,
        lambda model_config, **kwargs: ("deepseek_v4", model_config, kwargs),
    )

    args = renderer_registry.tokenizer_args_from_config(_model_config("kimi_k3"))

    assert args[0] == "deepseek_v4"


def test_renderer_calls_tokenizer_python_encoder_without_jinja():
    calls = []

    class RecordingTokenizer:
        def apply_chat_template(self, **kwargs):
            calls.append(kwargs)
            return [11, 12]

    renderer = object.__new__(KimiK3Renderer)
    renderer.tokenizer = RecordingTokenizer()
    conversation = [{"role": "user", "content": "hello"}]

    prompt = renderer._apply_chat_template(
        conversation,
        add_generation_prompt=False,
        chat_template="{{ should_not_run }}",
        continue_final_message=True,
        conversation=[{"role": "user", "content": "injected"}],
        enable_thinking=False,
        tokenize=False,
        image_prompts=["untrusted"],
        max_length=1,
        padding=True,
        reasoning_effort="none",
        response_format={"type": "json_object"},
        response_schema={"type": "object"},
        return_dict=True,
        return_tensors="pt",
        thinking=False,
        tool_choice="none",
        truncation=True,
    )

    assert prompt == [11, 12]
    assert calls == [
        {
            "conversation": conversation,
            "tokenize": True,
            "add_generation_prompt": True,
            "image_prompts": [],
            "padding": False,
            "truncation": False,
            "return_tensors": None,
            "return_dict": False,
            "thinking": False,
            "tool_choice": "none",
            "response_format": {"type": "json_object"},
            "response_schema": {"type": "object"},
        }
    ]


def test_renderer_converts_developer_role_without_reordering_or_flattening():
    conversation = [
        {"role": "user", "content": "question"},
        {
            "role": "developer",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "developer policy"},
            ],
            "tools": [],
        },
        {"role": "system", "content": "system policy"},
    ]

    normalized = renderer_patch._normalize_developer_messages(conversation)

    assert normalized == [
        {"role": "user", "content": "question"},
        {
            "role": "system",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "developer policy"},
            ],
        },
        {"role": "system", "content": "system policy"},
    ]
    assert conversation == [
        {"role": "user", "content": "question"},
        {
            "role": "developer",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "developer policy"},
            ],
            "tools": [],
        },
        {"role": "system", "content": "system policy"},
    ]


def test_renderer_preserves_multimodal_data(monkeypatch):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "describe"},
            ],
        }
    ]
    mm_data = {"vision_chunk": [{"type": "image", "image": object()}]}
    mm_uuids = {"vision_chunk": ["image-uuid"]}
    calls = []

    class RecordingTokenizer:
        def apply_chat_template(self, **kwargs):
            calls.append(kwargs)
            return [21, 22]

    def parse_messages(*args, **kwargs):
        assert kwargs["content_format"] == "openai"
        return conversation, mm_data, mm_uuids

    monkeypatch.setattr(renderer_patch, "parse_chat_messages", parse_messages)
    monkeypatch.setattr(
        renderer_patch,
        "parse_dec_only_prompt",
        lambda prompt: {"prompt_token_ids": prompt},
    )

    renderer = object.__new__(KimiK3Renderer)
    renderer.model_config = _model_config("kimi_k3")
    renderer.tokenizer = RecordingTokenizer()
    params = ChatParams(
        chat_template=None,
        chat_template_kwargs={"thinking": True, "thinking_effort": "max"},
    )

    rendered_conversation, prompt = renderer.render_messages(
        [{"role": "user", "content": "describe"}],
        params,
    )

    assert rendered_conversation == conversation
    assert prompt == {
        "prompt_token_ids": [21, 22],
        "multi_modal_data": mm_data,
        "multi_modal_uuids": mm_uuids,
    }
    assert calls[0]["conversation"] == conversation
    assert calls[0]["thinking_effort"] == "max"
    assert calls[0]["tokenize"] is True
    assert calls[0]["padding"] is False
    assert calls[0]["truncation"] is False
    assert calls[0]["return_tensors"] is None
    assert calls[0]["return_dict"] is False
    assert calls[0]["image_prompts"] == [KIMI_K3_IMAGE_PROMPT]
    assert "chat_template" not in calls[0]


def test_async_renderer_preserves_multimodal_data(monkeypatch):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "describe"},
            ],
        }
    ]
    mm_data = {"vision_chunk": [{"type": "image", "image": object()}]}
    mm_uuids = {"vision_chunk": ["image-uuid"]}
    parse_messages = AsyncMock(return_value=(conversation, mm_data, mm_uuids))
    apply_template = AsyncMock(return_value=[31, 32])
    monkeypatch.setattr(renderer_patch, "parse_chat_messages_async", parse_messages)
    monkeypatch.setattr(
        renderer_patch,
        "parse_dec_only_prompt",
        lambda prompt: {"prompt_token_ids": prompt},
    )

    renderer = object.__new__(KimiK3Renderer)
    renderer.model_config = _model_config("kimi_k3")
    renderer._apply_chat_template_async = apply_template

    rendered_conversation, prompt = asyncio.run(
        renderer.render_messages_async(
            [{"role": "user", "content": "describe"}],
            ChatParams(),
        )
    )

    assert rendered_conversation == conversation
    assert prompt == {
        "prompt_token_ids": [31, 32],
        "multi_modal_data": mm_data,
        "multi_modal_uuids": mm_uuids,
    }
    apply_template.assert_awaited_once_with(conversation, return_dict=False)
    assert parse_messages.await_args.kwargs["content_format"] == "openai"


def test_openai_chat_kwargs_are_scoped_by_served_model_type():
    kimi_serving = object.__new__(OpenAIServingChat)
    kimi_serving.model_config = _model_config("kimi_k3")
    kimi_serving.chat_template = None
    kimi_serving.chat_template_content_format = "auto"
    kimi_serving.default_chat_template_kwargs = {}

    other_serving = object.__new__(OpenAIServingChat)
    other_serving.model_config = _model_config("other")
    other_serving.chat_template = None
    other_serving.chat_template_content_format = "auto"
    other_serving.default_chat_template_kwargs = {}

    kimi_kwargs = kimi_serving._effective_chat_template_kwargs(_request())
    other_kwargs = other_serving._effective_chat_template_kwargs(_request())

    assert kimi_kwargs["thinking"] is True
    assert kimi_kwargs["thinking_effort"] == "high"
    assert "thinking" not in other_kwargs
    assert "thinking_effort" not in other_kwargs


def test_server_defaults_cannot_override_typed_kimi_k3_tool_controls():
    serving = object.__new__(OpenAIServingChat)
    serving.model_config = _model_config("kimi_k3")
    serving.chat_template = None
    serving.chat_template_content_format = "auto"
    serving.default_chat_template_kwargs = {
        "thinking": True,
        "tool_choice": "required",
        "tools": [
            {
                "type": "function",
                "function": {"name": "injected", "parameters": {"type": "object"}},
            }
        ],
    }
    request = _request(
        reasoning_effort="none",
        tools=None,
        tool_choice="none",
    )

    kwargs = serving._effective_chat_template_kwargs(request)

    assert kwargs["thinking"] is False
    assert kwargs["tool_choice"] == "none"
    assert kwargs["tools"] == []


def test_auto_tool_choice_survives_vllm_default_merging():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "parameters": {"type": "object"},
            },
        }
    ]
    request = _request(
        tools=tools,
        tool_choice="auto",
    )
    chat_params_patch.prepare_kimi_k3_chat_template_kwargs(request)
    params = request.build_chat_params(None, "auto").with_defaults(
        {
            "tool_choice": "required",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "injected",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }
    )
    assert params.chat_template_kwargs["tool_choice"] == "required"
    assert decode_kimi_k3_prompt_tool_choice(params.chat_template_kwargs[KIMI_K3_PROMPT_TOOL_CHOICE_KEY]) == "auto"

    serving = object.__new__(OpenAIServingChat)
    serving.model_config = _model_config("kimi_k3")
    serving.chat_template = None
    serving.chat_template_content_format = "auto"
    serving.default_chat_template_kwargs = {
        "tool_choice": "required",
        "tools": params.chat_template_kwargs["tools"],
    }
    assert serving._effective_chat_template_kwargs(request)["tool_choice"] == ("auto")

    calls = []

    class RecordingTokenizer:
        def apply_chat_template(self, **kwargs):
            calls.append(kwargs)
            return [41, 42]

    renderer = object.__new__(KimiK3Renderer)
    renderer.tokenizer = RecordingTokenizer()
    conversation = [{"role": "user", "content": "what time is it?"}]
    prompt = renderer._apply_chat_template(
        conversation,
        **params.get_apply_chat_template_kwargs(),
    )

    assert prompt == [41, 42]
    assert calls[0]["tool_choice"] == "auto"
    assert [tool["function"]["name"] for tool in calls[0]["tools"]] == ["get_time"]
    assert KIMI_K3_PROMPT_TOOL_CHOICE_KEY not in calls[0]


@pytest.mark.parametrize(
    ("model_type", "expected_thinking"),
    [("kimi_k3", True), ("other", None)],
)
def test_render_server_prepares_only_kimi_k3_requests(
    monkeypatch,
    model_type: str,
    expected_thinking: bool | None,
):
    async def original_render_chat(self, request, *, skip_mm_cache=False):
        del self, skip_mm_cache
        return dict(request.chat_template_kwargs or {})

    monkeypatch.setattr(
        OpenAIServingRender,
        chat_params_patch._ORIGINAL_RENDER_CHAT_ATTR,
        original_render_chat,
    )
    serving = object.__new__(OpenAIServingRender)
    serving.model_config = _model_config(model_type)

    kwargs = asyncio.run(serving.render_chat(_request()))

    assert kwargs.get("thinking") is expected_thinking
    if model_type == "kimi_k3":
        request = _request()
        asyncio.run(serving.render_chat(request))
        assert request.skip_special_tokens is False
        assert request.spaces_between_special_tokens is False


def test_kimi_k3_render_server_delegates_non_chat_requests(monkeypatch):
    async def original_render_chat(self, request, *, skip_mm_cache=False):
        del self, skip_mm_cache
        return request

    monkeypatch.setattr(
        OpenAIServingRender,
        chat_params_patch._ORIGINAL_RENDER_CHAT_ATTR,
        original_render_chat,
    )
    serving = object.__new__(OpenAIServingRender)
    serving.model_config = _model_config("kimi_k3")

    request = SimpleNamespace(chat_template_kwargs=None)
    assert asyncio.run(serving.render_chat(request)) is request
