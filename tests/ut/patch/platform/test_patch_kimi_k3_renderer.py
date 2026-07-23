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
from vllm_ascend.patch.platform.patch_kimi_k3_renderer import KimiK3Renderer


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
        chat_template="{{ should_not_run }}",
        tokenize=True,
        thinking=False,
        tool_choice="none",
    )

    assert prompt == [11, 12]
    assert calls == [
        {
            "conversation": conversation,
            "tokenize": True,
            "thinking": False,
            "tool_choice": "none",
        }
    ]


def test_renderer_normalizes_developer_messages_like_hf_renderer():
    conversation = [
        {"role": "user", "content": "question"},
        {"role": "developer", "content": "developer policy", "tools": []},
        {"role": "system", "content": "system policy"},
    ]

    normalized = renderer_patch._normalize_developer_messages(conversation)

    assert normalized == [
        {
            "role": "system",
            "content": "developer policy\n\nsystem policy",
        },
        {"role": "user", "content": "question"},
    ]
    assert conversation == [
        {"role": "user", "content": "question"},
        {"role": "developer", "content": "developer policy", "tools": []},
        {"role": "system", "content": "system policy"},
    ]


def test_renderer_preserves_multimodal_data(monkeypatch):
    conversation = [{"role": "user", "content": "<image>describe"}]
    mm_data = {"vision_chunk": [{"type": "image", "image": object()}]}
    mm_uuids = {"vision_chunk": ["image-uuid"]}
    calls = []

    class RecordingTokenizer:
        def apply_chat_template(self, **kwargs):
            calls.append(kwargs)
            return [21, 22]

    monkeypatch.setattr(
        renderer_patch,
        "parse_chat_messages",
        lambda *args, **kwargs: (conversation, mm_data, mm_uuids),
    )
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
    assert "chat_template" not in calls[0]


def test_async_renderer_preserves_multimodal_data(monkeypatch):
    conversation = [{"role": "user", "content": "<image>describe"}]
    mm_data = {"vision_chunk": [{"type": "image", "image": object()}]}
    mm_uuids = {"vision_chunk": ["image-uuid"]}
    parse_messages = AsyncMock(return_value=(conversation, mm_data, mm_uuids))
    apply_template = AsyncMock(return_value="rendered")
    monkeypatch.setattr(renderer_patch, "parse_chat_messages_async", parse_messages)
    monkeypatch.setattr(
        renderer_patch,
        "parse_dec_only_prompt",
        lambda prompt: {"prompt": prompt},
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
        "prompt": "rendered",
        "multi_modal_data": mm_data,
        "multi_modal_uuids": mm_uuids,
    }
    apply_template.assert_awaited_once_with(conversation, return_dict=False)


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
