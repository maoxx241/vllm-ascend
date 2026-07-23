# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser import ParserManager
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

from vllm_ascend.patch.platform.patch_kimi_k3_chat_params import prepare_kimi_k3_chat_template_kwargs
from vllm_ascend.patch.platform.patch_kimi_k3_parsers import (
    ARGUMENT_END,
    CALL_END,
    END_OF_MSG_TOKEN,
    JSON_END,
    MESSAGE_END,
    RESPONSE_END,
    RESPONSE_START,
    SEP_TOKEN,
    THINK_END,
    THINK_START,
    TOOLS_END,
    TOOLS_START,
    KimiK3ReasoningParser,
    KimiK3ToolParser,
)


class FakeTokenizer:
    def encode(self, text):
        return [ord(character) for character in text]

    def get_vocab(self):
        return {}


TOKENIZER = FakeTokenizer()


def _tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "plan_trip",
                "description": "Plan a trip.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "number"},
                        "flexible": {"type": "boolean"},
                        "metadata": {"type": "object"},
                        "stops": {"type": "array"},
                        "note": {"type": ["string", "null"]},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


def _argument(key: str, value_type: str, value: str) -> str:
    return f'<|open|>argument key="{key}" type="{value_type}"{SEP_TOKEN}{value}{ARGUMENT_END}'


def _call(name: str, arguments: str = "", index: int = 1) -> str:
    return f'<|open|>call tool="{name}" index="{index}"{SEP_TOKEN}{arguments}{CALL_END}'


def _tool_output(*calls: str, response: str = "") -> str:
    return (
        f"{RESPONSE_START}{response}{RESPONSE_END}"
        f"{TOOLS_START}{''.join(calls)}{TOOLS_END}"
        f"{MESSAGE_END}{END_OF_MSG_TOKEN}"
    )


def _request(**kwargs):
    defaults = {
        "model": "kimi-k3",
        "messages": [{"role": "user", "content": "help"}],
        "tools": _tools(),
        "tool_choice": "auto",
    }
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


def test_kimi_k3_parsers_are_registered():
    assert ReasoningParserManager.get_reasoning_parser("kimi_k3") is KimiK3ReasoningParser
    assert ToolParserManager.get_tool_parser("kimi_k3") is KimiK3ToolParser


def test_non_streaming_reasoning_and_all_xtml_argument_types():
    arguments = "".join(
        [
            _argument("city", "string", "北京 & 海淀"),
            _argument("days", "number", "3"),
            _argument("flexible", "boolean", "false"),
            _argument("metadata", "object", '{"seat":"window"}'),
            _argument("stops", "array", '["上海","东京"]'),
            _argument("note", "null", "null"),
        ]
    )
    model_output = (
        "I should use the trip planner."
        + THINK_END
        + _tool_output(_call("plan_trip", arguments), response="I will check. ")
    )
    request = _request()

    reasoning, content = KimiK3ReasoningParser(TOKENIZER).extract_reasoning(model_output, request)
    parsed = KimiK3ToolParser(TOKENIZER).extract_tool_calls(content or "", request)

    assert reasoning == "I should use the trip planner."
    assert parsed.tools_called
    assert parsed.content == "I will check. "
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].function.name == "plan_trip"
    assert json.loads(parsed.tool_calls[0].function.arguments) == {
        "city": "北京 & 海淀",
        "days": 3,
        "flexible": False,
        "metadata": {"seat": "window"},
        "stops": ["上海", "东京"],
        "note": None,
    }


def test_non_streaming_multiple_calls_zero_arguments_and_json_block():
    raw_json = '{"city":"New York","days":2}'
    json_block = f'<|open|>json type="object"{SEP_TOKEN}{raw_json}{JSON_END}'
    model_output = _tool_output(
        _call("plan_trip", json_block),
        _call("get_time", index=2),
    )

    parsed = KimiK3ToolParser(TOKENIZER).extract_tool_calls(model_output, _request(reasoning_effort="none"))

    assert [call.function.name for call in parsed.tool_calls] == [
        "plan_trip",
        "get_time",
    ]
    assert json.loads(parsed.tool_calls[0].function.arguments) == {
        "city": "New York",
        "days": 2,
    }
    assert json.loads(parsed.tool_calls[1].function.arguments) == {}


def test_non_streaming_json_block_preserves_non_json_argument_string():
    raw_arguments = '{"city":"unfinished"'
    json_block = f'<|open|>json type="object"{SEP_TOKEN}{raw_arguments}{JSON_END}'
    parsed = KimiK3ToolParser(TOKENIZER).extract_tool_calls(_tool_output(_call("plan_trip", json_block)), _request())

    assert parsed.tools_called
    assert parsed.tool_calls[0].function.arguments == raw_arguments


def test_malformed_or_unknown_calls_do_not_become_api_tool_calls():
    malformed = _call("plan_trip", _argument("days", "number", "not-a-number"))
    unknown = _call("delete_everything")
    parsed = KimiK3ToolParser(TOKENIZER).extract_tool_calls(
        _tool_output(malformed, unknown, response="Could not call safely."),
        _request(),
    )

    assert not parsed.tools_called
    assert parsed.tool_calls == []
    assert parsed.content == "Could not call safely."


def test_named_choice_filters_prompt_tools_and_uses_required_xtml_instruction():
    request = _request(
        tool_choice={"type": "function", "function": {"name": "get_time"}},
        reasoning_effort="high",
    )
    prepare_kimi_k3_chat_template_kwargs(request)
    params = request.build_chat_params(None, "auto")

    assert params.chat_template is None
    assert params.chat_template_kwargs["thinking"] is True
    assert params.chat_template_kwargs["thinking_effort"] == "high"
    assert params.chat_template_kwargs["tool_choice"] == "required"
    assert [tool["function"]["name"] for tool in params.chat_template_kwargs["tools"]] == ["get_time"]

    unknown_choice = _request()
    unknown_choice.tool_choice = {
        "type": "function",
        "function": {"name": "missing_tool"},
    }
    with pytest.raises(ValueError, match="is not declared"):
        prepare_kimi_k3_chat_template_kwargs(unknown_choice)


def test_k3_chat_params_map_reasoning_tool_choice_and_response_format():
    request = _request(
        reasoning_effort="minimal",
        response_format={"type": "json_object"},
    )
    prepare_kimi_k3_chat_template_kwargs(request)
    params = request.build_chat_params(None, "auto")

    assert params.chat_template_kwargs["thinking"] is True
    assert params.chat_template_kwargs["thinking_effort"] == "low"
    assert params.chat_template_kwargs["tool_choice"] == "auto"
    assert params.chat_template_kwargs["response_format"] == {"type": "json_object"}

    no_thinking_request = _request(reasoning_effort="none")
    prepare_kimi_k3_chat_template_kwargs(no_thinking_request)
    no_thinking = no_thinking_request.build_chat_params(None, "auto")
    assert no_thinking.chat_template_kwargs["thinking"] is False
    assert "thinking_effort" not in no_thinking.chat_template_kwargs

    KimiK3ReasoningParser(TOKENIZER).adjust_request(request)
    assert request.response_format is None
    assert request._kimi_k3_original_response_format.type == "json_object"


def test_plain_chat_params_are_unchanged_without_k3_serving_hook():
    params = _request(reasoning_effort="low").build_chat_params(None, "auto")
    assert "thinking" not in params.chat_template_kwargs
    assert "thinking_effort" not in params.chat_template_kwargs
    assert "tool_choice" not in params.chat_template_kwargs


def test_tool_choice_none_keeps_semantics_while_stripping_xtml_wrappers():
    request = _request(tool_choice="none", reasoning_effort="none")
    reasoning_parser = KimiK3ReasoningParser(TOKENIZER, chat_template_kwargs={"thinking": False})
    reasoning, content = reasoning_parser.extract_reasoning(_tool_output(response="No tool is needed."), request)
    assert reasoning is None
    assert content is not None
    reasoning_parser.adjust_request(request)

    assert request.tool_choice == "auto"
    assert request.skip_special_tokens is False
    assert request._kimi_k3_original_tool_choice_none is True

    parsed = KimiK3ToolParser(TOKENIZER).extract_tool_calls(
        _tool_output(_call("get_time"), response="No tool is needed."), request
    )
    assert not parsed.tools_called
    assert parsed.content == "No tool is needed."


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 64, 4096])
def test_streaming_reconstructs_reasoning_content_and_multiple_tool_calls(
    chunk_size: int,
):
    parser_cls = ParserManager.get_parser(
        tool_parser_name="kimi_k3",
        reasoning_parser_name="kimi_k3",
        enable_auto_tools=True,
        model_name="kimi-k3",
    )
    assert parser_cls is not None
    parser = parser_cls(
        TOKENIZER,
        _tools(),
        chat_template_kwargs={"thinking": True},
    )
    request = _request()

    arguments = "".join(
        [
            _argument("city", "string", 'New "York"'),
            _argument("days", "number", "3"),
            _argument("flexible", "boolean", "true"),
        ]
    )
    generated = (
        "Need two calls."
        + THINK_END
        + _tool_output(
            _call("plan_trip", arguments),
            _call("get_time", index=2),
            response="Checking first. ",
        )
    )

    reasoning_parts = []
    content_parts = []
    names: dict[int, str] = {}
    argument_parts: dict[int, list[str]] = {}
    for start in range(0, len(generated), chunk_size):
        chunk = generated[start : start + chunk_size]
        delta = parser.parse_delta(
            delta_text=chunk,
            delta_token_ids=TOKENIZER.encode(chunk),
            request=request,
            prompt_token_ids=TOKENIZER.encode(THINK_START),
            finished=start + chunk_size >= len(generated),
        )
        if delta is None:
            continue
        if delta.reasoning:
            reasoning_parts.append(delta.reasoning)
        if delta.content:
            content_parts.append(delta.content)
        for tool_call in delta.tool_calls or []:
            if tool_call.function and tool_call.function.name:
                names[tool_call.index] = tool_call.function.name
            if tool_call.function and tool_call.function.arguments is not None:
                argument_parts.setdefault(tool_call.index, []).append(tool_call.function.arguments)

    assert "".join(reasoning_parts) == "Need two calls."
    assert "".join(content_parts) == "Checking first. "
    assert names == {0: "plan_trip", 1: "get_time"}
    assert json.loads("".join(argument_parts[0])) == {
        "city": 'New "York"',
        "days": 3,
        "flexible": True,
    }
    assert json.loads("".join(argument_parts[1])) == {}

    combined_output = "".join(reasoning_parts + content_parts)
    assert "<|" not in combined_output
