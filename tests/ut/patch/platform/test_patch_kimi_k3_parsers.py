# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser import ParserManager
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

from vllm_ascend.patch.platform.kimi_k3_xtml import KimiK3XTMLParseError
from vllm_ascend.patch.platform.patch_kimi_k3_chat_params import (
    prepare_kimi_k3_chat_template_kwargs,
)
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
    KimiK3Parser,
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
    tools = f"{TOOLS_START}{''.join(calls)}{TOOLS_END}" if calls else ""
    return f"{RESPONSE_START}{response}{RESPONSE_END}{tools}{MESSAGE_END}{END_OF_MSG_TOKEN}"


def _request(**kwargs):
    defaults = {
        "model": "kimi-k3",
        "messages": [{"role": "user", "content": "help"}],
        "tools": _tools(),
        "tool_choice": "auto",
    }
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


def _parser(*, thinking: bool):
    return KimiK3Parser(
        TOKENIZER,
        _tools(),
        chat_template_kwargs={"thinking": thinking},
    )


def test_kimi_k3_parsers_are_registered_and_unified():
    assert ReasoningParserManager.get_reasoning_parser("kimi_k3") is KimiK3ReasoningParser
    assert ToolParserManager.get_tool_parser("kimi_k3") is KimiK3ToolParser
    assert (
        ParserManager.get_parser(
            tool_parser_name="kimi_k3",
            reasoning_parser_name="kimi_k3",
            enable_auto_tools=True,
            model_name="kimi-k3",
        )
        is KimiK3Parser
    )


@pytest.mark.parametrize(
    ("tool_parser_name", "reasoning_parser_name", "enable_auto_tools"),
    [
        ("kimi_k3", None, True),
        (None, "kimi_k3", True),
        ("kimi_k3", "kimi_k3", False),
    ],
)
def test_kimi_k3_rejects_partial_parser_configuration(
    tool_parser_name,
    reasoning_parser_name,
    enable_auto_tools,
):
    with pytest.raises(ValueError, match="requires --enable-auto-tool-choice"):
        ParserManager.get_parser(
            tool_parser_name=tool_parser_name,
            reasoning_parser_name=reasoning_parser_name,
            enable_auto_tools=enable_auto_tools,
            model_name="kimi-k3",
        )


def test_kimi_k3_adjusts_non_chat_requests_without_rejecting_them():
    parser = KimiK3ReasoningParser(
        TOKENIZER,
        chat_template_kwargs={"thinking": True},
    )
    request = SimpleNamespace(
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
    )

    assert parser.adjust_request(request) is request
    assert request.skip_special_tokens is False
    assert request.spaces_between_special_tokens is False


def test_reasoning_compatibility_adapter_streams_content_when_thinking_is_disabled():
    parser = KimiK3ReasoningParser(
        TOKENIZER,
        chat_template_kwargs={"thinking": False},
    )

    delta = parser.extract_reasoning_streaming(
        previous_text="plain ",
        current_text="plain answer",
        delta_text="answer",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
    )

    assert delta is not None
    assert delta.reasoning is None
    assert delta.content == "answer"


def test_reasoning_compatibility_adapter_does_not_repeat_content_after_thinking():
    parser = KimiK3ReasoningParser(
        TOKENIZER,
        chat_template_kwargs={"thinking": True},
    )

    first = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="private" + THINK_END + RESPONSE_START + "first",
        delta_text="private" + THINK_END + RESPONSE_START + "first",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
    )
    second = parser.extract_reasoning_streaming(
        previous_text="private" + THINK_END + RESPONSE_START + "first",
        current_text="private" + THINK_END + RESPONSE_START + "first second",
        delta_text=" second",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
    )

    assert first is not None
    assert first.reasoning == "private"
    assert first.content == RESPONSE_START + "first"
    assert second is not None
    assert second.reasoning is None
    assert second.content == " second"


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
        + _tool_output(
            _call("plan_trip", arguments),
            response="I will check. ",
        )
    )

    reasoning, content, calls = _parser(thinking=True).parse(
        model_output,
        _request(),
        enable_auto_tools=True,
    )

    assert reasoning == "I should use the trip planner."
    assert content == "I will check. "
    assert calls is not None and len(calls) == 1
    assert calls[0].name == "plan_trip"
    assert json.loads(calls[0].arguments) == {
        "city": "北京 & 海淀",
        "days": 3,
        "flexible": False,
        "metadata": {"seat": "window"},
        "stops": ["上海", "东京"],
        "note": None,
    }


def test_multiple_calls_zero_arguments_and_json_block():
    raw_json = '{"city":"New York","days":2}'
    json_block = f'<|open|>json type="object"{SEP_TOKEN}{raw_json}{JSON_END}'
    output = _tool_output(
        _call("plan_trip", json_block),
        _call("get_time", index=2),
    )

    _, content, calls = _parser(thinking=False).parse(
        output,
        _request(reasoning_effort="none"),
        enable_auto_tools=True,
    )

    assert content is None
    assert calls is not None
    assert [call.name for call in calls] == ["plan_trip", "get_time"]
    assert json.loads(calls[0].arguments) == {
        "city": "New York",
        "days": 2,
    }
    assert json.loads(calls[1].arguments) == {}


def test_tool_choice_none_preserves_request_and_only_cleans_envelope():
    request = _request(tool_choice="none", reasoning_effort="none")
    parser = _parser(thinking=False)

    reasoning_parser = KimiK3ReasoningParser(
        TOKENIZER,
        chat_template_kwargs={"thinking": False},
    )
    reasoning_parser.adjust_request(request)
    assert request.tool_choice == "none"
    assert request.skip_special_tokens is False
    assert request.spaces_between_special_tokens is False

    reasoning, content, calls = parser.parse(
        _tool_output(
            _call("get_time"),
            response="No tool is needed.",
        ),
        request,
        enable_auto_tools=True,
    )
    assert reasoning is None
    assert content == "No tool is needed."
    assert calls == []


@pytest.mark.parametrize("tool_choice", ["none", None])
def test_tools_omitted_plain_chat_cleans_envelope_without_fallback(tool_choice):
    request = ChatCompletionRequest(
        model="kimi-k3",
        messages=[{"role": "user", "content": "help"}],
        tool_choice=tool_choice,
        reasoning_effort="none",
    )
    original_tool_choice = request.tool_choice
    prepare_kimi_k3_chat_template_kwargs(request)

    reasoning, content, calls = _parser(thinking=False).parse(
        f"plain answer{RESPONSE_END}{MESSAGE_END}",
        request,
        enable_auto_tools=True,
    )

    assert request.tool_choice == original_tool_choice
    assert reasoning is None
    assert content == "plain answer"
    assert calls == []


def test_auto_content_only_is_a_normal_completion():
    reasoning, content, calls = _parser(thinking=False).parse(
        f"no call needed{RESPONSE_END}{MESSAGE_END}",
        _request(reasoning_effort="none"),
        enable_auto_tools=True,
    )

    assert reasoning is None
    assert content == "no call needed"
    assert calls == []


def test_gpqa_and_mmmu_default_chat_replay_split_reasoning_and_content():
    request = ChatCompletionRequest(
        model="kimi-k3",
        messages=[{"role": "user", "content": "answer the question"}],
        reasoning_effort="high",
    )
    prepare_kimi_k3_chat_template_kwargs(request)

    for reasoning_text, answer in (
        ("GPQA reasoning trace", "Answer: D"),
        ("MMMU-Pro visual reasoning trace", "Answer: H"),
    ):
        generated = reasoning_text + THINK_END + RESPONSE_START + answer + RESPONSE_END + MESSAGE_END
        reasoning, content, calls = _parser(thinking=True).parse(
            generated,
            request,
            enable_auto_tools=True,
        )

        assert reasoning == reasoning_text
        assert content == answer
        assert content
        assert calls == []
        assert "<|" not in reasoning + content


@pytest.mark.parametrize(
    "tool_choice",
    [
        "required",
        {"type": "function", "function": {"name": "get_time"}},
    ],
)
def test_required_and_named_never_return_empty_success(tool_choice):
    request = _request(
        tool_choice=tool_choice,
        reasoning_effort="none",
    )
    with pytest.raises(KimiK3XTMLParseError, match="without a valid tool call"):
        _parser(thinking=False).parse(
            _tool_output(response="I did not call anything."),
            request,
            enable_auto_tools=True,
        )


def test_named_choice_rejects_a_different_function():
    request = _request(
        tool_choice={"type": "function", "function": {"name": "get_time"}},
        reasoning_effort="none",
    )
    with pytest.raises(KimiK3XTMLParseError, match="requires 'get_time'"):
        _parser(thinking=False).parse(
            _tool_output(_call("plan_trip")),
            request,
            enable_auto_tools=True,
        )


@pytest.mark.parametrize(
    ("tool_choice", "tool_name"),
    [
        ("required", "plan_trip"),
        ({"type": "function", "function": {"name": "get_time"}}, "get_time"),
    ],
)
def test_required_and_named_success_are_stream_full_equivalent(
    tool_choice,
    tool_name,
):
    request = _request(
        tool_choice=tool_choice,
        reasoning_effort="none",
    )
    generated = _tool_output(
        _call(tool_name),
        response="this response is intentionally suppressed",
    )

    _, full_content, full_calls = _parser(thinking=False).parse(
        generated,
        request,
        enable_auto_tools=True,
    )

    parser = _parser(thinking=False)
    streamed_content: list[str] = []
    streamed_calls = []
    for start in range(0, len(generated), 2):
        chunk = generated[start : start + 2]
        delta = parser.parse_delta(
            delta_text=chunk,
            delta_token_ids=TOKENIZER.encode(chunk),
            request=request,
            finished=start + 2 >= len(generated),
        )
        if delta is not None:
            if delta.content:
                streamed_content.append(delta.content)
            streamed_calls.extend(delta.tool_calls)

    assert full_content is None
    assert streamed_content == []
    assert full_calls is not None and [call.name for call in full_calls] == [tool_name]
    assert [call.function.name for call in streamed_calls] == [tool_name]
    assert [json.loads(call.function.arguments) for call in streamed_calls] == [{}]


def test_bfcl_case5_exact_native_output_is_extracted():
    tool = {
        "type": "function",
        "function": {
            "name": "solve_quadratic_equation",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                    "c": {"type": "number"},
                },
                "required": ["a", "b", "c"],
            },
        },
    }
    request = _request(
        tools=[tool],
        reasoning_effort="none",
    )
    arguments = "".join(
        [
            _argument("a", "number", "2"),
            _argument("b", "number", "6"),
            _argument("c", "number", "5"),
        ]
    )
    # K3's generation prompt already opens ``response``. The captured BFCL
    # output therefore begins with RESPONSE_END, not RESPONSE_START.
    generated = RESPONSE_END + TOOLS_START + _call("solve_quadratic_equation", arguments) + TOOLS_END + MESSAGE_END

    reasoning, content, calls = _parser(thinking=False).parse(
        generated,
        request,
        enable_auto_tools=True,
    )

    assert reasoning is None
    assert content is None
    assert calls is not None and len(calls) == 1
    assert calls[0].name == "solve_quadratic_equation"
    assert json.loads(calls[0].arguments) == {"a": 2, "b": 6, "c": 5}


@pytest.mark.parametrize(
    ("suffix", "message"),
    [
        (
            lambda: _call(
                "plan_trip",
                _argument("days", "number", "not-a-number"),
            ),
            "Invalid JSON",
        ),
        (lambda: _call("delete_everything"), "Unknown K3 tool"),
        (
            lambda: f'<|open|>call tool="plan_trip" index="1" index="1"{SEP_TOKEN}{CALL_END}',
            "Duplicate XTML attribute",
        ),
        (
            lambda: _call(
                "plan_trip",
                _argument("days", "number", "3") + _argument("days", "number", "4"),
            ),
            "Duplicate K3 argument key",
        ),
        (
            lambda: _call(
                "plan_trip",
                f'<|open|>json type="object"{SEP_TOKEN}{{"days":2}}{JSON_END}' + _argument("city", "string", "Paris"),
            ),
            "mixes json arguments",
        ),
        (
            lambda: _call(
                "plan_trip",
                f'<|open|>json type="object"{SEP_TOKEN}{{"days":2,"days":3}}{JSON_END}',
            ),
            "Duplicate JSON key",
        ),
        (
            lambda: _call(
                "plan_trip",
                f'<|open|>json type="object"{SEP_TOKEN}{{"days":NaN}}{JSON_END}',
            ),
            "Non-finite JSON number",
        ),
        (
            lambda: f'<|open|>call tool="plan_trip" index="01"{SEP_TOKEN}{CALL_END}',
            "canonical positive integer",
        ),
        (
            lambda: _call(
                "plan_trip",
                "stray text" + _argument("days", "number", "3"),
            ),
            "Unexpected text or tag",
        ),
    ],
)
def test_malformed_or_unknown_calls_raise_explicitly(suffix, message):
    with pytest.raises(KimiK3XTMLParseError, match=message):
        _parser(thinking=False).parse(
            _tool_output(suffix(), response="unsafe"),
            _request(reasoning_effort="none"),
            enable_auto_tools=True,
        )


def test_missing_tools_end_is_rejected():
    output = f"{RESPONSE_START}unsafe{RESPONSE_END}{TOOLS_START}{_call('get_time')}"
    with pytest.raises(KimiK3XTMLParseError, match="tools block is not closed"):
        _parser(thinking=False).parse(
            output,
            _request(reasoning_effort="none"),
            enable_auto_tools=True,
        )


def test_complete_response_requires_closing_message_marker():
    output = RESPONSE_START + "answer" + RESPONSE_END
    with pytest.raises(KimiK3XTMLParseError, match="closing message marker"):
        _parser(thinking=False).parse(
            output,
            _request(reasoning_effort="none"),
            enable_auto_tools=True,
        )


@pytest.mark.parametrize(
    ("thinking", "output", "expected_reasoning", "expected_content"),
    [
        (True, "truncated reasoning", "truncated reasoning", None),
        (False, "truncated response", None, "truncated response"),
    ],
)
def test_delimiter_free_prefix_remains_available_for_length_truncation(
    thinking,
    output,
    expected_reasoning,
    expected_content,
):
    reasoning, content, calls = _parser(thinking=thinking).parse(
        output,
        _request(
            tools=None,
            tool_choice="none",
            reasoning_effort="max" if thinking else "none",
        ),
        enable_auto_tools=True,
    )

    assert reasoning == expected_reasoning
    assert content == expected_content
    assert calls == []


def test_duplicate_call_index_and_multiple_tools_blocks_are_rejected():
    duplicate_index = _tool_output(
        _call("plan_trip"),
        _call("get_time", index=1),
    )
    with pytest.raises(KimiK3XTMLParseError, match="unique and sequential"):
        _parser(thinking=False).parse(
            duplicate_index,
            _request(reasoning_effort="none"),
            enable_auto_tools=True,
        )

    multiple_blocks = _tool_output(_call("get_time")) + TOOLS_START + _call("get_time") + TOOLS_END
    with pytest.raises(KimiK3XTMLParseError, match="Unexpected text"):
        _parser(thinking=False).parse(
            multiple_blocks,
            _request(reasoning_effort="none"),
            enable_auto_tools=True,
        )


def test_named_choice_filters_prompt_tools_and_uses_required_instruction():
    request = _request(
        tool_choice={"type": "function", "function": {"name": "get_time"}},
        reasoning_effort="high",
    )
    prepare_kimi_k3_chat_template_kwargs(request)
    params = request.build_chat_params(None, "auto")

    assert params.chat_template_kwargs["thinking"] is True
    assert params.chat_template_kwargs["thinking_effort"] == "high"
    assert params.chat_template_kwargs["tool_choice"] == "required"
    assert [tool["function"]["name"] for tool in params.chat_template_kwargs["tools"]] == ["get_time"]
    assert request.tool_choice.function.name == "get_time"


def test_chat_params_accept_kimi_native_kwargs_and_optional_openai_fields():
    response_format = {"type": "json_object"}
    request = _request(
        chat_template_kwargs={
            "thinking": True,
            "thinking_effort": "max",
            "response_format": response_format,
            "return_tensors": "pt",
        },
        response_format=response_format,
        structured_outputs={"json": {"type": "object"}},
        chat_template="{{ ignored by the K3 renderer }}",
        add_generation_prompt=False,
        continue_final_message=True,
    )

    prepare_kimi_k3_chat_template_kwargs(request)
    params = request.build_chat_params(None, "auto")

    assert params.chat_template_kwargs["thinking"] is True
    assert params.chat_template_kwargs["thinking_effort"] == "max"
    assert params.chat_template_kwargs["response_format"] == response_format
    assert params.chat_template_kwargs["return_tensors"] == "pt"


def test_chat_params_treat_null_tool_choice_with_tools_as_auto():
    request = _request(tool_choice=None)

    prepare_kimi_k3_chat_template_kwargs(request)

    assert request.tool_choice == "auto"
    assert request.chat_template_kwargs["tool_choice"] == "auto"


def test_chat_params_set_canonical_sampling_and_reasoning_controls():
    request = _request(reasoning_effort="minimal")
    prepare_kimi_k3_chat_template_kwargs(request)
    prepare_kimi_k3_chat_template_kwargs(request)  # idempotent

    params = request.build_chat_params(None, "auto")
    assert params.chat_template_kwargs["thinking"] is True
    assert params.chat_template_kwargs["thinking_effort"] == "low"
    assert params.chat_template_kwargs["tool_choice"] == "auto"
    assert request.skip_special_tokens is False
    assert request.spaces_between_special_tokens is False

    no_thinking = _request(reasoning_effort="none")
    prepare_kimi_k3_chat_template_kwargs(no_thinking)
    no_thinking_params = no_thinking.build_chat_params(None, "auto")
    assert no_thinking_params.chat_template_kwargs["thinking"] is False
    assert "thinking_effort" not in no_thinking_params.chat_template_kwargs


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 64, 4096])
def test_streaming_reconstructs_nonstream_for_multiple_calls(chunk_size: int):
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
    full_reasoning, full_content, full_calls = _parser(thinking=True).parse(
        generated,
        request,
        enable_auto_tools=True,
    )

    parser = _parser(thinking=True)
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    names: dict[int, str] = {}
    arguments_by_index: dict[int, str] = {}
    ids: dict[int, str] = {}
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
        for tool_call in delta.tool_calls:
            assert tool_call.function is not None
            ids.setdefault(tool_call.index, tool_call.id)
            assert ids[tool_call.index] == tool_call.id
            if tool_call.function.name:
                names[tool_call.index] = tool_call.function.name
            if tool_call.function.arguments is not None:
                arguments_by_index[tool_call.index] = (
                    arguments_by_index.get(tool_call.index, "") + tool_call.function.arguments
                )

    assert "".join(reasoning_parts) == full_reasoning == "Need two calls."
    assert "".join(content_parts) == full_content == "Checking first. "
    assert full_calls is not None
    assert names == {index: call.name for index, call in enumerate(full_calls)}
    assert [json.loads(arguments_by_index[index]) for index in sorted(arguments_by_index)] == [
        json.loads(call.arguments) for call in full_calls
    ]
    assert "<|" not in "".join(reasoning_parts + content_parts)


def test_streaming_include_reasoning_false_and_choice_state_isolation():
    request = _request(include_reasoning=False)
    generated = "private reasoning" + THINK_END + _tool_output(response="public answer")
    parsers = [_parser(thinking=True), _parser(thinking=True)]
    outputs: list[list[str]] = [[], []]

    for start in range(0, len(generated), 3):
        chunk = generated[start : start + 3]
        for index, parser in enumerate(parsers):
            delta = parser.parse_delta(
                delta_text=chunk,
                delta_token_ids=TOKENIZER.encode(chunk),
                request=request,
                prompt_token_ids=TOKENIZER.encode(THINK_START),
                finished=start + 3 >= len(generated),
            )
            if delta is not None:
                assert delta.reasoning is None
                if delta.content:
                    outputs[index].append(delta.content)

    assert ["".join(parts) for parts in outputs] == [
        "public answer",
        "public answer",
    ]
