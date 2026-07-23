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
"""Kimi K3 XTML reasoning and tool-call parser support.

Kimi K3 renders chat messages with the following structural markers::

    <|open|>think<|sep|>...<|close|>think<|sep|>
    <|open|>response<|sep|>...<|close|>response<|sep|>
    <|open|>tools<|sep|>
      <|open|>call tool="name" index="1"<|sep|>
        <|open|>argument key="arg" type="string"<|sep|>...
        <|close|>argument<|sep|>
      <|close|>call<|sep|>
    <|close|>tools<|sep|>

The K3 tokenizer opens either ``think`` or ``response`` in the generation
prompt. Consequently, generated text normally starts with reasoning/response
content rather than an opening tag. The parsers below deliberately support
both forms and retain incomplete structural markers until a streaming boundary
is unambiguous.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import regex as re
from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.reasoning.abs_reasoning_parsers import (
    ReasoningParser,
    ReasoningParserManager,
)
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
    ToolParserManager,
)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

OPEN_TOKEN = "<|open|>"
CLOSE_TOKEN = "<|close|>"
SEP_TOKEN = "<|sep|>"
END_OF_MSG_TOKEN = "<|end_of_msg|>"

THINK_START = f"{OPEN_TOKEN}think{SEP_TOKEN}"
THINK_END = f"{CLOSE_TOKEN}think{SEP_TOKEN}"
RESPONSE_START = f"{OPEN_TOKEN}response{SEP_TOKEN}"
RESPONSE_END = f"{CLOSE_TOKEN}response{SEP_TOKEN}"
TOOLS_START = f"{OPEN_TOKEN}tools{SEP_TOKEN}"
TOOLS_END = f"{CLOSE_TOKEN}tools{SEP_TOKEN}"
CALL_END = f"{CLOSE_TOKEN}call{SEP_TOKEN}"
ARGUMENT_END = f"{CLOSE_TOKEN}argument{SEP_TOKEN}"
JSON_END = f"{CLOSE_TOKEN}json{SEP_TOKEN}"
MESSAGE_END = f"{CLOSE_TOKEN}message{SEP_TOKEN}"

_ORIGINAL_TOOL_CHOICE_NONE_ATTR = "_kimi_k3_original_tool_choice_none"
_ORIGINAL_RESPONSE_FORMAT_ATTR = "_kimi_k3_original_response_format"
_ATTR_RE = re.compile(r'([A-Za-z_][\w.-]*)="([^"]*)"')
_CALL_START_RE = re.compile(re.escape(f"{OPEN_TOKEN}call") + r"(?P<attrs>[^<]*?)" + re.escape(SEP_TOKEN))
_ARGUMENT_START_RE = re.compile(re.escape(f"{OPEN_TOKEN}argument") + r"(?P<attrs>[^<]*?)" + re.escape(SEP_TOKEN))
_JSON_START_RE = re.compile(re.escape(f"{OPEN_TOKEN}json") + r"(?P<attrs>[^<]*?)" + re.escape(SEP_TOKEN))


def _partial_marker_overlap(text: str, marker: str) -> int:
    """Return the suffix length that may be the beginning of *marker*."""

    max_overlap = min(len(text), len(marker) - 1)
    for overlap in range(max_overlap, 0, -1):
        if text.endswith(marker[:overlap]):
            return overlap
    return 0


def _find_subsequence(values: Sequence[int], pattern: Sequence[int]) -> int:
    if not pattern or len(pattern) > len(values):
        return -1
    for index in range(len(values) - len(pattern), -1, -1):
        if list(values[index : index + len(pattern)]) == list(pattern):
            return index
    return -1


def _encode_marker(tokenizer: TokenizerLike, marker: str) -> list[int]:
    """Encode K3 markers without kwargs that alter its custom tokenizer path."""

    encoded = tokenizer.encode(marker)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, dict):
        encoded = encoded.get("input_ids", [])
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


def _decode_attr_value(value: str) -> str:
    # K3's encoder escapes only these two entities. Decode in this order so
    # ``&amp;quot;`` remains the literal text ``&quot;``.
    return value.replace("&quot;", '"').replace("&amp;", "&")


def _parse_attrs(raw_attrs: str) -> dict[str, str]:
    return {key: _decode_attr_value(value) for key, value in _ATTR_RE.findall(raw_attrs)}


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _request_uses_thinking(request: Any) -> bool:
    template_kwargs = getattr(request, "chat_template_kwargs", None) or {}
    if "thinking" in template_kwargs:
        return bool(template_kwargs["thinking"])
    return getattr(request, "reasoning_effort", None) != "none"


class KimiK3ReasoningParser(ReasoningParser):
    """Split K3's multi-token XTML ``think`` block from its response."""

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        if not self.model_tokenizer:
            raise ValueError("KimiK3ReasoningParser requires a tokenizer.")

        self.think_start_token_ids = _encode_marker(tokenizer, THINK_START)
        self.think_end_token_ids = _encode_marker(tokenizer, THINK_END)
        if not self.think_start_token_ids or not self.think_end_token_ids:
            raise RuntimeError("Unable to encode Kimi K3 reasoning markers.")

        template_kwargs = kwargs.get("chat_template_kwargs") or {}
        self._thinking_enabled = bool(template_kwargs.get("thinking", True))
        self._streamed_reasoning = ""

    @property
    def reasoning_start_str(self) -> str:
        return THINK_START

    @property
    def reasoning_end_str(self) -> str:
        return THINK_END

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        last_start = _find_subsequence(input_ids, self.think_start_token_ids)
        last_end = _find_subsequence(input_ids, self.think_end_token_ids)
        # The generation prompt ends in THINK_START when thinking is active.
        # If there is no unmatched start marker, parsing can begin as content.
        return last_start < 0 or last_end > last_start

    def is_reasoning_end_streaming(
        self,
        input_ids: Sequence[int],
        delta_ids: Iterable[int],
    ) -> bool:
        del delta_ids
        if not self._thinking_enabled:
            return True
        # Generated K3 text does not repeat THINK_START because that marker is
        # already in the prompt. Wait for the multi-token close marker instead
        # of applying the prompt-oriented unmatched-start check above.
        return _find_subsequence(input_ids, self.think_end_token_ids) >= 0

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        end_index = _find_subsequence(input_ids, self.think_end_token_ids)
        if end_index >= 0:
            return input_ids[end_index + len(self.think_end_token_ids) :]
        if _find_subsequence(input_ids, self.think_start_token_ids) < 0:
            return input_ids
        return []

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if not self._thinking_enabled or not _request_uses_thinking(request):
            return None, model_output

        output = model_output
        if output.startswith(THINK_START):
            output = output[len(THINK_START) :]

        if THINK_END in output:
            reasoning, content = output.split(THINK_END, 1)
            return reasoning or None, content or None

        # A well-formed thinking response always closes ``think`` before
        # opening ``response``. If the close marker was dropped, preserve the
        # response instead of misclassifying it as chain-of-thought.
        if RESPONSE_START in output:
            reasoning, content = output.split(RESPONSE_START, 1)
            return reasoning or None, RESPONSE_START + content

        return output or None, None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        del previous_text, delta_text, previous_token_ids, current_token_ids
        del delta_token_ids

        output = current_text
        if output.startswith(THINK_START):
            reasoning_start = len(THINK_START)
        elif THINK_START.startswith(output):
            return None
        else:
            reasoning_start = 0

        end_index = output.find(THINK_END, reasoning_start)
        if end_index >= 0:
            reasoning_snapshot = output[reasoning_start:end_index]
            content = output[end_index + len(THINK_END) :]
        else:
            safe_end = len(output) - _partial_marker_overlap(output, THINK_END)
            reasoning_snapshot = output[reasoning_start:safe_end]
            content = None

        reasoning_delta: str | None = None
        if reasoning_snapshot.startswith(self._streamed_reasoning):
            reasoning_delta = reasoning_snapshot[len(self._streamed_reasoning) :]
        self._streamed_reasoning = reasoning_snapshot

        if not reasoning_delta and not content:
            return None
        return DeltaMessage(reasoning=reasoning_delta or None, content=content or None)

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        if not self._thinking_enabled:
            return 0

        start_index = _find_subsequence(token_ids, self.think_start_token_ids)
        if start_index >= 0:
            content_start = start_index + len(self.think_start_token_ids)
        else:
            content_start = 0

        end_index = _find_subsequence(token_ids, self.think_end_token_ids)
        if end_index < content_start:
            end_index = len(token_ids)
        return max(0, end_index - content_start)

    def adjust_request(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ChatCompletionRequest | ResponsesRequest:
        # K3's tag names (``think``, ``response``, ``call``...) are ordinary
        # text between special control tokens, so the complete decoded markers
        # must reach both parsers.
        if hasattr(request, "skip_special_tokens"):
            request.skip_special_tokens = False

        # vLLM bypasses ToolParser for tool_choice="none". K3 still needs that
        # parser to remove its response wrapper. Rendering has already happened
        # before adjust_request, so this internal switch does not change K3's
        # tool-choice instruction. The marker makes ToolParser suppress any
        # hallucinated calls and retain the original API semantics.
        if isinstance(request, ChatCompletionRequest) and request.tool_choice == "none":
            object.__setattr__(request, _ORIGINAL_TOOL_CHOICE_NONE_ATTR, True)
            request.tool_choice = "auto"

        # The source K3 encoder implements response_format as an XTML system
        # instruction. vLLM's generic JSON grammar would start immediately
        # after THINK_END and reject K3's mandatory RESPONSE_START marker.
        # Rendering has already embedded the schema/instruction, so disable
        # only that generic post-render grammar path.
        if isinstance(request, ChatCompletionRequest) and request.response_format is not None:
            object.__setattr__(
                request,
                _ORIGINAL_RESPONSE_FORMAT_ATTR,
                request.response_format,
            )
            request.response_format = None

        return request


def _tool_name(tool: Any) -> str | None:
    if isinstance(tool, dict):
        function = tool.get("function")
        if isinstance(function, dict):
            return function.get("name")
        return getattr(function, "name", None)
    function = getattr(tool, "function", None)
    return getattr(function, "name", None)


def _named_tool_choice(request: Any) -> str | None:
    choice = getattr(request, "tool_choice", None)
    function = choice.get("function") if isinstance(choice, dict) else getattr(choice, "function", None)
    if isinstance(function, dict):
        return function.get("name")
    return getattr(function, "name", None)


def _allowed_tool_names(request: Any) -> set[str]:
    if getattr(request, _ORIGINAL_TOOL_CHOICE_NONE_ATTR, False):
        return set()

    named_tool = _named_tool_choice(request)
    if named_tool:
        return {named_tool}

    return {name for tool in (getattr(request, "tools", None) or []) if (name := _tool_name(tool))}


def _decode_argument(raw_value: str, value_type: str) -> Any:
    if value_type == "string":
        return raw_value

    value = json.loads(raw_value)
    if value_type == "number":
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("K3 number argument is not numeric.")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("K3 number argument must be finite.")
    elif value_type == "boolean":
        if not isinstance(value, bool):
            raise ValueError("K3 boolean argument is not a boolean.")
    elif value_type == "null":
        if value is not None:
            raise ValueError("K3 null argument is not null.")
    elif value_type == "object":
        if not isinstance(value, dict):
            raise ValueError("K3 object argument is not an object.")
    elif value_type == "array":
        if not isinstance(value, list):
            raise ValueError("K3 array argument is not an array.")
    else:
        raise ValueError(f"Unsupported K3 argument type: {value_type!r}.")
    return value


@dataclass(frozen=True)
class _ParsedCall:
    name: str
    arguments: str
    arguments_value: dict[str, Any] | str | None
    complete: bool


def _parse_json_arguments(
    body: str,
) -> tuple[str, dict[str, Any] | str | None] | None:
    json_start = _JSON_START_RE.search(body)
    if json_start is None:
        return None
    json_end = body.find(JSON_END, json_start.end())
    if json_end < 0:
        return "", None

    raw_value = body[json_start.end() : json_end]
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        # encoding_k3 deliberately uses a <json> block to round-trip a
        # non-empty arguments string that is not a JSON object. Preserve that
        # string verbatim in the OpenAI ``arguments`` field.
        if raw_value.strip():
            return raw_value, raw_value
        raise
    if not isinstance(value, dict):
        raise ValueError("K3 json tool arguments must be an object.")
    return _json_compact(value), value


def _parse_arguments(
    body: str,
    *,
    call_complete: bool,
) -> tuple[str, dict[str, Any] | str | None]:
    json_arguments = _parse_json_arguments(body)
    if json_arguments is not None:
        arguments, value = json_arguments
        if value is None:
            return arguments, None
        return arguments, value if call_complete else None

    parts = ["{"]
    values: dict[str, Any] = {}
    matches = list(_ARGUMENT_START_RE.finditer(body))

    for match_index, match in enumerate(matches):
        attrs = _parse_attrs(match.group("attrs"))
        key = attrs.get("key")
        value_type = attrs.get("type")
        if not key or not value_type:
            raise ValueError("K3 argument tag requires key and type attributes.")

        next_start = matches[match_index + 1].start() if match_index + 1 < len(matches) else len(body) + 1
        argument_end = body.find(ARGUMENT_END, match.end())
        argument_complete = 0 <= argument_end < next_start

        if len(parts) > 1:
            prefix = ","
        else:
            prefix = ""
        encoded_key = _json_compact(key)

        if not argument_complete:
            if value_type == "string":
                raw_value = body[match.end() :]
                safe_end = len(raw_value) - _partial_marker_overlap(raw_value, ARGUMENT_END)
                escaped_value = _json_compact(raw_value[:safe_end])[1:-1]
                parts.append(f'{prefix}{encoded_key}:"{escaped_value}')
            return "".join(parts), None

        raw_value = body[match.end() : argument_end]
        value = _decode_argument(raw_value, value_type)
        values[key] = value
        parts.append(f"{prefix}{encoded_key}:{_json_compact(value)}")

    if call_complete:
        parts.append("}")
        return "".join(parts), values
    return "".join(parts), None


def _extract_call_regions(text: str, request: Any) -> list[_ParsedCall]:
    allowed_names = _allowed_tool_names(request)
    if not allowed_names:
        return []

    tools_start = text.find(TOOLS_START)
    if tools_start < 0:
        return []
    tools_body_start = tools_start + len(TOOLS_START)
    tools_end = text.find(TOOLS_END, tools_body_start)
    tools_body_end = tools_end if tools_end >= 0 else len(text)
    tools_body = text[tools_body_start:tools_body_end]

    call_starts = list(_CALL_START_RE.finditer(tools_body))
    parsed_calls: list[_ParsedCall] = []
    for call_index, call_start in enumerate(call_starts):
        attrs = _parse_attrs(call_start.group("attrs"))
        name = attrs.get("tool")
        if not name or name not in allowed_names:
            continue

        next_call_start = (
            call_starts[call_index + 1].start() if call_index + 1 < len(call_starts) else len(tools_body) + 1
        )
        call_end = tools_body.find(CALL_END, call_start.end())
        call_complete = 0 <= call_end < next_call_start
        body_end = call_end if call_complete else min(next_call_start, len(tools_body))
        body = tools_body[call_start.end() : body_end]

        try:
            arguments, arguments_value = _parse_arguments(body, call_complete=call_complete)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

        parsed_calls.append(
            _ParsedCall(
                name=name,
                arguments=arguments,
                arguments_value=arguments_value,
                complete=call_complete and arguments_value is not None,
            )
        )
    return parsed_calls


def _response_content(text: str, *, streaming: bool) -> str:
    if THINK_END in text:
        text = text.rsplit(THINK_END, 1)[1]

    response_start = text.find(RESPONSE_START)
    if response_start >= 0:
        content_start = response_start + len(RESPONSE_START)
    elif streaming and RESPONSE_START.startswith(text):
        return ""
    else:
        content_start = 0

    boundary_positions = [
        position
        for marker in (RESPONSE_END, TOOLS_START, MESSAGE_END, END_OF_MSG_TOKEN)
        if (position := text.find(marker, content_start)) >= 0
    ]
    if boundary_positions:
        return text[content_start : min(boundary_positions)]

    content_end = len(text)
    if streaming:
        boundary_markers = (
            RESPONSE_END,
            TOOLS_START,
            MESSAGE_END,
            END_OF_MSG_TOKEN,
        )
        content_end -= max(_partial_marker_overlap(text[content_start:], marker) for marker in boundary_markers)
    return text[content_start:content_end]


class KimiK3ToolParser(ToolParser):
    """Parse K3 XTML calls, including typed and streaming arguments."""

    supports_required_and_named = False

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
    ):
        super().__init__(tokenizer, tools)
        self._streamed_content = ""
        self._streamed_call_names: list[str] = []
        self._streamed_call_ids: list[str] = []

    def adjust_request(self, request: ChatCompletionRequest):
        # Do not call ToolParser.adjust_request: required/named K3 calls use
        # XTML rather than vLLM's JSON structured-output grammar.
        request.skip_special_tokens = False
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        content = _response_content(model_output, streaming=False) or None
        parsed_calls = [
            parsed_call
            for parsed_call in _extract_call_regions(model_output, request)
            if parsed_call.complete and parsed_call.arguments_value is not None
        ]

        if not parsed_calls:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=content,
            )

        tool_calls = [
            ToolCall(
                id=make_tool_call_id(),
                type="function",
                function=FunctionCall(
                    name=parsed_call.name,
                    arguments=parsed_call.arguments,
                ),
            )
            for parsed_call in parsed_calls
        ]
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=content,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        del previous_text, delta_text, previous_token_ids, current_token_ids
        del delta_token_ids

        content_snapshot = _response_content(current_text, streaming=True)
        content_delta: str | None = None
        if content_snapshot.startswith(self._streamed_content):
            content_delta = content_snapshot[len(self._streamed_content) :]
        self._streamed_content = content_snapshot

        parsed_calls = _extract_call_regions(current_text, request)
        tool_deltas: list[DeltaToolCall] = []
        self.prev_tool_call_arr = []

        for index, parsed_call in enumerate(parsed_calls):
            while len(self._streamed_call_ids) <= index:
                self._streamed_call_ids.append(make_tool_call_id())
                self._streamed_call_names.append("")
            while len(self.streamed_args_for_tool) <= index:
                self.streamed_args_for_tool.append("")

            streamed_arguments = self.streamed_args_for_tool[index]
            if parsed_call.arguments.startswith(streamed_arguments):
                argument_delta = parsed_call.arguments[len(streamed_arguments) :]
            else:
                argument_delta = ""

            is_new_call = not self._streamed_call_names[index]
            if is_new_call:
                self._streamed_call_names[index] = parsed_call.name

            if is_new_call or argument_delta:
                tool_deltas.append(
                    DeltaToolCall(
                        index=index,
                        id=self._streamed_call_ids[index] if is_new_call else None,
                        type="function" if is_new_call else None,
                        function=DeltaFunctionCall(
                            name=parsed_call.name if is_new_call else None,
                            arguments=argument_delta,
                        ),
                    )
                )
                self.streamed_args_for_tool[index] += argument_delta

            self.prev_tool_call_arr.append(
                {
                    "name": parsed_call.name,
                    "arguments": parsed_call.arguments,
                }
            )

        if not content_delta and not tool_deltas:
            return None
        return DeltaMessage(content=content_delta or None, tool_calls=tool_deltas)


if "kimi_k3" not in ReasoningParserManager.list_registered():
    ReasoningParserManager.register_module(
        name="kimi_k3",
        module=KimiK3ReasoningParser,
        force=False,
    )

if "kimi_k3" not in ToolParserManager.list_registered():
    ToolParserManager.register_module(
        name="kimi_k3",
        module=KimiK3ToolParser,
        force=False,
    )
