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
"""Strict Kimi K3 XTML parsing shared by streaming and full responses."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Literal

import regex as re

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

ToolMode = Literal["none", "auto", "required", "named"]

_ATTR_RE = re.compile(r'([A-Za-z_][\w.-]*)="([^"]*)"')
_UNKNOWN_ENTITY_RE = re.compile(r"&(?!amp;|quot;)")
_CALL_START_RE = re.compile(re.escape(f"{OPEN_TOKEN}call") + r"(?P<attrs>[^<]*?)" + re.escape(SEP_TOKEN))
_ARGUMENT_START_RE = re.compile(re.escape(f"{OPEN_TOKEN}argument") + r"(?P<attrs>[^<]*?)" + re.escape(SEP_TOKEN))
_JSON_START_RE = re.compile(re.escape(f"{OPEN_TOKEN}json") + r"(?P<attrs>[^<]*?)" + re.escape(SEP_TOKEN))


class KimiK3XTMLParseError(RuntimeError):
    """Raised when a completed K3 response violates the XTML contract."""


@dataclass(frozen=True)
class KimiK3ParsedCall:
    name: str
    index: int
    arguments: str


@dataclass(frozen=True)
class KimiK3ParseSnapshot:
    reasoning: str = ""
    content: str = ""
    tool_calls: tuple[KimiK3ParsedCall, ...] = ()
    protocol_complete: bool = False


def partial_marker_overlap(text: str, marker: str) -> int:
    """Return the suffix length that can still grow into *marker*."""

    max_overlap = min(len(text), len(marker) - 1)
    for overlap in range(max_overlap, 0, -1):
        if text.endswith(marker[:overlap]):
            return overlap
    return 0


def _safe_prefix(text: str, markers: tuple[str, ...], *, final: bool) -> str:
    overlap = max((partial_marker_overlap(text, marker) for marker in markers), default=0)
    if final:
        # K3 control tokens are atomic. A completed generation ending after a
        # full ``<|open|>``/``<|close|>`` token but before its tag is malformed,
        # not user content that may safely be returned.
        if overlap >= min(len(OPEN_TOKEN), len(CLOSE_TOKEN)):
            raise KimiK3XTMLParseError("K3 generation ended inside an XTML structural marker.")
        return text
    return text[:-overlap] if overlap else text


def _decode_attr_value(value: str) -> str:
    if _UNKNOWN_ENTITY_RE.search(value):
        raise KimiK3XTMLParseError(f"Unsupported XTML attribute entity in {value!r}.")
    # Decode in this order so ``&amp;quot;`` remains literal ``&quot;``.
    return value.replace("&quot;", '"').replace("&amp;", "&")


def _parse_attrs(raw_attrs: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    cursor = 0
    for match in _ATTR_RE.finditer(raw_attrs):
        if raw_attrs[cursor : match.start()].strip():
            raise KimiK3XTMLParseError(f"Malformed XTML attributes: {raw_attrs!r}.")
        key = match.group(1)
        if key in attrs:
            raise KimiK3XTMLParseError(f"Duplicate XTML attribute {key!r}.")
        attrs[key] = _decode_attr_value(match.group(2))
        cursor = match.end()
    if raw_attrs[cursor:].strip():
        raise KimiK3XTMLParseError(f"Malformed XTML attributes: {raw_attrs!r}.")
    return attrs


def _require_attrs(
    raw_attrs: str,
    *,
    required: frozenset[str],
    tag: str,
) -> dict[str, str]:
    attrs = _parse_attrs(raw_attrs)
    if frozenset(attrs) != required:
        raise KimiK3XTMLParseError(f"K3 {tag} attributes must be exactly {sorted(required)}, got {sorted(attrs)}.")
    return attrs


def _reject_json_constant(value: str):
    raise KimiK3XTMLParseError(f"Non-finite JSON number {value!r} is not allowed.")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise KimiK3XTMLParseError(f"Duplicate JSON key {key!r}.")
        value[key] = item
    return value


def _load_json(raw_value: str):
    try:
        return json.loads(
            raw_value,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except KimiK3XTMLParseError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise KimiK3XTMLParseError("Invalid JSON in K3 tool arguments.") from exc


def _json_compact(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _decode_typed_argument(raw_value: str, value_type: str):
    if value_type == "string":
        return raw_value

    value = _load_json(raw_value)
    if value_type == "number":
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise KimiK3XTMLParseError("K3 number argument is not numeric.")
        if isinstance(value, float) and not math.isfinite(value):
            raise KimiK3XTMLParseError("K3 number argument must be finite.")
    elif value_type == "boolean":
        if not isinstance(value, bool):
            raise KimiK3XTMLParseError("K3 boolean argument is not a boolean.")
    elif value_type == "null":
        if value is not None:
            raise KimiK3XTMLParseError("K3 null argument is not null.")
    elif value_type == "object":
        if not isinstance(value, dict):
            raise KimiK3XTMLParseError("K3 object argument is not an object.")
    elif value_type == "array":
        if not isinstance(value, list):
            raise KimiK3XTMLParseError("K3 array argument is not an array.")
    else:
        raise KimiK3XTMLParseError(f"Unsupported K3 argument type: {value_type!r}.")
    return value


def _parse_call_arguments(body: str) -> str:
    cursor = 0
    while cursor < len(body) and body[cursor].isspace():
        cursor += 1
    if cursor == len(body):
        return "{}"

    json_match = _JSON_START_RE.match(body, cursor)
    if json_match is not None:
        attrs = _require_attrs(
            json_match.group("attrs"),
            required=frozenset({"type"}),
            tag="json",
        )
        if attrs["type"] != "object":
            raise KimiK3XTMLParseError("K3 json arguments must use type='object'.")
        json_end = body.find(JSON_END, json_match.end())
        if json_end < 0:
            raise KimiK3XTMLParseError("K3 json argument block is not closed.")
        raw_value = body[json_match.end() : json_end]
        value = _load_json(raw_value)
        if not isinstance(value, dict):
            raise KimiK3XTMLParseError("K3 json tool arguments must be an object.")
        if body[json_end + len(JSON_END) :].strip():
            raise KimiK3XTMLParseError("K3 call mixes json arguments with typed arguments or stray text.")
        return _json_compact(value)

    values: dict[str, object] = {}
    while cursor < len(body):
        argument_match = _ARGUMENT_START_RE.match(body, cursor)
        if argument_match is None:
            raise KimiK3XTMLParseError("Unexpected text or tag in K3 typed arguments.")
        attrs = _require_attrs(
            argument_match.group("attrs"),
            required=frozenset({"key", "type"}),
            tag="argument",
        )
        key = attrs["key"]
        if not key:
            raise KimiK3XTMLParseError("K3 argument key must not be empty.")
        if key in values:
            raise KimiK3XTMLParseError(f"Duplicate K3 argument key {key!r}.")

        argument_end = body.find(ARGUMENT_END, argument_match.end())
        if argument_end < 0:
            raise KimiK3XTMLParseError(f"K3 argument {key!r} is not closed.")
        raw_value = body[argument_match.end() : argument_end]
        values[key] = _decode_typed_argument(raw_value, attrs["type"])
        cursor = argument_end + len(ARGUMENT_END)
        while cursor < len(body) and body[cursor].isspace():
            cursor += 1

    return _json_compact(values)


def _parse_calls(
    body: str,
    *,
    allowed_tool_names: frozenset[str],
    validate_tool_names: bool,
    named_tool: str | None,
) -> tuple[KimiK3ParsedCall, ...]:
    calls: list[KimiK3ParsedCall] = []
    seen_indices: set[int] = set()
    cursor = 0
    while cursor < len(body):
        while cursor < len(body) and body[cursor].isspace():
            cursor += 1
        if cursor == len(body):
            break
        call_match = _CALL_START_RE.match(body, cursor)
        if call_match is None:
            raise KimiK3XTMLParseError("Unexpected text or tag in K3 tools block.")
        attrs = _require_attrs(
            call_match.group("attrs"),
            required=frozenset({"tool", "index"}),
            tag="call",
        )
        name = attrs["tool"]
        if not name:
            raise KimiK3XTMLParseError("K3 tool name must not be empty.")
        try:
            index = int(attrs["index"])
        except ValueError as exc:
            raise KimiK3XTMLParseError("K3 call index must be an integer.") from exc
        if index <= 0 or str(index) != attrs["index"]:
            raise KimiK3XTMLParseError("K3 call index must be a canonical positive integer.")
        expected_index = len(calls) + 1
        if index != expected_index or index in seen_indices:
            raise KimiK3XTMLParseError(
                f"K3 call indices must be unique and sequential from 1; expected {expected_index}, got {index}."
            )
        seen_indices.add(index)

        if validate_tool_names and name not in allowed_tool_names:
            raise KimiK3XTMLParseError(f"Unknown K3 tool {name!r}.")
        if named_tool is not None and name != named_tool:
            raise KimiK3XTMLParseError(f"Named K3 tool choice requires {named_tool!r}, got {name!r}.")

        call_end = body.find(CALL_END, call_match.end())
        if call_end < 0:
            raise KimiK3XTMLParseError(f"K3 call {name!r} is not closed.")
        arguments = _parse_call_arguments(body[call_match.end() : call_end])
        calls.append(
            KimiK3ParsedCall(
                name=name,
                index=index,
                arguments=arguments,
            )
        )
        cursor = call_end + len(CALL_END)

    return tuple(calls)


def _consume_terminal_envelope(text: str, cursor: int, *, final: bool) -> bool:
    tail = text[cursor:]
    if not tail:
        if final:
            raise KimiK3XTMLParseError("K3 response is missing the closing message marker.")
        return False

    if tail.startswith(MESSAGE_END):
        tail = tail[len(MESSAGE_END) :]
    elif not final and MESSAGE_END.startswith(tail):
        return False
    else:
        raise KimiK3XTMLParseError("K3 response is missing the closing message marker.")

    if tail.startswith(END_OF_MSG_TOKEN):
        tail = tail[len(END_OF_MSG_TOKEN) :]
    elif not final and END_OF_MSG_TOKEN.startswith(tail):
        return False

    if tail.strip():
        raise KimiK3XTMLParseError("Unexpected text or additional XTML blocks after the K3 response.")
    return True


class KimiK3XTMLStateMachine:
    """Parse a K3 assistant envelope without streaming-only fallback rules."""

    def __init__(
        self,
        *,
        thinking_enabled: bool,
        tool_mode: ToolMode,
        allowed_tool_names: frozenset[str],
        named_tool: str | None = None,
    ) -> None:
        self.thinking_enabled = thinking_enabled
        self.tool_mode = tool_mode
        self.allowed_tool_names = allowed_tool_names
        self.named_tool = named_tool

    def _validate_required_call(self, *, final: bool, call_count: int) -> None:
        if final and self.tool_mode in ("required", "named") and call_count == 0:
            raise KimiK3XTMLParseError(f"K3 tool_choice={self.tool_mode!r} completed without a valid tool call.")

    def parse(self, text: str, *, final: bool) -> KimiK3ParseSnapshot:
        reasoning = ""
        cursor = 0

        if self.thinking_enabled:
            if text.startswith(THINK_START):
                cursor = len(THINK_START)
            elif not final and THINK_START.startswith(text):
                return KimiK3ParseSnapshot()

            think_end = text.find(THINK_END, cursor)
            if think_end < 0:
                if any(marker in text[cursor:] for marker in (RESPONSE_START, RESPONSE_END, TOOLS_START)):
                    raise KimiK3XTMLParseError("K3 response entered response/tools before closing think.")
                reasoning = _safe_prefix(
                    text[cursor:],
                    (THINK_END,),
                    final=final,
                )
                self._validate_required_call(final=final, call_count=0)
                return KimiK3ParseSnapshot(reasoning=reasoning)

            reasoning = text[cursor:think_end]
            cursor = think_end + len(THINK_END)
            response_tail = text[cursor:]
            if response_tail.startswith(RESPONSE_START):
                cursor += len(RESPONSE_START)
            elif not final and RESPONSE_START.startswith(response_tail):
                return KimiK3ParseSnapshot(reasoning=reasoning)
            else:
                raise KimiK3XTMLParseError("K3 think block must be followed by a response block.")
        else:
            if text.startswith(RESPONSE_START):
                cursor = len(RESPONSE_START)
            elif not final and RESPONSE_START.startswith(text):
                return KimiK3ParseSnapshot()

        response_end = text.find(RESPONSE_END, cursor)
        if response_end < 0:
            response_body = text[cursor:]
            if any(marker in response_body for marker in (THINK_START, THINK_END, TOOLS_START, TOOLS_END)):
                raise KimiK3XTMLParseError("Unexpected XTML block before K3 response was closed.")
            self._validate_required_call(final=final, call_count=0)
            return KimiK3ParseSnapshot(
                reasoning=reasoning,
                content=_safe_prefix(
                    response_body,
                    (RESPONSE_END,),
                    final=final,
                ),
            )

        content = text[cursor:response_end]
        cursor = response_end + len(RESPONSE_END)
        tail = text[cursor:]

        if not tail:
            self._validate_required_call(final=final, call_count=0)
            protocol_complete = _consume_terminal_envelope(
                text,
                cursor,
                final=final,
            )
            return KimiK3ParseSnapshot(
                reasoning=reasoning,
                content=content,
                protocol_complete=protocol_complete,
            )

        if not final and any(marker.startswith(tail) for marker in (TOOLS_START, MESSAGE_END, END_OF_MSG_TOKEN)):
            return KimiK3ParseSnapshot(reasoning=reasoning, content=content)

        parsed_calls: tuple[KimiK3ParsedCall, ...] = ()
        if tail.startswith(TOOLS_START):
            tools_body_start = cursor + len(TOOLS_START)
            tools_end = text.find(TOOLS_END, tools_body_start)
            if tools_end < 0:
                if final:
                    raise KimiK3XTMLParseError("K3 tools block is not closed.")
                return KimiK3ParseSnapshot(reasoning=reasoning, content=content)

            parsed_calls = _parse_calls(
                text[tools_body_start:tools_end],
                allowed_tool_names=self.allowed_tool_names,
                validate_tool_names=self.tool_mode != "none",
                named_tool=self.named_tool,
            )
            cursor = tools_end + len(TOOLS_END)

        protocol_complete = _consume_terminal_envelope(text, cursor, final=final)

        self._validate_required_call(
            final=final,
            call_count=len(parsed_calls),
        )

        # ``none`` still parses and validates the envelope, but never emits
        # hallucinated calls to the API.
        emitted_calls = () if self.tool_mode == "none" else parsed_calls
        return KimiK3ParseSnapshot(
            reasoning=reasoning,
            content=content,
            tool_calls=emitted_calls,
            protocol_complete=protocol_complete,
        )
