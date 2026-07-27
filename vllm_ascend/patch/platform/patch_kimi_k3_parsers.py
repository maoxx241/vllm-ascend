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
"""Kimi K3 reasoning/tool adapters backed by one strict XTML state machine."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.parser_manager import ParserManager
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

from vllm_ascend.patch.platform.kimi_k3_xtml import (
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
    KimiK3ParseSnapshot,
    KimiK3XTMLParseError,
    KimiK3XTMLStateMachine,
    partial_marker_overlap,
)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

_ORIGINAL_GET_PARSER_ATTR = "_ascend_original_kimi_k3_get_parser"
_ORIGINAL_CHAT_FULL_ATTR = "_ascend_original_kimi_k3_chat_completion_full_generator"

__all__ = [
    "ARGUMENT_END",
    "CALL_END",
    "END_OF_MSG_TOKEN",
    "JSON_END",
    "MESSAGE_END",
    "RESPONSE_END",
    "RESPONSE_START",
    "SEP_TOKEN",
    "THINK_END",
    "THINK_START",
    "TOOLS_END",
    "TOOLS_START",
    "KimiK3Parser",
    "KimiK3ReasoningParser",
    "KimiK3ToolParser",
]


def _find_subsequence(values: Sequence[int], pattern: Sequence[int]) -> int:
    if not pattern or len(pattern) > len(values):
        return -1
    for index in range(len(values) - len(pattern), -1, -1):
        if list(values[index : index + len(pattern)]) == list(pattern):
            return index
    return -1


def _encode_marker(tokenizer: TokenizerLike, marker: str) -> list[int]:
    encoded = tokenizer.encode(marker)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, dict):
        encoded = encoded.get("input_ids", [])
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


def _request_uses_thinking(request: Any) -> bool:
    template_kwargs = getattr(request, "chat_template_kwargs", None) or {}
    if "thinking" in template_kwargs:
        return bool(template_kwargs["thinking"])
    return getattr(request, "reasoning_effort", None) != "none"


def adjust_kimi_k3_request(request: Any):
    """Preserve K3's adjacent control/text markers during detokenization."""

    if hasattr(request, "skip_special_tokens"):
        request.skip_special_tokens = False
    if hasattr(request, "spaces_between_special_tokens"):
        request.spaces_between_special_tokens = False
    return request


def _tool_name(tool: Any) -> str | None:
    if isinstance(tool, dict):
        function = tool.get("function")
        if isinstance(function, dict):
            return function.get("name")
        return tool.get("name") or getattr(function, "name", None)
    function = getattr(tool, "function", None)
    return getattr(function, "name", None) or getattr(tool, "name", None)


def _named_tool_choice(request: Any) -> str | None:
    choice = getattr(request, "tool_choice", None)
    function = choice.get("function") if isinstance(choice, dict) else getattr(choice, "function", None)
    if isinstance(function, dict):
        return function.get("name")
    return getattr(function, "name", None) or getattr(choice, "name", None)


def _state_machine_for_request(
    request: ChatCompletionRequest,
    *,
    thinking_enabled: bool,
) -> KimiK3XTMLStateMachine:
    tools = getattr(request, "tools", None) or []
    allowed_tool_names = frozenset(name for tool in tools if (name := _tool_name(tool)))
    choice = request.tool_choice
    named_tool = _named_tool_choice(request)

    if named_tool:
        tool_mode = "named"
    elif choice == "required":
        tool_mode = "required"
    elif choice == "auto":
        tool_mode = "auto"
    elif choice == "none" or (choice is None and not tools):
        tool_mode = "none"
    elif choice is None:
        tool_mode = "auto"
    else:
        raise KimiK3XTMLParseError(f"Unsupported K3 tool_choice: {choice!r}.")

    if tool_mode != "none" and not allowed_tool_names:
        raise KimiK3XTMLParseError(f"K3 tool_choice={tool_mode!r} requires declared tools.")
    if named_tool is not None and named_tool not in allowed_tool_names:
        raise KimiK3XTMLParseError(f"Named K3 tool choice {named_tool!r} is not declared.")

    return KimiK3XTMLStateMachine(
        thinking_enabled=thinking_enabled,
        tool_mode=tool_mode,
        allowed_tool_names=allowed_tool_names,
        named_tool=named_tool,
    )


class KimiK3ReasoningParser(ReasoningParser):
    """Expose K3 reasoning boundaries to vLLM's scheduler and adapters."""

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
        self._streamed_content = ""

    @property
    def reasoning_start_str(self) -> str:
        return THINK_START

    @property
    def reasoning_end_str(self) -> str:
        return THINK_END

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        last_start = _find_subsequence(input_ids, self.think_start_token_ids)
        last_end = _find_subsequence(input_ids, self.think_end_token_ids)
        return last_start < 0 or last_end > last_start

    def is_reasoning_end_streaming(
        self,
        input_ids: Sequence[int],
        delta_ids: Iterable[int],
    ) -> bool:
        del delta_ids
        if not self._thinking_enabled:
            return True
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
        if RESPONSE_START in output or TOOLS_START in output:
            raise KimiK3XTMLParseError("K3 response/tools began before the think block was closed.")
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
        if not self._thinking_enabled:
            return DeltaMessage(content=delta_text) if delta_text else None

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
            safe_end = len(output) - partial_marker_overlap(output, THINK_END)
            reasoning_snapshot = output[reasoning_start:safe_end]
            content = None

        if not reasoning_snapshot.startswith(self._streamed_reasoning):
            raise KimiK3XTMLParseError("K3 reasoning stream no longer extends its emitted prefix.")
        if content is not None and not content.startswith(self._streamed_content):
            raise KimiK3XTMLParseError("K3 content stream no longer extends its emitted prefix.")
        reasoning_delta = reasoning_snapshot[len(self._streamed_reasoning) :]
        content_delta = None
        if content is not None:
            content_delta = content[len(self._streamed_content) :]
            self._streamed_content = content
        self._streamed_reasoning = reasoning_snapshot
        if not reasoning_delta and not content_delta:
            return None
        return DeltaMessage(
            reasoning=reasoning_delta or None,
            content=content_delta or None,
        )

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        if not self._thinking_enabled:
            return 0
        start_index = _find_subsequence(token_ids, self.think_start_token_ids)
        content_start = start_index + len(self.think_start_token_ids) if start_index >= 0 else 0
        end_index = _find_subsequence(token_ids, self.think_end_token_ids)
        if end_index < content_start:
            end_index = len(token_ids)
        return max(0, end_index - content_start)

    def adjust_request(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ChatCompletionRequest | ResponsesRequest:
        return adjust_kimi_k3_request(request)


class _KimiK3StreamingAdapter:
    def _init_streaming_adapter(self) -> None:
        self._stream_text = ""
        self._stream_reasoning = ""
        self._stream_content = ""
        self._stream_call_count = 0
        self._stream_call_ids: list[str] = []
        self._stream_machine: KimiK3XTMLStateMachine | None = None

    def _get_stream_machine(
        self,
        request: ChatCompletionRequest,
        *,
        thinking_enabled: bool,
    ) -> KimiK3XTMLStateMachine:
        if self._stream_machine is None:
            self._stream_machine = _state_machine_for_request(
                request,
                thinking_enabled=thinking_enabled,
            )
        return self._stream_machine

    def _snapshot_delta(
        self,
        snapshot: KimiK3ParseSnapshot,
        *,
        include_content: bool,
        include_reasoning: bool,
        emit_new_tool_calls: bool = True,
    ) -> DeltaMessage | None:
        if not snapshot.reasoning.startswith(self._stream_reasoning):
            raise KimiK3XTMLParseError("K3 reasoning stream no longer extends its emitted prefix.")
        if not snapshot.content.startswith(self._stream_content):
            raise KimiK3XTMLParseError("K3 content stream no longer extends its emitted prefix.")
        visible_tool_calls = (
            snapshot.tool_calls if emit_new_tool_calls else snapshot.tool_calls[: self._stream_call_count]
        )
        if len(visible_tool_calls) < self._stream_call_count:
            raise KimiK3XTMLParseError("K3 tool stream attempted to retract an emitted call.")

        reasoning_delta = snapshot.reasoning[len(self._stream_reasoning) :]
        content_delta = snapshot.content[len(self._stream_content) :]
        tool_deltas: list[DeltaToolCall] = []
        for index in range(self._stream_call_count, len(visible_tool_calls)):
            call = visible_tool_calls[index]
            while len(self._stream_call_ids) <= index:
                self._stream_call_ids.append(make_tool_call_id())
            tool_deltas.append(
                DeltaToolCall(
                    index=index,
                    id=self._stream_call_ids[index],
                    type="function",
                    function=DeltaFunctionCall(
                        name=call.name,
                        arguments=call.arguments,
                    ),
                )
            )

        self._stream_reasoning = snapshot.reasoning
        self._stream_content = snapshot.content
        self._stream_call_count = len(visible_tool_calls)

        if not include_content:
            content_delta = ""
        if not include_reasoning:
            reasoning_delta = ""
        if not reasoning_delta and not content_delta and not tool_deltas:
            return None
        return DeltaMessage(
            reasoning=reasoning_delta or None,
            content=content_delta or None,
            tool_calls=tool_deltas,
        )


class KimiK3ToolParser(ToolParser, _KimiK3StreamingAdapter):
    """Strict K3 XTML ToolParser adapter for compatibility APIs."""

    supports_required_and_named = False

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
    ):
        super().__init__(tokenizer, tools)
        self._init_streaming_adapter()

    def adjust_request(self, request: ChatCompletionRequest):
        return adjust_kimi_k3_request(request)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        thinking_enabled = _request_uses_thinking(request) and (
            THINK_END in model_output or model_output.startswith(THINK_START)
        )
        machine = _state_machine_for_request(
            request,
            thinking_enabled=thinking_enabled,
        )
        snapshot = machine.parse(model_output, final=True)
        tool_calls = [
            ToolCall(
                id=make_tool_call_id(),
                type="function",
                function=FunctionCall(
                    name=call.name,
                    arguments=call.arguments,
                ),
            )
            for call in snapshot.tool_calls
        ]
        content = snapshot.content or None
        if machine.tool_mode in ("required", "named"):
            content = None
        return ExtractedToolCallInformation(
            tools_called=bool(tool_calls),
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
        machine = self._get_stream_machine(
            request,
            thinking_enabled=False,
        )
        snapshot = machine.parse(current_text, final=False)
        return self._snapshot_delta(
            snapshot,
            include_content=machine.tool_mode not in ("required", "named"),
            include_reasoning=False,
        )


class KimiK3Parser(DelegatingParser, _KimiK3StreamingAdapter):
    """K3-local unified parser used by Chat Completions in vLLM 0.23."""

    reasoning_parser_cls = KimiK3ReasoningParser
    tool_parser_cls = KimiK3ToolParser

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(tokenizer, tools, *args, **kwargs)
        template_kwargs = kwargs.get("chat_template_kwargs") or {}
        self._thinking_enabled = bool(template_kwargs.get("thinking", True))
        self._init_streaming_adapter()

    def parse(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
        enable_auto_tools: bool = False,
    ) -> tuple[str | None, str | None, list[FunctionCall] | None]:
        del enable_auto_tools
        if not isinstance(request, ChatCompletionRequest):
            return super().parse(model_output, request, enable_auto_tools=False)

        machine = _state_machine_for_request(
            request,
            thinking_enabled=self._thinking_enabled,
        )
        snapshot = machine.parse(model_output, final=True)
        tool_calls = [
            FunctionCall(
                id=make_tool_call_id(),
                name=call.name,
                arguments=call.arguments,
            )
            for call in snapshot.tool_calls
        ]
        content = snapshot.content or None
        if machine.tool_mode in ("required", "named"):
            content = None
        return snapshot.reasoning or None, content, tool_calls

    def parse_delta(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: ChatCompletionRequest | ResponsesRequest,
        prompt_token_ids: list[int] | None = None,
        *,
        finished: bool,
    ) -> DeltaMessage | None:
        if not isinstance(request, ChatCompletionRequest):
            return super().parse_delta(
                delta_text,
                delta_token_ids,
                request,
                prompt_token_ids=prompt_token_ids,
                finished=finished,
            )

        self._stream_text += delta_text
        machine = self._get_stream_machine(
            request,
            thinking_enabled=self._thinking_enabled,
        )
        snapshot = machine.parse(self._stream_text, final=finished)
        return self._snapshot_delta(
            snapshot,
            include_content=machine.tool_mode not in ("required", "named"),
            include_reasoning=bool(request.include_reasoning),
            # A completed tools block can still be followed by malformed
            # trailing protocol. Buffer calls until the terminal generation
            # chunk so streamed deltas never need to be retracted.
            emit_new_tool_calls=finished,
        )


def _is_remote_decode_request(request: ChatCompletionRequest) -> bool:
    kv_transfer_params = getattr(request, "kv_transfer_params", None)
    return bool(kv_transfer_params and kv_transfer_params.get("do_remote_decode") is True)


async def _wrapped_chat_completion_full_generator(
    self,
    request,
    result_generator,
    request_id,
    model_name,
    conversation,
    tokenizer,
    request_metadata,
    parser=None,
):
    original = getattr(self, _ORIGINAL_CHAT_FULL_ATTR)
    if isinstance(parser, KimiK3Parser) and _is_remote_decode_request(request):
        # P-side output is an internal transfer token, not a complete K3
        # response envelope, so strict XTML parsing must not run here.
        parser = None
    return await original(
        request,
        result_generator,
        request_id,
        model_name,
        conversation,
        tokenizer,
        request_metadata,
        parser,
    )


def _install_chat_completion_full_generator_patch() -> None:
    if hasattr(OpenAIServingChat, _ORIGINAL_CHAT_FULL_ATTR):
        return
    setattr(
        OpenAIServingChat,
        _ORIGINAL_CHAT_FULL_ATTR,
        OpenAIServingChat.chat_completion_full_generator,
    )
    OpenAIServingChat.chat_completion_full_generator = _wrapped_chat_completion_full_generator


_install_chat_completion_full_generator_patch()


if not hasattr(ParserManager, _ORIGINAL_GET_PARSER_ATTR):
    setattr(
        ParserManager,
        _ORIGINAL_GET_PARSER_ATTR,
        ParserManager.get_parser.__func__,
    )


def _get_parser_with_kimi_k3(
    cls,
    tool_parser_name: str | None = None,
    reasoning_parser_name: str | None = None,
    enable_auto_tools: bool = False,
    model_name: str | None = None,
):
    uses_kimi_k3 = tool_parser_name == "kimi_k3" or reasoning_parser_name == "kimi_k3"
    if uses_kimi_k3:
        if tool_parser_name != "kimi_k3" or reasoning_parser_name != "kimi_k3" or not enable_auto_tools:
            raise ValueError(
                "Kimi K3 requires --enable-auto-tool-choice together with "
                "--reasoning-parser kimi_k3 and --tool-call-parser kimi_k3."
            )
        return KimiK3Parser

    original = getattr(ParserManager, _ORIGINAL_GET_PARSER_ATTR)
    return original(
        cls,
        tool_parser_name=tool_parser_name,
        reasoning_parser_name=reasoning_parser_name,
        enable_auto_tools=enable_auto_tools,
        model_name=model_name,
    )


ParserManager.get_parser = classmethod(_get_parser_with_kimi_k3)


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
