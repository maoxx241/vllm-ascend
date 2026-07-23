# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
from pathlib import Path

import pytest
from transformers import AutoTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tokenizers.detokenizer_utils import detokenize_incrementally

from vllm_ascend.patch.platform.patch_kimi_k3_parsers import (
    ARGUMENT_END,
    CALL_END,
    MESSAGE_END,
    RESPONSE_END,
    RESPONSE_START,
    THINK_END,
    TOOLS_END,
    TOOLS_START,
    KimiK3Parser,
)
from vllm_ascend.patch.platform.patch_kimi_k3_renderer import (
    KIMI_K3_IMAGE_PROMPT,
)

_TOKENIZER_PATH_ENV = "KIMI_K3_TOKENIZER_PATH"
_KIMI_K3_TOKENIZER_FILE_SHA256 = {
    "tiktoken.model": "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103",
    "tokenization_kimi.py": "f28ea66e2d862a2a5814970b2ce40c2f7d8296ff09aed90a7e7def689b906944",
    "encoding_k3.py": "c3869cdb7c5a81b1ee621e55ba589d8f3ffae83063c1085571ee96e2feb826a8",
    "tokenizer_config.json": "5d0803c94db9cd78763499e0956c95fd5a225c14a727e5a6cf5db3f96f010a6e",
}


@pytest.fixture(scope="module")
def real_kimi_k3_tokenizer():
    tokenizer_path = os.getenv(_TOKENIZER_PATH_ENV)
    if not tokenizer_path:
        pytest.skip(f"{_TOKENIZER_PATH_ENV} is not set")
    path = Path(tokenizer_path)
    for filename, expected_digest in _KIMI_K3_TOKENIZER_FILE_SHA256.items():
        tokenizer_file = path / filename
        assert tokenizer_file.is_file(), f"{tokenizer_file} is unavailable"
        assert hashlib.sha256(tokenizer_file.read_bytes()).hexdigest() == (expected_digest)
    return AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
        use_fast=False,
    )


def _incremental_deltas(tokenizer, token_ids, *, spaces_between_special_tokens):
    all_ids: list[int] = []
    previous_tokens: list[str] = []
    prefix_offset = 0
    read_offset = 0
    parts: list[str] = []
    for token_id in token_ids:
        all_ids.append(token_id)
        (
            new_tokens,
            delta,
            prefix_offset,
            read_offset,
        ) = detokenize_incrementally(
            tokenizer,
            all_ids,
            previous_tokens,
            prefix_offset,
            read_offset,
            skip_special_tokens=False,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )
        previous_tokens.extend(new_tokens)
        parts.append(delta)
    return parts


def _incremental_decode(tokenizer, token_ids, *, spaces_between_special_tokens):
    return "".join(
        _incremental_deltas(
            tokenizer,
            token_ids,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )
    )


def test_real_incremental_detokenizer_preserves_adjacent_xtml_markers(
    real_kimi_k3_tokenizer,
):
    expected = THINK_END + RESPONSE_START
    token_ids = real_kimi_k3_tokenizer.encode(
        expected,
        add_special_tokens=False,
    )

    assert (
        _incremental_decode(
            real_kimi_k3_tokenizer,
            token_ids,
            spaces_between_special_tokens=False,
        )
        == expected
    )
    assert (
        _incremental_decode(
            real_kimi_k3_tokenizer,
            token_ids,
            spaces_between_special_tokens=True,
        )
        != expected
    )


def test_real_incremental_detokenizer_reconstructs_thinking_chat_response(
    real_kimi_k3_tokenizer,
):
    generated = "private reasoning" + THINK_END + RESPONSE_START + "public answer" + RESPONSE_END + MESSAGE_END
    token_ids = real_kimi_k3_tokenizer.encode(
        generated,
        add_special_tokens=False,
    )
    deltas = _incremental_deltas(
        real_kimi_k3_tokenizer,
        token_ids,
        spaces_between_special_tokens=False,
    )
    request = ChatCompletionRequest(
        model="kimi-k3",
        messages=[{"role": "user", "content": "answer"}],
        reasoning_effort="max",
    )
    parser = KimiK3Parser(
        real_kimi_k3_tokenizer,
        chat_template_kwargs={"thinking": True},
    )
    reasoning_parts: list[str] = []
    content_parts: list[str] = []

    for index, (token_id, delta_text) in enumerate(zip(token_ids, deltas, strict=True)):
        delta = parser.parse_delta(
            delta_text=delta_text,
            delta_token_ids=[token_id],
            request=request,
            finished=index == len(token_ids) - 1,
        )
        if delta is not None:
            if delta.reasoning:
                reasoning_parts.append(delta.reasoning)
            if delta.content:
                content_parts.append(delta.content)

    assert "".join(deltas) == generated
    assert "".join(reasoning_parts) == "private reasoning"
    assert "".join(content_parts) == "public answer"


def test_real_incremental_detokenizer_extracts_bfcl_native_tool_call(
    real_kimi_k3_tokenizer,
):
    generated = (
        RESPONSE_END
        + TOOLS_START
        + '<|open|>call tool="solve_quadratic_equation" index="1"<|sep|>'
        + '<|open|>argument key="a" type="number"<|sep|>2'
        + ARGUMENT_END
        + '<|open|>argument key="b" type="number"<|sep|>6'
        + ARGUMENT_END
        + '<|open|>argument key="c" type="number"<|sep|>5'
        + ARGUMENT_END
        + CALL_END
        + TOOLS_END
        + MESSAGE_END
    )
    token_ids = real_kimi_k3_tokenizer.encode(
        generated,
        add_special_tokens=False,
    )
    deltas = _incremental_deltas(
        real_kimi_k3_tokenizer,
        token_ids,
        spaces_between_special_tokens=False,
    )
    tools = [
        {
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
                },
            },
        }
    ]
    request = ChatCompletionRequest(
        model="kimi-k3",
        messages=[{"role": "user", "content": "solve"}],
        tools=tools,
        tool_choice="auto",
        reasoning_effort="none",
    )
    parser = KimiK3Parser(
        real_kimi_k3_tokenizer,
        tools,
        chat_template_kwargs={"thinking": False},
    )
    emitted_calls = []
    content_parts: list[str] = []

    for index, (token_id, delta_text) in enumerate(zip(token_ids, deltas, strict=True)):
        delta = parser.parse_delta(
            delta_text=delta_text,
            delta_token_ids=[token_id],
            request=request,
            finished=index == len(token_ids) - 1,
        )
        if delta is not None:
            if delta.content:
                content_parts.append(delta.content)
            emitted_calls.extend(delta.tool_calls)

    assert "".join(deltas) == generated
    assert content_parts == []
    assert len(emitted_calls) == 1
    assert emitted_calls[0].function is not None
    assert emitted_calls[0].function.name == "solve_quadratic_equation"
    assert emitted_calls[0].function.arguments == '{"a":2,"b":6,"c":5}'


def test_real_segmented_encoding_blocks_prompt_marker_injection(
    real_kimi_k3_tokenizer,
):
    user_text = f"literal user text: {TOOLS_START}"
    conversation = [{"role": "user", "content": user_text}]

    trusted_ids = real_kimi_k3_tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        thinking=False,
    )
    rendered_text = real_kimi_k3_tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        thinking=False,
    )
    unsafe_ids = real_kimi_k3_tokenizer.encode(
        rendered_text,
        add_special_tokens=False,
    )

    open_token_id = real_kimi_k3_tokenizer.convert_tokens_to_ids("<|open|>")
    assert unsafe_ids.count(open_token_id) > trusted_ids.count(open_token_id)
    assert trusted_ids != unsafe_ids


@pytest.mark.parametrize("image_count", [0, 1, 2])
def test_real_segmented_encoding_uses_only_trusted_image_prompts(
    real_kimi_k3_tokenizer,
    image_count,
):
    content = [{"type": "text", "text": "before"}]
    for index in range(image_count):
        content.extend(
            [
                {"type": "image"},
                {"type": "text", "text": f"after-{index}"},
            ]
        )
    conversation = [{"role": "user", "content": content}]

    prompt_ids = real_kimi_k3_tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        thinking=False,
        image_prompts=[KIMI_K3_IMAGE_PROMPT] * image_count,
    )

    media_pad_id = real_kimi_k3_tokenizer.convert_tokens_to_ids("<|media_pad|>")
    assert prompt_ids.count(media_pad_id) == image_count
