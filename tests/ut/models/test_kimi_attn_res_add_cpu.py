# SPDX-License-Identifier: Apache-2.0
"""Test residual-add ownership without importing the NPU runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def load_functions():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/kimi_k3.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    apply = next(node for node in tree.body if getattr(node, "name", None) == "_apply_ascend_attn_res")
    layer = next(node for node in tree.body if getattr(node, "name", None) == "AscendKimiDecoderLayer")
    forward = next(node for node in layer.body if getattr(node, "name", None) == "forward_attn_residual")
    scope = {"torch": torch, "apply_attn_res": None}
    module = ast.Module(body=[apply, forward], type_ignores=[])
    exec(compile("from __future__ import annotations\n" + ast.unparse(module), str(path), "exec"), scope)
    return scope


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tokens", [0, 4])
@pytest.mark.parametrize("blocks", [0, 1, 4])
def test_add_preserves_input_alias_and_intermediate_rounding(dtype, tokens, blocks):
    apply = load_functions()["_apply_ascend_attn_res"]
    torch.manual_seed(42)
    prefix_storage = torch.randn(tokens + 2, 32).to(dtype)
    addend_storage = torch.randn(tokens + 2, 32).to(dtype)
    prefix = prefix_storage[1:-1, ::2]
    addend = addend_storage[1:-1, ::2]
    prefix_before = prefix_storage.clone()
    addend_before = addend_storage.clone()
    residual = torch.randn(tokens, 8, 16).to(dtype)
    residual[:, blocks:] = float("nan")
    projection = SimpleNamespace(weight=torch.randn(1, 16).to(dtype))
    norm = SimpleNamespace(weight=torch.randn(16).to(dtype), variance_epsilon=1e-5)
    expected_prefix = prefix + addend
    expected = apply(expected_prefix, residual, projection, norm, blocks)
    out_storage = torch.full_like(prefix_storage, 13)
    out = out_storage[1:-1, ::2]

    actual = apply(prefix, residual, projection, norm, blocks, addend, out)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(out, expected_prefix, rtol=0, atol=0)
    torch.testing.assert_close(prefix_storage, prefix_before, rtol=0, atol=0)
    torch.testing.assert_close(addend_storage, addend_before, rtol=0, atol=0)
    assert torch.all(out_storage[[0, -1]] == 13)
    assert torch.all(out_storage[:, 1::2] == 13)


def test_93_layers_fuse_85_adds_and_preserve_saved_dspark_prefixes():
    scope = load_functions()
    apply = scope["_apply_ascend_attn_res"]
    forward = scope["forward_attn_residual"]
    calls = []

    def recording_apply(*args, **kwargs):
        calls.append(len(args) > 5 and args[5] is not None)
        return apply(*args, **kwargs)

    scope["_apply_ascend_attn_res"] = recording_apply
    torch.manual_seed(17)
    hidden = torch.randn(4, 16).to(torch.bfloat16)
    residual = torch.empty(4, 8, 16, dtype=hidden.dtype)
    projection = SimpleNamespace(weight=torch.randn(1, 16))
    norm = SimpleNamespace(weight=torch.randn(16), variance_epsilon=1e-5)
    for idx in range(93):
        prev_blocks = (idx + 11) // 12
        write = idx % 12 == 0
        layer = SimpleNamespace(
            use_sequence_parallel=False,
            prev_valid_blocks=prev_blocks,
            is_block_write_layer=write,
            block_write_idx=idx // 12,
            self_attention_res_proj=projection,
            self_attention_res_norm=norm,
            mlp_res_proj=projection,
            mlp_res_norm=norm,
            input_layernorm=lambda x: x,
            post_attention_layernorm=lambda x: x,
            self_attn=lambda *, hidden_states, positions: hidden_states * 0.25,
            mlp=lambda x: x * 0.125,
        )
        old_alias, old_copy = hidden, hidden.clone()
        materialized = apply(hidden, residual, projection, norm, prev_blocks)
        if write:
            residual[:, idx // 12].copy_(hidden)
        attn_out = materialized * 0.25
        expected_prefix = attn_out if write else hidden + attn_out
        expected = (
            expected_prefix + apply(expected_prefix, residual, projection, norm, prev_blocks + int(write)) * 0.125
        )

        hidden, returned_residual = forward(layer, torch.arange(4), hidden, residual)

        torch.testing.assert_close(hidden, expected, rtol=0, atol=0)
        torch.testing.assert_close(old_alias, old_copy, rtol=0, atol=0)
        assert returned_residual is residual
    assert len(calls) == 186
    assert sum(calls) == 85
