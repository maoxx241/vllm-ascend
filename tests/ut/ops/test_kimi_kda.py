# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention

from vllm_ascend.ops.kimi_kda import (
    AscendKimiGatedDeltaNetAttention,
    _load_kimi_k3_a_log,
)
from vllm_ascend.ops.kimi_kda_state import kimi_kda_state_shape
from vllm_ascend.ops.parallel_types import AscendTokenLayout
from vllm_ascend.transformers_utils.configs.kimi_k3 import KimiK3TextConfig


def _bare_kimi_kda(*, head_dim: int = 2, lower_bound: float = -5.0):
    layer = object.__new__(AscendKimiGatedDeltaNetAttention)
    nn.Module.__init__(layer)
    layer.head_dim = head_dim
    layer.gate_lower_bound = lower_bound
    layer.A_log = nn.Parameter(torch.zeros(1, 1, 1, 1, dtype=torch.float32))
    layer.dt_bias = nn.Parameter(torch.zeros(head_dim, dtype=torch.float32))
    return layer


def _patch_kimi_kda_construction(monkeypatch: pytest.MonkeyPatch) -> dict[str, tuple[int, int, dict[str, Any]]]:
    import vllm_ascend.ops.kimi_kda as kimi_kda

    def fake_base_init(self, config, vllm_config, prefix):
        nn.Module.__init__(self)
        self.prefix = prefix
        self.tp_size = 1
        self.tp_rank = 0
        self.layer_idx = 0
        self.hidden_size = config.hidden_size
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps
        self.model_config = SimpleNamespace(dtype=torch.bfloat16)
        self.cache_config = SimpleNamespace(mamba_cache_dtype="auto")
        self.quant_config = vllm_config.quant_config
        self.speculative_config = None
        self.num_spec = 0

    created: dict[str, tuple[int, int, dict[str, Any]]] = {}

    def fake_parallel_linear(*args, **kwargs):
        if args:
            input_size, output_size = args
        else:
            input_size = kwargs.pop("input_size")
            output_size = kwargs.pop("output_size")
        prefix = kwargs["prefix"]
        created[prefix] = (input_size, output_size, dict(kwargs))
        return nn.Linear(input_size, output_size, bias=kwargs.get("bias", False))

    class FakeFusedRMSNormGated(nn.Module):
        def __init__(self, hidden_size, *, eps, activation):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(hidden_size))
            self.eps = eps
            self.activation = activation

    def fake_set_weight_attrs(weight, attrs):
        for name, value in attrs.items():
            setattr(weight, name, value)

    monkeypatch.setattr(GatedDeltaNetAttention, "__init__", fake_base_init)
    monkeypatch.setattr(kimi_kda, "ColumnParallelLinear", fake_parallel_linear)
    monkeypatch.setattr(kimi_kda, "_KimiK3TPColumnParallelLinear", fake_parallel_linear)
    monkeypatch.setattr(kimi_kda, "ReplicatedLinear", fake_parallel_linear)
    monkeypatch.setattr(kimi_kda, "RowParallelLinear", fake_parallel_linear)
    monkeypatch.setattr(kimi_kda, "FusedRMSNormGated", FakeFusedRMSNormGated)
    monkeypatch.setattr(kimi_kda, "set_weight_attrs", fake_set_weight_attrs)
    monkeypatch.setattr(kimi_kda, "get_tensor_model_parallel_rank", lambda: 0)
    return created


@pytest.mark.parametrize(
    ("dim_first", "expected_conv_shape"),
    [(False, (6, 48)), (True, (48, 6))],
)
def test_kimi_kda_speculative_cache_shape_includes_lookahead(
    monkeypatch: pytest.MonkeyPatch,
    dim_first: bool,
    expected_conv_shape: tuple[int, int],
):
    import vllm_ascend.ops.kimi_kda_state as state_shape

    monkeypatch.setattr(state_shape, "is_conv_state_dim_first", lambda: dim_first)

    actual = kimi_kda_state_shape(
        tp_world_size=2,
        num_heads=4,
        head_dim=8,
        conv_kernel_size=4,
        num_spec=3,
    )
    layer = object.__new__(AscendKimiGatedDeltaNetAttention)
    layer.tp_size = 2
    layer.num_heads = 4
    layer.head_dim = 8
    layer.conv_size = 4
    layer.num_spec = 3

    assert actual == (expected_conv_shape, (2, 8, 8))
    assert layer.get_state_shape() == actual


def test_kimi_k3_constructs_exact_full_rank_gate_topology(monkeypatch: pytest.MonkeyPatch):
    created = _patch_kimi_kda_construction(monkeypatch)
    prefix = "model.layers.0.self_attn"
    config = KimiK3TextConfig(
        hidden_size=16,
        rms_norm_eps=1e-6,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [2],
            "head_dim": 4,
            "num_heads": 3,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
    )
    compilation_config = SimpleNamespace(static_forward_context={})
    layer = AscendKimiGatedDeltaNetAttention(
        config,
        SimpleNamespace(
            quant_config="quant-config",
            compilation_config=compilation_config,
        ),
        prefix=prefix,
        input_layout=AscendTokenLayout.TOKEN_SHARDED,
    )

    assert type(layer) is AscendKimiGatedDeltaNetAttention
    assert isinstance(layer, GatedDeltaNetAttention)
    assert not hasattr(layer, "g_a_proj")
    assert not hasattr(layer, "g_b_proj")
    assert isinstance(layer.g_proj, nn.Linear)
    assert created[f"{prefix}.g_proj"] == (
        16,
        12,
        {
            "bias": False,
            "quant_config": "quant-config",
            "prefix": f"{prefix}.g_proj",
        },
    )
    assert f"{prefix}.g_a_proj" not in created
    assert f"{prefix}.g_b_proj" not in created
    assert layer.A_log.shape == (1, 1, 3, 1)
    assert layer.gate_lower_bound == -5.0
    assert layer.input_layout is AscendTokenLayout.TOKEN_SHARDED
    assert layer.o_norm.eps == 1e-6
    assert layer.o_norm.activation == "sigmoid"
    assert compilation_config.static_forward_context[prefix] is layer
    layer.A_log.weight_loader(layer.A_log, torch.arange(128, dtype=torch.float32))
    torch.testing.assert_close(
        layer.A_log.flatten(),
        torch.arange(3, dtype=torch.float32),
    )


def test_kimi_k3_a_log_loader_trims_padding_then_tp8_shards(monkeypatch):
    import vllm_ascend.ops.kimi_kda as kimi_kda

    # Non-zero tail values make loading any of the 32 padding heads visible.
    loaded_weight = torch.arange(128, dtype=torch.float32)
    for rank in range(8):
        monkeypatch.setattr(
            kimi_kda,
            "get_tensor_model_parallel_rank",
            lambda rank=rank: rank,
        )
        param = nn.Parameter(torch.empty(1, 1, 12, 1))

        _load_kimi_k3_a_log(param, loaded_weight, num_heads=96)

        expected = torch.arange(rank * 12, (rank + 1) * 12, dtype=torch.float32)
        torch.testing.assert_close(param.flatten(), expected)


@pytest.mark.parametrize(
    ("case", "input_layout", "input_rows", "expected_gather_label"),
    [
        ("vl_first_layer", AscendTokenLayout.GLOBAL, 4, False),
        ("vl_later_layer", AscendTokenLayout.TOKEN_SHARDED, 2, True),
        ("text_only_model", AscendTokenLayout.TOKEN_SHARDED, 2, True),
    ],
)
def test_kimi_kda_flashcomm_gathers_once_before_projections(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    input_layout: AscendTokenLayout,
    input_rows: int,
    expected_gather_label: bool,
):
    """KDA must run its stateful core over global, not SP-local, tokens."""
    del case
    import vllm_ascend.ops.kimi_kda as kimi_kda

    class TupleProjection(nn.Module):
        def __init__(self, out_features: int, name: str) -> None:
            super().__init__()
            self.out_features = out_features
            self.name = name

        def forward(self, hidden_states: torch.Tensor):
            projection_rows[self.name] = hidden_states.shape[0]
            values = hidden_states[:, :1].expand(-1, self.out_features).clone()
            return values, None

    class GateNorm(nn.Module):
        def forward(self, core_attn_out: torch.Tensor, output_gate: torch.Tensor):
            norm_shapes.append((tuple(core_attn_out.shape), tuple(output_gate.shape)))
            return core_attn_out

    class LocalOutputProjection(nn.Module):
        def forward(self, core_attn_out: torch.Tensor):
            o_proj_input_rows.append(core_attn_out.shape[0])
            # Model FlashComm's row-parallel reduce-scatter: the stateful KDA
            # core sees four global rows while o_proj returns two local rows.
            local = core_attn_out[:2].repeat(1, 2)
            return local, None

    layer = _bare_kimi_kda()
    layer.input_layout = input_layout
    layer.local_num_heads = 1
    layer.prefix = "model.layers.0.self_attn"

    projection_rows: dict[str, int] = {}
    layer.q_proj = TupleProjection(2, "q")
    layer.k_proj = TupleProjection(2, "k")
    layer.v_proj = TupleProjection(2, "v")
    layer.b_proj = TupleProjection(1, "beta")
    layer.f_a_proj = TupleProjection(2, "f_a")
    layer.f_b_proj = TupleProjection(2, "f_b")
    layer.g_proj = TupleProjection(2, "g")
    norm_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    layer.o_norm = GateNorm()
    o_proj_input_rows: list[int] = []
    layer.o_proj = LocalOutputProjection()

    gather_calls: list[tuple[int, bool, bool]] = []

    def fake_gather(hidden_states: torch.Tensor, label: bool):
        gather_calls.append((hidden_states.shape[0], label, hidden_states.is_contiguous()))
        if label:
            return torch.cat((hidden_states, hidden_states + 10), dim=0)
        return hidden_states

    core_shapes: list[tuple[int, ...]] = []

    def fake_kda_attention(q, k, v, raw_gate, beta, core_attn_out, prefix):
        del k, v, raw_gate, beta
        assert prefix == layer.prefix
        core_shapes.append(tuple(core_attn_out.shape))
        core_attn_out.copy_(q.reshape(1, q.shape[0], 1, 2))

    monkeypatch.setattr(
        kimi_kda.torch.ops.vllm,
        "maybe_all_gather_and_maybe_unpad",
        fake_gather,
        raising=False,
    )
    monkeypatch.setattr(
        kimi_kda.torch.ops.vllm,
        "kda_attention",
        fake_kda_attention,
        raising=False,
    )

    hidden_states = torch.arange(input_rows * 4, dtype=torch.float32).reshape(input_rows, 4)
    output = torch.full((2, 4), torch.nan)
    layer(hidden_states, torch.zeros(input_rows, dtype=torch.long), output)

    assert gather_calls == [(input_rows, expected_gather_label, True)]
    assert set(projection_rows.values()) == {4}
    assert core_shapes == [(1, 4, 1, 2)]
    assert norm_shapes == [((1, 4, 1, 2), (4, 1, 2))]
    assert o_proj_input_rows == [4]
    expected_output = torch.tensor([[0.0, 0.0, 0.0, 0.0], [4.0, 4.0, 4.0, 4.0]])
    torch.testing.assert_close(output, expected_output)


def test_recurrent_path_uses_ascendc_with_safe_gate_and_spec_metadata(monkeypatch: pytest.MonkeyPatch):
    import vllm_ascend.ops.kimi_kda as kimi_kda

    layer = _bare_kimi_kda()
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_recurrent(*args, **kwargs):
        calls.append((args, kwargs))
        return args[0].clone()

    monkeypatch.setattr(
        kimi_kda.torch.ops._C_ascend,
        "recurrent_kda",
        fake_recurrent,
        raising=False,
    )

    q = torch.randn(1, 3, 1, 2)
    raw_gate = torch.randn(1, 3, 1, 2)
    beta = torch.rand(1, 3, 1)
    state = torch.randn(8, 1, 2, 2)
    cu_seqlens = torch.tensor([0, 2, 3], dtype=torch.int32)
    state_indices = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    accepted = torch.tensor([2, 1], dtype=torch.int32)

    out = layer._run_recurrent(
        q,
        q,
        q,
        raw_gate,
        beta,
        state,
        cu_seqlens,
        state_indices,
        num_accepted_tokens=accepted,
    )

    assert torch.equal(out, q)
    assert len(calls) == 1
    args, kwargs = calls[0]
    for actual, expected in zip(args[:5], (q, q, q, raw_gate, beta)):
        assert torch.equal(actual, expected)
    assert args[5] is state
    assert args[6] is cu_seqlens
    assert args[7] is state_indices
    assert args[8].shape == (1,)
    assert args[9].shape == (2,)
    assert kwargs == {
        "num_accepted_tokens": accepted,
        "scale": 2**-0.5,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "use_beta_sigmoid_in_kernel": False,
        "allow_neg_eigval": False,
        "safe_gate": True,
        "lower_bound": -5.0,
    }


def test_prefill_compacts_empty_rows_and_transposes_cache_boundary(monkeypatch: pytest.MonkeyPatch):
    import vllm_ascend.ops.kimi_kda as kimi_kda

    layer = _bare_kimi_kda()
    monkeypatch.setattr(kimi_kda, "get_pcp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(kimi_kda, "l2norm_fwd", lambda x: x)

    def fake_clear(states, has_initial_state):
        states[~has_initial_state] = 0

    monkeypatch.setattr(kimi_kda, "clear_ssm_states", fake_clear)

    calls: dict[str, Any] = {}

    def fake_gate_cumsum(raw_gate, chunk_size, **kwargs):
        calls["gate"] = (raw_gate, chunk_size, kwargs)
        return raw_gate.float()

    def fake_chunk_kda(q, k, v, gate, beta, scale, chunk_size, **kwargs):
        calls["chunk"] = (q, k, v, gate, beta, scale, chunk_size, kwargs)
        initial_state = kwargs["initial_state"]
        return v.clone(), initial_state + 10

    monkeypatch.setattr(torch.ops._C_ascend, "kda_gate_cumsum", fake_gate_cumsum, raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "chunk_kda_fwd", fake_chunk_kda, raising=False)

    q = torch.randn(1, 5, 1, 2)
    raw_gate = torch.randn(1, 5, 1, 2)
    beta = torch.rand(1, 5, 1)
    recurrent_state = torch.zeros(4, 1, 2, 2)
    recurrent_state[1, 0] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    recurrent_state[2, 0] = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    untouched = recurrent_state[0].clone()

    cu_seqlens_kern = (0, 2, 5)
    prebuilt = SimpleNamespace(
        cu_seqlens_host=(0, 2, 2, 5),
        cu_seqlens_kern=cu_seqlens_kern,
        keep_meta=torch.tensor([True, False, True]),
        chunk_indices_chunk64_host=(0, 0, 1, 0),
    )
    out = layer._run_prefill(
        q,
        q,
        q,
        raw_gate,
        beta,
        recurrent_state,
        torch.tensor([1, 0, 2]),
        torch.tensor([True, True, False]),
        prebuilt,
    )

    assert torch.equal(out, q)
    gate_kwargs = calls["gate"][2]
    assert gate_kwargs["cu_seqlens"] == (0, 2, 5)
    assert gate_kwargs["use_gate_in_kernel"] is True
    assert gate_kwargs["safe_gate"] is True
    assert gate_kwargs["lower_bound"] == -5.0

    chunk_kwargs = calls["chunk"][7]
    expected_initial_kv = torch.stack(
        (
            torch.tensor([[[1.0, 3.0], [2.0, 4.0]]]),
            torch.zeros(1, 2, 2),
        )
    )
    assert torch.equal(chunk_kwargs["initial_state"], expected_initial_kv)
    assert chunk_kwargs["cu_seqlens"] == (0, 2, 5)
    assert chunk_kwargs["chunk_indices"] == (0, 0, 1, 0)

    assert torch.equal(recurrent_state[0], untouched)
    assert torch.equal(recurrent_state[1], (expected_initial_kv[0] + 10).transpose(-1, -2))
    assert torch.equal(recurrent_state[2], (expected_initial_kv[1] + 10).transpose(-1, -2))


def test_chunk_metadata_uses_kimi_linear_attention_head_count(monkeypatch: pytest.MonkeyPatch):
    import vllm_ascend.ops.gdn_attn_builder as builder_module

    chunk_sizes = []

    def fake_chunk_indices(cu_seqlens, chunk_size):
        chunk_sizes.append(chunk_size)
        return torch.tensor([[0, 0]], dtype=cu_seqlens.dtype)

    monkeypatch.setattr(builder_module, "prepare_chunk_indices", fake_chunk_indices)
    monkeypatch.setattr(
        builder_module,
        "prepare_chunk_offsets",
        lambda cu_seqlens, chunk_size: torch.tensor([0, 1], dtype=cu_seqlens.dtype),
    )
    monkeypatch.setattr(
        builder_module,
        "prepare_update_chunk_offsets",
        lambda cu_seqlens, chunk_size: torch.tensor([0, 1], dtype=cu_seqlens.dtype),
    )
    monkeypatch.setattr(
        builder_module,
        "prepare_final_chunk_indices",
        lambda cu_seqlens, chunk_size: torch.tensor([0], dtype=cu_seqlens.dtype),
    )

    model_config = SimpleNamespace(
        hf_text_config=KimiK3TextConfig(linear_attn_config={"num_heads": 96}),
        get_num_attention_heads=lambda parallel_config: 8,
    )
    builder = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=model_config,
            parallel_config=SimpleNamespace(tensor_parallel_size=2),
        )
    )
    builder_module._build_non_spec_chunked_prefill_metadata(
        builder,
        torch.tensor([0, 64], dtype=torch.int32),
        torch.device("cpu"),
    )

    # 96 global KDA heads / TP2 => 48 local heads.  The cumsum working-set
    # calculation therefore rounds 85 chunks up to 128.
    assert chunk_sizes == [64, 1216, 128]
