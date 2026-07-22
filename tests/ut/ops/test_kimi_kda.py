# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn
from vllm.model_executor.custom_op import op_registry_oot
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import KimiGatedDeltaNetAttention

from vllm_ascend.ops.kimi_kda import (
    AscendKimiGatedDeltaNetAttention,
    _load_kimi_k3_a_log,
)
from vllm_ascend.ops.kimi_kda_state import kimi_kda_state_shape
from vllm_ascend.ops.triton.kda.kda import fused_kda_gate
from vllm_ascend.transformers_utils.configs.kimi_k3 import KimiK3TextConfig
from vllm_ascend.utils import uses_global_inputs_embeds


def _bare_kimi_kda(*, head_dim: int = 2, lower_bound: float | None = -5.0):
    layer = object.__new__(AscendKimiGatedDeltaNetAttention)
    nn.Module.__init__(layer)
    layer.head_dim = head_dim
    layer.gate_lower_bound = lower_bound
    layer.A_log = nn.Parameter(torch.zeros(1, 1, 1, 1, dtype=torch.float32))
    layer.dt_bias = nn.Parameter(torch.zeros(head_dim, dtype=torch.float32))
    return layer


def test_kimi_kda_is_registered_by_upstream_class_name():
    assert op_registry_oot["KimiGatedDeltaNetAttention"] is AscendKimiGatedDeltaNetAttention


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


def test_kimi_k3_full_rank_gate_replaces_upstream_low_rank_modules(monkeypatch: pytest.MonkeyPatch):
    import vllm_ascend.ops.kimi_kda as kimi_kda

    def fake_upstream_init(self, config, vllm_config, prefix):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.head_dim = config.linear_attn_config["head_dim"]
        self.num_heads = config.linear_attn_config["num_heads"]
        self.quant_config = vllm_config.quant_config
        self.g_a_proj = nn.Identity()
        self.g_b_proj = nn.Identity()
        self.o_norm = SimpleNamespace(eps=1e-5)
        self.A_log = nn.Parameter(torch.empty(1, 1, self.num_heads, 1, dtype=torch.float32))
        self.A_log.weight_loader = lambda param, loaded: None

    created: dict[str, Any] = {}

    def fake_column_parallel(input_size, output_size, **kwargs):
        created.update(input_size=input_size, output_size=output_size, kwargs=kwargs)
        return nn.Linear(input_size, output_size, bias=False)

    monkeypatch.setattr(KimiGatedDeltaNetAttention, "__init__", fake_upstream_init)
    monkeypatch.setattr(kimi_kda, "ColumnParallelLinear", fake_column_parallel)
    monkeypatch.setattr(kimi_kda, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(kimi_kda, "uses_global_inputs_embeds", lambda vllm_config, modality: False)

    config = KimiK3TextConfig(
        hidden_size=16,
        rms_norm_eps=1e-6,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [2],
            "head_dim": 4,
            "num_heads": 3,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
    )
    # Production constructs the upstream name. PluggableLayer must replace it
    # with the Ascend implementation before initialization.
    layer = KimiGatedDeltaNetAttention(
        config,
        SimpleNamespace(quant_config="quant-config"),
        prefix="model.layers.0.self_attn",
    )

    assert type(layer) is AscendKimiGatedDeltaNetAttention
    assert not hasattr(layer, "g_a_proj")
    assert not hasattr(layer, "g_b_proj")
    assert isinstance(layer.g_proj, nn.Linear)
    assert created == {
        "input_size": 16,
        "output_size": 12,
        "kwargs": {
            "bias": False,
            "quant_config": "quant-config",
            "prefix": "model.layers.0.self_attn.g_proj",
        },
    }
    assert layer.gate_lower_bound == -5.0
    assert layer.is_vl_first_layer is False
    assert layer.o_norm.eps == 1e-6
    layer.A_log.weight_loader(layer.A_log, torch.arange(128, dtype=torch.float32))
    torch.testing.assert_close(
        layer.A_log.flatten(),
        torch.arange(3, dtype=torch.float32),
    )


@pytest.mark.parametrize(
    ("uses_global_embeds", "prefix", "expected"),
    [
        (True, "model.layers.0.self_attn", True),
        (True, "model.layers.1.self_attn", False),
        (False, "model.layers.0.self_attn", False),
    ],
)
def test_kimi_kda_identifies_only_global_input_layer_zero_as_already_global(
    monkeypatch: pytest.MonkeyPatch,
    uses_global_embeds: bool,
    prefix: str,
    expected: bool,
):
    import vllm_ascend.ops.kimi_kda as kimi_kda

    def fake_upstream_init(self, config, vllm_config, prefix):
        del vllm_config, prefix
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.head_dim = config.linear_attn_config["head_dim"]
        self.num_heads = config.linear_attn_config["num_heads"]
        self.quant_config = None
        self.o_norm = SimpleNamespace(eps=1e-5)
        self.A_log = nn.Parameter(torch.empty(1, 1, self.num_heads, 1))

    monkeypatch.setattr(KimiGatedDeltaNetAttention, "__init__", fake_upstream_init)
    monkeypatch.setattr(
        kimi_kda,
        "uses_global_inputs_embeds",
        lambda vllm_config, modality: uses_global_embeds,
    )
    config = KimiK3TextConfig(
        hidden_size=4,
        rms_norm_eps=1e-6,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [2],
            "head_dim": 2,
            "num_heads": 1,
            "use_full_rank_gate": False,
        },
    )

    layer = AscendKimiGatedDeltaNetAttention(
        config,
        SimpleNamespace(quant_config=None),
        prefix=prefix,
    )

    assert layer.is_vl_first_layer is expected


@pytest.mark.parametrize(
    ("is_vl", "limit", "enable_mm_embeds", "enable_prompt_embeds", "expected"),
    [
        (True, 1, False, False, True),
        (True, 0, False, False, False),
        (True, 0, True, False, True),
        (False, 0, False, True, True),
        (False, 0, False, False, False),
    ],
)
def test_kimi_kda_global_input_layout_matches_runner_input_path(
    monkeypatch: pytest.MonkeyPatch,
    is_vl: bool,
    limit: int,
    enable_mm_embeds: bool,
    enable_prompt_embeds: bool,
    expected: bool,
):
    import vllm_ascend.utils as ascend_utils

    monkeypatch.setattr(ascend_utils, "is_vl_model", lambda vllm_config: is_vl)
    mm_config = SimpleNamespace(
        enable_mm_embeds=enable_mm_embeds,
        get_limit_per_prompt=lambda modality: limit,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            enable_prompt_embeds=enable_prompt_embeds,
            multimodal_config=mm_config,
        )
    )

    assert uses_global_inputs_embeds(vllm_config, "vision_chunk") is expected


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


def test_kimi_k3_a_log_loader_accepts_converted_4d_weight(monkeypatch):
    import vllm_ascend.ops.kimi_kda as kimi_kda

    monkeypatch.setattr(
        kimi_kda,
        "get_tensor_model_parallel_rank",
        lambda: 6,
    )
    param = nn.Parameter(torch.empty(1, 1, 12, 1))
    loaded_weight = torch.arange(128, dtype=torch.float32).reshape(1, 1, 128, 1)

    _load_kimi_k3_a_log(param, loaded_weight, num_heads=96)

    torch.testing.assert_close(
        param.flatten(),
        torch.arange(72, 84, dtype=torch.float32),
    )


@pytest.mark.parametrize(
    ("case", "is_vl_first_layer", "input_rows", "expected_gather_label"),
    [
        ("vl_first_layer", True, 4, False),
        ("vl_later_layer", False, 2, True),
        ("text_only_model", False, 2, True),
    ],
)
def test_kimi_kda_flashcomm_gathers_once_before_projections(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    is_vl_first_layer: bool,
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
    layer.is_vl_first_layer = is_vl_first_layer
    layer.local_num_heads = 1
    layer.prefix = "model.layers.0.self_attn"
    layer.use_full_rank_gate = True

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


def test_fused_kda_gate_rejects_invalid_safe_lower_bound_before_launch():
    with pytest.raises(ValueError, match="lower_bound must be in"):
        fused_kda_gate(
            torch.zeros(1, 2),
            torch.zeros(1),
            2,
            safe_gate=True,
            lower_bound=-6.0,
        )


def test_prefill_requires_pr141_ascendc_schemas(monkeypatch: pytest.MonkeyPatch):
    import vllm_ascend.ops.kimi_kda as kimi_kda

    monkeypatch.setattr(kimi_kda.torch.ops, "_C_ascend", SimpleNamespace(), raising=False)
    with pytest.raises(RuntimeError, match=r"PR141 AscendC operators.*kda_gate_cumsum.*chunk_kda_fwd"):
        kimi_kda._require_ascendc_prefill_ops()


def test_ascendc_runtime_error_propagates(monkeypatch: pytest.MonkeyPatch):
    import vllm_ascend.ops.kimi_kda as kimi_kda

    layer = _bare_kimi_kda()
    monkeypatch.setattr(kimi_kda, "get_pcp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(kimi_kda, "clear_ssm_states", lambda states, has_initial: None)
    monkeypatch.setattr(kimi_kda, "l2norm_fwd", lambda x: x)

    def fail_gate_cumsum(*args, **kwargs):
        raise RuntimeError("ascendc sentinel failure")

    monkeypatch.setattr(torch.ops._C_ascend, "kda_gate_cumsum", fail_gate_cumsum, raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "chunk_kda_fwd", lambda *args, **kwargs: None, raising=False)
    prebuilt = SimpleNamespace(
        cu_seqlens_host=(0, 1),
        cu_seqlens_kern=None,
        keep_meta=None,
        chunk_indices_chunk64_host=(0, 0),
    )
    q = torch.randn(1, 1, 1, 2)
    with pytest.raises(RuntimeError, match="ascendc sentinel failure"):
        layer._run_prefill(
            q,
            q,
            q,
            q,
            torch.rand(1, 1, 1),
            torch.zeros(2, 1, 2, 2),
            torch.tensor([1]),
            torch.tensor([True]),
            prebuilt,
        )


def test_recurrent_path_threads_safe_gate_and_spec_metadata(monkeypatch: pytest.MonkeyPatch):
    import vllm_ascend.ops.kimi_kda as kimi_kda

    layer = _bare_kimi_kda()
    calls: dict[str, Any] = {}

    def fake_gate(g, a_log, head_dim, **kwargs):
        calls["gate"] = (g, a_log, head_dim, kwargs)
        return torch.zeros(*g.shape[:-1], 1, head_dim, dtype=torch.float32)

    def fake_recurrent(**kwargs):
        calls["recurrent"] = kwargs
        return kwargs["q"].clone(), kwargs["initial_state"]

    monkeypatch.setattr(kimi_kda, "fused_kda_gate", fake_gate)
    monkeypatch.setattr(kimi_kda, "fused_recurrent_kda", fake_recurrent)

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
    gate_kwargs = calls["gate"][3]
    assert gate_kwargs["safe_gate"] is True
    assert gate_kwargs["lower_bound"] == -5.0
    recurrent_kwargs = calls["recurrent"]
    assert recurrent_kwargs["num_accepted_tokens"] is accepted
    assert recurrent_kwargs["ssm_state_indices"] is state_indices


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

    cu_seqlens_kern = torch.tensor([0, 2, 5], dtype=torch.int32)
    prebuilt = SimpleNamespace(
        cu_seqlens_host=(0, 2, 2, 5),
        # A non-empty Tensor must be selected without evaluating its truth
        # value, which is ambiguous for multi-element tensors.
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
        hf_text_config=SimpleNamespace(linear_attn_config={"num_heads": 96}),
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
