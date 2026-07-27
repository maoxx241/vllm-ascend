# SPDX-License-Identifier: Apache-2.0
"""Focused Kimi K3 recurrent-KDA acceptance coverage.

These tests cover the three contracts that are specific to K3 decode:
single-token model wiring, speculative state-slot indirection, and graph
padding. Broader operator accuracy lives in the generic recurrent-KDA tests.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

HEADS_PER_TP16_RANK = 6
KDA_HEAD_DIM = 128
GATE_LOWER_BOUND = -5.0


def _state_slot(
    state_indices: torch.Tensor,
    sequence_index: int,
    sequence_start: int,
    token_index: int,
) -> int:
    if state_indices.ndim == 1:
        return int(state_indices[token_index].item())
    return int(state_indices[sequence_index, token_index - sequence_start].item())


def _k3_recurrent_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    cu_seqlens: Sequence[int],
    state_indices: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Small CPU oracle for K3's fixed recurrent-kernel options."""
    q_flat = F.normalize(q.reshape(-1, q.shape[-2], q.shape[-1]).float(), p=2, dim=-1)
    k_flat = F.normalize(k.reshape_as(q_flat).float(), p=2, dim=-1)
    v_flat = v.reshape_as(q_flat).float()
    gate_input = raw_gate.reshape_as(q_flat).float() + dt_bias.float().reshape(
        1,
        q.shape[-2],
        q.shape[-1],
    )
    gate = GATE_LOWER_BOUND * torch.sigmoid(torch.exp(a_log.float()).reshape(1, q.shape[-2], 1) * gate_input)
    gate_decay = torch.exp(gate)
    beta_flat = beta.reshape(-1, beta.shape[-1]).float()

    q_flat *= q.shape[-1] ** -0.5
    state = initial_state.float().clone()
    output = torch.zeros_like(v_flat)

    for sequence_index, (start, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        if start == end:
            continue
        accepted = 1 if num_accepted_tokens is None else int(num_accepted_tokens[sequence_index].item())
        initial_slot = _state_slot(
            state_indices,
            sequence_index,
            start,
            start + accepted - 1,
        )
        for head in range(q.shape[-2]):
            current_state = state[initial_slot, head].clone()
            for token in range(start, end):
                current_state *= gate_decay[token, head].unsqueeze(0)
                delta = v_flat[token, head] - torch.mv(
                    current_state,
                    k_flat[token, head],
                )
                current_state += torch.outer(
                    delta * beta_flat[token, head],
                    k_flat[token, head],
                )
                output[token, head] = torch.mv(
                    current_state,
                    q_flat[token, head],
                )
                output_slot = _state_slot(
                    state_indices,
                    sequence_index,
                    start,
                    token,
                )
                state[output_slot, head] = current_state

    return output.reshape_as(v).to(v.dtype), state.to(initial_state.dtype)


def _run_ascendc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    num_accepted_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops._C_ascend.recurrent_kda(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        raw_gate.contiguous(),
        beta.contiguous(),
        state,
        cu_seqlens,
        state_indices,
        a_log.contiguous(),
        dt_bias.contiguous(),
        num_accepted_tokens=num_accepted_tokens,
        scale=KDA_HEAD_DIM**-0.5,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False,
        safe_gate=True,
        lower_bound=GATE_LOWER_BOUND,
    )


@torch.inference_mode()
def test_kimi_k3_tp16_recurrent_kda_single_token_decode_wrapper():
    """The model wrapper must update only the cache slots owned by the batch."""
    from torch import nn

    from vllm_ascend.ops.kimi_kda import AscendKimiGatedDeltaNetAttention

    torch.manual_seed(20260723)
    device = torch.device("npu")
    batch = 4
    q = torch.randn(
        1,
        batch,
        HEADS_PER_TP16_RANK,
        KDA_HEAD_DIM,
        dtype=torch.bfloat16,
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    raw_gate = torch.randn_like(q) * 0.25
    beta = torch.rand(1, batch, HEADS_PER_TP16_RANK, dtype=torch.float32).sigmoid()
    initial_state = (
        torch.randn(
            17,
            HEADS_PER_TP16_RANK,
            KDA_HEAD_DIM,
            KDA_HEAD_DIM,
            dtype=torch.float32,
        )
        * 0.01
    )
    state_indices = torch.tensor([9, 2, 15, 4], dtype=torch.int64)
    cu_seqlens = torch.arange(batch + 1, dtype=torch.int32)
    a_log = torch.randn(HEADS_PER_TP16_RANK, dtype=torch.float32) * 0.05
    dt_bias = torch.randn(HEADS_PER_TP16_RANK, KDA_HEAD_DIM, dtype=torch.float32) * 0.05
    expected_output, expected_state = _k3_recurrent_reference(
        q,
        k,
        v,
        raw_gate,
        beta,
        initial_state,
        cu_seqlens=cu_seqlens.tolist(),
        state_indices=state_indices,
        a_log=a_log,
        dt_bias=dt_bias,
    )

    layer = object.__new__(AscendKimiGatedDeltaNetAttention)
    nn.Module.__init__(layer)
    layer.head_dim = KDA_HEAD_DIM
    layer.gate_lower_bound = GATE_LOWER_BOUND
    layer.A_log = nn.Parameter(a_log.reshape(1, 1, HEADS_PER_TP16_RANK, 1).to(device))
    layer.dt_bias = nn.Parameter(dt_bias.to(device))
    actual_state = initial_state.to(device)
    actual_output = layer._run_recurrent(
        q.to(device),
        k.to(device),
        v.to(device),
        raw_gate.to(device),
        beta.to(device),
        actual_state,
        cu_seqlens.to(device),
        state_indices.to(device),
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        actual_output.cpu(),
        expected_output,
        rtol=0.02,
        atol=0.02,
    )
    torch.testing.assert_close(
        actual_state.cpu(),
        expected_state,
        rtol=0.02,
        atol=0.02,
    )
    active_slots = set(state_indices.tolist())
    untouched = [slot for slot in range(initial_state.shape[0]) if slot not in active_slots]
    torch.testing.assert_close(
        actual_state.cpu()[untouched],
        initial_state[untouched],
        rtol=0,
        atol=0,
    )


@torch.inference_mode()
def test_kimi_k3_tp16_recurrent_kda_safe_gate_and_speculative_state_slots():
    """Accepted-token metadata must select non-contiguous speculative slots."""
    torch.manual_seed(20260721)
    device = torch.device("npu")
    cu_seqlens = torch.tensor([0, 1, 4, 4, 5], dtype=torch.int32)
    tokens = int(cu_seqlens[-1])
    shape = (1, tokens, HEADS_PER_TP16_RANK, KDA_HEAD_DIM)
    q = torch.randn(shape, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    raw_gate = torch.randn(shape, dtype=torch.float32) * 0.25
    beta = torch.rand(
        1,
        tokens,
        HEADS_PER_TP16_RANK,
        dtype=torch.float32,
    ).sigmoid()
    initial_state = (
        torch.randn(
            13,
            HEADS_PER_TP16_RANK,
            KDA_HEAD_DIM,
            KDA_HEAD_DIM,
            dtype=torch.float32,
        )
        * 0.01
    )
    state_indices = torch.tensor(
        [
            [8, 8, 8, 8, 8],
            [2, 11, 6, 2, 2],
            [4, 4, 4, 4, 4],
            [1, 3, 5, 7, 9],
        ],
        dtype=torch.int64,
    )
    accepted = torch.tensor([1, 2, 0, 1], dtype=torch.int64)
    a_log = torch.randn(HEADS_PER_TP16_RANK, dtype=torch.float32) * 0.05
    dt_bias = torch.randn(HEADS_PER_TP16_RANK, KDA_HEAD_DIM, dtype=torch.float32) * 0.05
    expected_output, expected_state = _k3_recurrent_reference(
        q,
        k,
        v,
        raw_gate,
        beta,
        initial_state,
        cu_seqlens=cu_seqlens.tolist(),
        state_indices=state_indices,
        a_log=a_log,
        dt_bias=dt_bias,
        num_accepted_tokens=accepted,
    )

    actual_state = initial_state.to(device)
    actual_output = _run_ascendc(
        q.to(device),
        k.to(device),
        v.to(device),
        raw_gate.to(device),
        beta.to(device),
        actual_state,
        cu_seqlens.to(device),
        state_indices.to(device),
        a_log.to(device),
        dt_bias.to(device),
        num_accepted_tokens=accepted.to(device),
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        actual_output.cpu(),
        expected_output,
        rtol=0.02,
        atol=0.02,
    )
    torch.testing.assert_close(
        actual_state.cpu(),
        expected_state,
        rtol=0.02,
        atol=0.02,
    )
    untouched = [slot for slot in range(initial_state.shape[0]) if slot not in {1, 2, 6, 8, 11}]
    torch.testing.assert_close(
        actual_state.cpu()[untouched],
        initial_state[untouched],
        rtol=0,
        atol=0,
    )


@torch.inference_mode()
def test_kimi_k3_tp16_recurrent_kda_full_decode_graph_padding():
    """Zero-length graph rows must not update padding cache slots."""
    torch.manual_seed(20260723)
    device = torch.device("npu")
    graph_tokens = 16
    shape = (1, graph_tokens, HEADS_PER_TP16_RANK, KDA_HEAD_DIM)
    q = torch.zeros(shape, dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    raw_gate = torch.zeros_like(q)
    beta = torch.zeros(
        1,
        graph_tokens,
        HEADS_PER_TP16_RANK,
        dtype=torch.float32,
    )
    q[:, :1].normal_()
    k[:, :1].normal_()
    v[:, :1].normal_()
    raw_gate[:, :1].normal_(std=0.25)
    beta[:, :1].uniform_().sigmoid_()
    initial_state = (
        torch.randn(
            17,
            HEADS_PER_TP16_RANK,
            KDA_HEAD_DIM,
            KDA_HEAD_DIM,
            dtype=torch.float32,
        )
        * 0.01
    )
    cu_seqlens = torch.tensor(
        [0, 1, *([1] * (graph_tokens - 1))],
        dtype=torch.int32,
    )
    state_indices = torch.tensor(
        [2, *([0] * (graph_tokens - 1))],
        dtype=torch.int64,
    )
    a_log = torch.linspace(
        -0.43,
        1.26,
        HEADS_PER_TP16_RANK,
        dtype=torch.float32,
    )
    dt_bias = torch.linspace(
        -9.0,
        -1.47,
        HEADS_PER_TP16_RANK * KDA_HEAD_DIM,
        dtype=torch.float32,
    ).reshape(HEADS_PER_TP16_RANK, KDA_HEAD_DIM)
    expected_output, expected_state = _k3_recurrent_reference(
        q[:, :1],
        k[:, :1],
        v[:, :1],
        raw_gate[:, :1],
        beta[:, :1],
        initial_state,
        cu_seqlens=[0, 1],
        state_indices=state_indices[:1],
        a_log=a_log,
        dt_bias=dt_bias,
    )

    actual_state = initial_state.to(device)
    actual_output = _run_ascendc(
        q.to(device),
        k.to(device),
        v.to(device),
        raw_gate.to(device),
        beta.to(device),
        actual_state,
        cu_seqlens.to(device),
        state_indices.to(device),
        a_log.to(device),
        dt_bias.to(device),
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        actual_output[:, :1].cpu(),
        expected_output,
        rtol=0.02,
        atol=0.02,
    )
    torch.testing.assert_close(
        actual_state.cpu(),
        expected_state,
        rtol=0.02,
        atol=0.02,
    )
