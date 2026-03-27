# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.ops.triton.fla import chunk_o, chunk_o_update


class _FakeKernel:
    def __init__(self):
        self.grid = None
        self.grid_result = None
        self.launch_kwargs = None

    def __getitem__(self, grid):
        self.grid = grid
        self.grid_result = grid({"BV": 128})

        def launch(**kwargs):
            self.launch_kwargs = kwargs

        return launch


def test_chunk_fwd_o_uses_prebuilt_chunk_offsets(monkeypatch: pytest.MonkeyPatch):
    fake_kernel = _FakeKernel()
    sentinel = torch.tensor([0, 2, 5], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 4, 7], dtype=torch.int32)

    monkeypatch.setattr(chunk_o, "chunk_fwd_kernel_o", fake_kernel)
    monkeypatch.setattr(
        chunk_o,
        "prepare_chunk_offsets",
        lambda *args, **kwargs: pytest.fail("prepare_chunk_offsets should not be called"),
    )

    q = torch.zeros((2, 4, 1, 8), dtype=torch.float32)
    k = torch.zeros((2, 4, 1, 8), dtype=torch.float32)
    v = torch.zeros((2, 4, 1, 16), dtype=torch.float32)
    h = torch.zeros((4, 1, 8, 16), dtype=torch.float32)
    g = torch.zeros((2, 4, 1), dtype=torch.float32)

    chunk_o.chunk_fwd_o(
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_offsets=sentinel,
    )

    assert fake_kernel.launch_kwargs["chunk_offsets"] is sentinel
    assert fake_kernel.grid_result[1] == (len(cu_seqlens) - 1) * v.shape[-2]


def test_chunk_fwd_o_update_uses_prebuilt_chunk_offsets(monkeypatch: pytest.MonkeyPatch):
    fake_kernel = _FakeKernel()
    sentinel = torch.tensor([0, 2, 5], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 4, 7], dtype=torch.int32)

    monkeypatch.setattr(chunk_o_update, "chunk_fwd_kernel_o_update", fake_kernel)
    monkeypatch.setattr(
        chunk_o_update,
        "prepare_chunk_offsets",
        lambda *args, **kwargs: pytest.fail("prepare_chunk_offsets should not be called"),
    )

    q = torch.zeros((2, 4, 1, 8), dtype=torch.float32)
    v = torch.zeros((2, 4, 1, 16), dtype=torch.float32)
    h = torch.zeros((4, 1, 8, 16), dtype=torch.float32)
    h_update = torch.zeros((5, 1, 8, 8), dtype=torch.float32)
    updated_h_state = torch.zeros((1, 8, 16), dtype=torch.float32)

    chunk_o_update.chunk_fwd_o_update(
        q=q,
        v=v,
        h=h,
        h_update=h_update,
        updated_h_state=updated_h_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=sentinel,
    )

    assert fake_kernel.launch_kwargs["chunk_offsets"] is sentinel
    assert fake_kernel.grid_result[1] == (len(cu_seqlens) - 1) * v.shape[-2]
