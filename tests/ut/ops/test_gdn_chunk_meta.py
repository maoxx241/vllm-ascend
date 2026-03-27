# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import vllm_ascend.patch.worker.patch_gdn_attn as patch_gdn_attn
from vllm_ascend.ops.triton.fla import chunk, chunk_o, chunk_o_update


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


class _DummyTensor:
    def __init__(self, name: str):
        self.name = name
        self.shape = (1,)
        self.dtype = torch.float32

    def unsqueeze(self, dim: int):
        return self

    def new_empty(self, *shape):
        return _DummyTensor(f"{self.name}.new_empty")

    def __getitem__(self, item):
        return self

    def __setitem__(self, item, value):
        return None

    def __add__(self, other):
        return self

    def transpose(self, dim0, dim1):
        return self

    def contiguous(self):
        return self


class _GatherResult:
    def __init__(self, items):
        self.items = items

    def __getitem__(self, item):
        if isinstance(item, tuple):
            item = item[0]
        return self.items[item]


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _patch_missing_cdiv(monkeypatch: pytest.MonkeyPatch, module) -> None:
    if hasattr(module.triton, "cdiv"):
        return
    monkeypatch.setattr(
        module.triton,
        "cdiv",
        lambda x, y: (x + y - 1) // y,
        raising=False,
    )


def _prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    pairs: list[list[int]] = []
    for seq_idx, seq_len in enumerate(lens):
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            pairs.append([seq_idx, chunk_idx])
    if not pairs:
        return torch.empty((0, 2), dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    return torch.tensor(pairs, dtype=cu_seqlens.dtype, device=cu_seqlens.device)


def _prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    num_chunks = torch.div(
        lens + chunk_size - 1,
        chunk_size,
        rounding_mode="floor",
    )
    offsets = torch.zeros(len(num_chunks) + 1, dtype=cu_seqlens.dtype)
    torch.cumsum(num_chunks, dim=0, out=offsets[1:])
    return offsets.to(cu_seqlens.device)


def _prepare_update_chunk_offsets(
    cu_seqlens: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    num_chunks = torch.div(
        lens + chunk_size - 1,
        chunk_size,
        rounding_mode="floor",
    ) + 1
    offsets = torch.zeros(len(num_chunks) + 1, dtype=cu_seqlens.dtype)
    torch.cumsum(num_chunks, dim=0, out=offsets[1:])
    return offsets.to(cu_seqlens.device)


def _prepare_final_chunk_indices(
    cu_seqlens: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    num_chunks = torch.div(
        lens + chunk_size - 1,
        chunk_size,
        rounding_mode="floor",
    ) + 1
    return (torch.cumsum(num_chunks, dim=0) - 1).to(cu_seqlens.device)


def test_chunk_fwd_o_uses_prebuilt_chunk_offsets(monkeypatch: pytest.MonkeyPatch):
    fake_kernel = _FakeKernel()
    sentinel = torch.tensor([0, 2, 5], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 4, 7], dtype=torch.int32)

    _patch_missing_cdiv(monkeypatch, chunk_o)
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

    _patch_missing_cdiv(monkeypatch, chunk_o_update)
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


def test_chunk_gated_delta_rule_fwd_threads_prebuilt_chunk_offsets(
    monkeypatch: pytest.MonkeyPatch,
):
    chunk_offsets = torch.tensor([0, 2, 5], dtype=torch.int32)
    update_chunk_offsets = torch.tensor([0, 3, 7], dtype=torch.int32)
    final_chunk_indices = torch.tensor([1, 3], dtype=torch.int32)
    prebuilt_meta = type(
        "PrebuiltMeta",
        (),
        {
            "block_indices_cumsum": None,
            "chunk_indices_chunk64": None,
            "chunk_offsets_chunk64": chunk_offsets,
            "update_chunk_offsets_chunk64": update_chunk_offsets,
            "final_chunk_indices_chunk64": final_chunk_indices,
            "chunk_indices_large_block": None,
        },
    )()

    q = _DummyTensor("q")
    k = _DummyTensor("k")
    v = _DummyTensor("v")
    g = _DummyTensor("g")
    beta = _DummyTensor("beta")
    initial_state = _DummyTensor("initial_state")

    non_pcp_calls: list[tuple[str, object]] = []
    pcp_calls: list[tuple[str, object]] = []

    def run_case(world_size: int, calls: list[tuple[str, object]]):
        group = type(
            "Group",
            (),
            {
                "world_size": world_size,
                "rank_in_group": 0,
                "all_gather": lambda self, value, dim: _GatherResult([_DummyTensor("g0"), _DummyTensor("g1")]),
            },
        )()

        monkeypatch.setattr(chunk, "get_forward_context", lambda: type("Ctx", (), {"attn_metadata": None})())
        monkeypatch.setattr(chunk, "get_pcp_group", lambda: group)
        monkeypatch.setattr(chunk, "chunk_local_cumsum", lambda *args, **kwargs: _DummyTensor("g_cumsum"))
        monkeypatch.setattr(chunk, "chunk_scaled_dot_kkt_fwd", lambda *args, **kwargs: _DummyTensor("A"))
        monkeypatch.setattr(chunk, "solve_tril", lambda *args, **kwargs: _DummyTensor("A_solved"))
        monkeypatch.setattr(chunk, "recompute_w_u_fwd", lambda *args, **kwargs: (_DummyTensor("w"), _DummyTensor("u")))
        monkeypatch.setattr(
            chunk,
            "chunk_gated_delta_rule_fwd_h",
            lambda *args, **kwargs: (_DummyTensor("h"), _DummyTensor("v_new"), _DummyTensor("final_state")),
        )
        monkeypatch.setattr(
            chunk,
            "chunk_gated_delta_rule_fwd_hupdate",
            lambda *args, **kwargs: _DummyTensor("h_update"),
        )
        monkeypatch.setattr(
            chunk.torch,
            "matmul",
            lambda *args, **kwargs: _DummyTensor("matmul"),
            raising=False,
        )
        monkeypatch.setattr(
            chunk.torch,
            "zeros_like",
            lambda *args, **kwargs: _DummyTensor("zeros_like"),
            raising=False,
        )

        def fake_chunk_fwd_o(*args, **kwargs):
            calls.append(("o", kwargs["chunk_offsets"]))
            return _DummyTensor("o")

        def fake_chunk_fwd_o_update(*args, **kwargs):
            calls.append(("o_update", kwargs["chunk_offsets"]))
            return _DummyTensor("h_updated")

        monkeypatch.setattr(chunk, "chunk_fwd_o", fake_chunk_fwd_o)
        monkeypatch.setattr(chunk, "chunk_fwd_o_update", fake_chunk_fwd_o_update)

        chunk.chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=1.0,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=torch.tensor([0, 4, 7], dtype=torch.int32),
            prebuilt_meta=prebuilt_meta,
        )

    run_case(1, non_pcp_calls)
    assert non_pcp_calls == [("o", chunk_offsets)]

    run_case(2, pcp_calls)
    assert pcp_calls == [("o_update", chunk_offsets), ("o", chunk_offsets)]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU required")
def test_build_chunk_meta_device_matches_cpu_reference_helpers():
    from vllm_ascend.ops.triton.gdn_chunk_meta import build_chunk_meta_device

    device = torch.device("npu")
    num_heads = 32
    cu_seqlens = torch.tensor([0, 4, 12], dtype=torch.int32, device=device)
    cu_seqlens_cpu = cu_seqlens.cpu()

    expected_chunk_indices_64 = _prepare_chunk_indices(cu_seqlens_cpu, 64)
    expected_chunk_offsets_64 = _prepare_chunk_offsets(cu_seqlens_cpu, 64)
    expected_update_chunk_offsets_64 = _prepare_update_chunk_offsets(
        cu_seqlens_cpu,
        64,
    )
    expected_final_chunk_indices_64 = _prepare_final_chunk_indices(
        cu_seqlens_cpu,
        64,
    )
    expected_chunk_indices_large_block = _prepare_chunk_indices(
        cu_seqlens_cpu,
        patch_gdn_attn._GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE,
    )
    cumsum_chunk_size = _next_power_of_2(
        patch_gdn_attn._GDN_CUMSUM_WORKING_SET
        // (num_heads * patch_gdn_attn._GDN_CHUNK_SIZE)
    )
    expected_block_indices_cumsum = _prepare_chunk_indices(
        cu_seqlens_cpu,
        cumsum_chunk_size,
    )

    out_chunk_indices = torch.empty_like(expected_chunk_indices_64, device=device)
    out_chunk_offsets = torch.empty_like(expected_chunk_offsets_64, device=device)
    out_update_chunk_offsets = torch.empty_like(expected_update_chunk_offsets_64, device=device)
    out_final_chunk_indices = torch.empty_like(expected_final_chunk_indices_64, device=device)

    build_chunk_meta_device(
        cu_seqlens=cu_seqlens,
        chunk_size=64,
        out_chunk_indices=out_chunk_indices,
        out_chunk_offsets=out_chunk_offsets,
        out_update_chunk_offsets=out_update_chunk_offsets,
        out_final_chunk_indices=out_final_chunk_indices,
    )
    assert torch.equal(out_chunk_indices.cpu(), expected_chunk_indices_64)
    assert torch.equal(out_chunk_offsets.cpu(), expected_chunk_offsets_64)
    assert torch.equal(out_update_chunk_offsets.cpu(), expected_update_chunk_offsets_64)
    assert torch.equal(out_final_chunk_indices.cpu(), expected_final_chunk_indices_64)

    out_chunk_indices = torch.empty_like(expected_chunk_indices_large_block, device=device)
    out_chunk_offsets = torch.empty_like(expected_chunk_offsets_64, device=device)
    out_update_chunk_offsets = torch.empty_like(expected_update_chunk_offsets_64, device=device)
    out_final_chunk_indices = torch.empty_like(expected_final_chunk_indices_64, device=device)

    build_chunk_meta_device(
        cu_seqlens=cu_seqlens,
        chunk_size=patch_gdn_attn._GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE,
        out_chunk_indices=out_chunk_indices,
        out_chunk_offsets=out_chunk_offsets,
        out_update_chunk_offsets=out_update_chunk_offsets,
        out_final_chunk_indices=out_final_chunk_indices,
    )
    assert torch.equal(out_chunk_indices.cpu(), expected_chunk_indices_large_block)

    out_chunk_indices = torch.empty_like(expected_block_indices_cumsum, device=device)
    out_chunk_offsets = torch.empty_like(expected_chunk_offsets_64, device=device)
    out_update_chunk_offsets = torch.empty_like(expected_update_chunk_offsets_64, device=device)
    out_final_chunk_indices = torch.empty_like(expected_final_chunk_indices_64, device=device)

    build_chunk_meta_device(
        cu_seqlens=cu_seqlens,
        chunk_size=cumsum_chunk_size,
        out_chunk_indices=out_chunk_indices,
        out_chunk_offsets=out_chunk_offsets,
        out_update_chunk_offsets=out_update_chunk_offsets,
        out_final_chunk_indices=out_final_chunk_indices,
    )
    assert torch.equal(out_chunk_indices.cpu(), expected_block_indices_cumsum)
