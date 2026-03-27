# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.patch.worker.patch_gdn_attn as patch_gdn_attn
from vllm_ascend.patch.worker.patch_qwen3_5 import (
    get_non_spec_causal_conv1d_host_args,
)
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import MambaSpec


@dataclass
class BatchSpec:
    seq_lens: list[int]
    query_lens: list[int]
    name: str = "unnamed"

    @property
    def batch_size(self):
        return len(self.seq_lens)


def create_common_attn_metadata(
    batch_spec: BatchSpec,
    block_size: int,
    device: torch.device,
) -> CommonAttentionMetadata:
    query_start_loc = torch.zeros(
        batch_spec.batch_size + 1,
        dtype=torch.int32,
        device=device,
    )
    query_start_loc[1:] = torch.tensor(
        batch_spec.query_lens,
        dtype=torch.int32,
        device=device,
    ).cumsum(0)
    query_start_loc_cpu = query_start_loc.cpu()
    num_tokens = sum(batch_spec.query_lens)

    seq_lens = torch.tensor(batch_spec.seq_lens, dtype=torch.int32, device=device)
    seq_lens_cpu = seq_lens.cpu()
    max_seq_len = int(seq_lens_cpu.max())
    context_lens = [
        batch_spec.seq_lens[i] - batch_spec.query_lens[i]
        for i in range(batch_spec.batch_size)
    ]
    num_computed_tokens_cpu = torch.tensor(context_lens, dtype=torch.int32)
    max_blocks = (max(batch_spec.seq_lens) + block_size - 1) // block_size
    block_table_tensor = torch.arange(
        batch_spec.batch_size * max_blocks,
        dtype=torch.int32,
        device=device,
    ).view(batch_spec.batch_size, max_blocks)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=batch_spec.batch_size,
        num_actual_tokens=num_tokens,
        max_query_len=max(batch_spec.query_lens),
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=True,
    )
    return common_attn_metadata


def _make_vllm_config(
    *,
    max_model_len: int = 8192,
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    num_heads: int = 32,
    num_speculative_tokens: int = 0,
):
    speculative_config = None
    if num_speculative_tokens > 0:
        speculative_config = SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens,
            parallel_drafting=False,
        )

    model_config = SimpleNamespace(max_model_len=max_model_len)
    model_config.get_num_attention_heads = lambda parallel_config: num_heads

    return SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.NONE,
            max_cudagraph_capture_size=None,
        ),
        speculative_config=speculative_config,
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            tensor_parallel_size=1,
        ),
        model_config=model_config,
    )


def _make_builder(*, device: torch.device, num_heads: int, num_speculative_tokens: int):
    vllm_config = _make_vllm_config(
        num_heads=num_heads,
        num_speculative_tokens=num_speculative_tokens,
    )
    spec = MambaSpec(
        block_size=16,
        shapes=((1,), (1,)),
        dtypes=(torch.float32,),
        mamba_cache_mode="none",
    )
    return GDNAttentionMetadataBuilder(spec, ["layer0"], vllm_config, device)


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


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


def _build_non_spec_query_start_loc_cpu(
    batch_spec: BatchSpec, spec_mask_cpu: torch.Tensor | None
) -> torch.Tensor:
    query_lens = torch.tensor(batch_spec.query_lens, dtype=torch.int32)
    if spec_mask_cpu is not None:
        query_lens = query_lens[~spec_mask_cpu]
    query_start_loc = torch.zeros(query_lens.numel() + 1, dtype=torch.int32)
    torch.cumsum(query_lens, dim=0, out=query_start_loc[1:])
    return query_start_loc


def _build_expected_non_spec_cache_indices_cpu(
    batch_spec: BatchSpec,
    block_size: int,
    spec_mask_cpu: torch.Tensor | None,
) -> torch.Tensor:
    max_blocks = (max(batch_spec.seq_lens) + block_size - 1) // block_size
    block_table_tensor_cpu = torch.arange(
        batch_spec.batch_size * max_blocks,
        dtype=torch.int32,
    ).view(batch_spec.batch_size, max_blocks)
    cache_indices_cpu = block_table_tensor_cpu[:, 0]
    if spec_mask_cpu is not None:
        cache_indices_cpu = cache_indices_cpu[~spec_mask_cpu]
    return cache_indices_cpu


def _build_expected_non_spec_has_initial_state_cpu(
    batch_spec: BatchSpec,
    spec_mask_cpu: torch.Tensor | None,
) -> torch.Tensor:
    query_lens_cpu = torch.tensor(batch_spec.query_lens, dtype=torch.int32)
    seq_lens_cpu = torch.tensor(batch_spec.seq_lens, dtype=torch.int32)
    has_initial_state_cpu = seq_lens_cpu - query_lens_cpu > 0
    if spec_mask_cpu is not None:
        has_initial_state_cpu = has_initial_state_cpu[~spec_mask_cpu]
    return has_initial_state_cpu


@pytest.mark.parametrize(
    ("batch_spec", "num_speculative_tokens", "num_decode_draft_tokens_cpu"),
    [
        (
            BatchSpec(
                seq_lens=[8, 12],
                query_lens=[4, 8],
                name="pure_non_spec_prefill",
            ),
            0,
            None,
        ),
        (
            BatchSpec(
                seq_lens=[8, 4, 0, 12],
                query_lens=[4, 4, 0, 8],
                name="mixed_spec_non_spec_with_padding",
            ),
            3,
            torch.tensor([-1, 3, -1, -1], dtype=torch.int32),
        ),
        (
            BatchSpec(
                seq_lens=[5, 12, 0, 9],
                query_lens=[1, 8, 0, 1],
                name="mixed_prefill_decode_without_spec",
            ),
            0,
            None,
        ),
    ],
    ids=lambda case: case.name if isinstance(case, BatchSpec) else None,
)
def test_builder_prebuilds_non_spec_chunk_metadata_exactly(
    batch_spec: BatchSpec,
    num_speculative_tokens: int,
    num_decode_draft_tokens_cpu: torch.Tensor | None,
):
    device = torch.device("cpu")
    num_heads = 32
    common_attn_metadata = create_common_attn_metadata(
        batch_spec=batch_spec,
        block_size=16,
        device=device,
    )
    builder = _make_builder(
        device=device,
        num_heads=num_heads,
        num_speculative_tokens=num_speculative_tokens,
    )

    num_accepted_tokens = None
    spec_mask_cpu = None
    if num_decode_draft_tokens_cpu is not None:
        num_accepted_tokens = torch.ones(
            batch_spec.batch_size, dtype=torch.int32, device=device
        )
        spec_mask_cpu = num_decode_draft_tokens_cpu >= 0

    attn_metadata = builder.build(
        0,
        common_attn_metadata,
        num_accepted_tokens=num_accepted_tokens,
        num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
    )

    non_spec_query_start_loc_cpu = _build_non_spec_query_start_loc_cpu(
        batch_spec,
        spec_mask_cpu,
    )
    legacy_chunk_indices_64 = _prepare_chunk_indices(non_spec_query_start_loc_cpu, 64)
    legacy_chunk_offsets_64 = _prepare_chunk_offsets(non_spec_query_start_loc_cpu, 64)
    legacy_update_chunk_offsets_64 = _prepare_update_chunk_offsets(
        non_spec_query_start_loc_cpu,
        64,
    )
    legacy_final_chunk_indices_64 = _prepare_final_chunk_indices(
        non_spec_query_start_loc_cpu,
        64,
    )
    legacy_chunk_indices_large_block = _prepare_chunk_indices(
        non_spec_query_start_loc_cpu,
        patch_gdn_attn._GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE,
    )
    optim_block_size = _next_power_of_2(
        patch_gdn_attn._GDN_CUMSUM_WORKING_SET
        // (num_heads * patch_gdn_attn._GDN_CHUNK_SIZE)
    )
    legacy_block_indices_cumsum = _prepare_chunk_indices(
        non_spec_query_start_loc_cpu,
        optim_block_size,
    )

    fallback_meta = getattr(attn_metadata, "non_spec_prefill_fallback_meta", None)
    assert fallback_meta is not None
    prebuilt_meta = fallback_meta.chunk
    assert prebuilt_meta is not None
    assert torch.equal(prebuilt_meta.chunk_indices_chunk64, legacy_chunk_indices_64)
    assert torch.equal(prebuilt_meta.chunk_offsets_chunk64, legacy_chunk_offsets_64)
    assert torch.equal(
        prebuilt_meta.update_chunk_offsets_chunk64, legacy_update_chunk_offsets_64
    )
    assert torch.equal(
        prebuilt_meta.final_chunk_indices_chunk64, legacy_final_chunk_indices_64
    )
    assert torch.equal(
        prebuilt_meta.chunk_indices_large_block,
        legacy_chunk_indices_large_block,
    )
    assert torch.equal(
        prebuilt_meta.block_indices_cumsum,
        legacy_block_indices_cumsum,
    )


def test_builder_delegates_device_fill_for_non_spec_chunk_metadata(monkeypatch):
    device = torch.device("cpu")
    batch_spec = BatchSpec(
        seq_lens=[8, 12],
        query_lens=[4, 8],
        name="pure_non_spec_prefill",
    )
    cu_seqlens_cpu = _build_non_spec_query_start_loc_cpu(batch_spec, None)
    builder = _make_builder(
        device=device,
        num_heads=32,
        num_speculative_tokens=0,
    )
    builder._ascend_gdn_chunk_meta_device = SimpleNamespace(type="npu")
    builder._ascend_gdn_chunk_size = patch_gdn_attn._GDN_CHUNK_SIZE
    builder._ascend_gdn_large_block_size = patch_gdn_attn._GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE
    builder._ascend_gdn_cumsum_block_size = _next_power_of_2(
        patch_gdn_attn._GDN_CUMSUM_WORKING_SET
        // (32 * patch_gdn_attn._GDN_CHUNK_SIZE)
    )

    class DummyChunkBuffer:
        def __init__(self, shape):
            self.cpu = torch.zeros(shape, dtype=torch.int32)
            self.copy_calls = 0

        def copy_to_gpu(self, num_items: int):
            self.copy_calls += 1
            return self.cpu[:num_items].clone()

    slot = SimpleNamespace(
        chunk_indices_chunk64=DummyChunkBuffer((8, 2)),
        chunk_offsets_chunk64=DummyChunkBuffer((8,)),
        update_chunk_offsets_chunk64=DummyChunkBuffer((8,)),
        final_chunk_indices_chunk64=DummyChunkBuffer((8,)),
        chunk_indices_large_block=DummyChunkBuffer((8, 2)),
        block_indices_cumsum=DummyChunkBuffer((8, 2)),
    )
    builder._ascend_gdn_chunked_prefill_pool_idx = -1
    builder._ascend_gdn_chunked_prefill_pool = [slot]

    called_chunk_sizes: list[int] = []

    def fake_build_chunk_meta_device(builder_arg, cu_seqlens_arg, chunk_size):
        assert builder_arg is builder
        assert torch.equal(cu_seqlens_arg, cu_seqlens_cpu)
        called_chunk_sizes.append(chunk_size)

    monkeypatch.setattr(
        patch_gdn_attn,
        "build_chunk_meta_device",
        fake_build_chunk_meta_device,
        raising=False,
    )

    patch_gdn_attn._build_non_spec_chunked_prefill_meta(builder, cu_seqlens_cpu)

    assert called_chunk_sizes == [
        64,
        patch_gdn_attn._GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE,
        builder._ascend_gdn_cumsum_block_size,
    ]
    assert slot.chunk_indices_chunk64.copy_calls == 0
    assert slot.chunk_offsets_chunk64.copy_calls == 0
    assert slot.update_chunk_offsets_chunk64.copy_calls == 0
    assert slot.final_chunk_indices_chunk64.copy_calls == 0
    assert slot.chunk_indices_large_block.copy_calls == 0
    assert slot.block_indices_cumsum.copy_calls == 0


@pytest.mark.parametrize(
    ("batch_spec", "num_speculative_tokens", "num_decode_draft_tokens_cpu"),
    [
        (
            BatchSpec(
                seq_lens=[8, 12],
                query_lens=[4, 8],
                name="pure_non_spec_prefill",
            ),
            0,
            None,
        ),
        (
            BatchSpec(
                seq_lens=[8, 4, 0, 12],
                query_lens=[4, 4, 0, 8],
                name="mixed_spec_non_spec_with_padding",
            ),
            3,
            torch.tensor([-1, 3, -1, -1], dtype=torch.int32),
        ),
    ],
    ids=lambda case: case.name if isinstance(case, BatchSpec) else None,
)
def test_builder_prebuilds_non_spec_causal_conv1d_host_metadata(
    batch_spec: BatchSpec,
    num_speculative_tokens: int,
    num_decode_draft_tokens_cpu: torch.Tensor | None,
):
    device = torch.device("cpu")
    common_attn_metadata = create_common_attn_metadata(
        batch_spec=batch_spec,
        block_size=16,
        device=device,
    )
    builder = _make_builder(
        device=device,
        num_heads=32,
        num_speculative_tokens=num_speculative_tokens,
    )

    num_accepted_tokens = None
    spec_mask_cpu = None
    if num_decode_draft_tokens_cpu is not None:
        num_accepted_tokens = torch.ones(
            batch_spec.batch_size, dtype=torch.int32, device=device
        )
        spec_mask_cpu = num_decode_draft_tokens_cpu >= 0

    attn_metadata = builder.build(
        0,
        common_attn_metadata,
        num_accepted_tokens=num_accepted_tokens,
        num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
    )

    non_spec_query_start_loc_cpu = _build_non_spec_query_start_loc_cpu(
        batch_spec,
        spec_mask_cpu,
    )
    expected_cache_indices_cpu = _build_expected_non_spec_cache_indices_cpu(
        batch_spec,
        block_size=16,
        spec_mask_cpu=spec_mask_cpu,
    )
    expected_has_initial_state_cpu = _build_expected_non_spec_has_initial_state_cpu(
        batch_spec,
        spec_mask_cpu,
    )

    fallback_meta = getattr(attn_metadata, "non_spec_prefill_fallback_meta", None)
    assert fallback_meta is not None
    prebuilt_meta = fallback_meta.causal_conv1d
    assert prebuilt_meta is not None
    assert torch.equal(
        prebuilt_meta.query_start_loc_cpu,
        non_spec_query_start_loc_cpu,
    )
    assert torch.equal(
        prebuilt_meta.cache_indices_cpu,
        expected_cache_indices_cpu,
    )
    assert torch.equal(
        prebuilt_meta.has_initial_state_cpu,
        expected_has_initial_state_cpu,
    )


def test_get_non_spec_causal_conv1d_host_args_prefers_prefill_fallback_meta():
    attn_metadata = SimpleNamespace(
        non_spec_prefill_fallback_meta=patch_gdn_attn.GDNPrefillFallbackMeta(
            causal_conv1d=patch_gdn_attn.GDNCausalConv1dHostMetadata(
                query_start_loc_cpu=torch.tensor([0, 4, 12], dtype=torch.int32),
                cache_indices_cpu=torch.tensor([3, 9], dtype=torch.int32),
                has_initial_state_cpu=torch.tensor([True, False]),
            ),
            chunk=None,
        ),
    )

    host_args = get_non_spec_causal_conv1d_host_args(
        attn_metadata,
        non_spec_query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        non_spec_state_indices_tensor=torch.tensor([100], dtype=torch.int32),
        has_initial_state=torch.tensor([False]),
    )

    assert host_args == ((0, 4, 12), (3, 9), (1, 0))


def test_allocate_chunked_prefill_slot_uses_cpugpubuffer(monkeypatch):
    class DummyCpuGpuBuffer:
        def __init__(
            self,
            *size,
            dtype: torch.dtype,
            device: torch.device,
            pin_memory: bool,
            with_numpy: bool = True,
        ) -> None:
            self.cpu = torch.zeros(*size, dtype=dtype, device="cpu")
            self.gpu = torch.zeros_like(self.cpu, device=device)
            self.dtype = dtype
            self.device = device
            self.pin_memory = pin_memory
            self.with_numpy = with_numpy

    device = torch.device("cpu")
    builder = _make_builder(
        device=device,
        num_heads=32,
        num_speculative_tokens=0,
    )
    monkeypatch.setattr(patch_gdn_attn, "CpuGpuBuffer", DummyCpuGpuBuffer)

    slot = patch_gdn_attn._allocate_chunked_prefill_slot(builder, device)

    assert isinstance(slot.chunk_indices_chunk64, DummyCpuGpuBuffer)
    assert isinstance(slot.chunk_offsets_chunk64, DummyCpuGpuBuffer)
    assert isinstance(slot.update_chunk_offsets_chunk64, DummyCpuGpuBuffer)
    assert isinstance(slot.final_chunk_indices_chunk64, DummyCpuGpuBuffer)
    assert slot.chunk_indices_chunk64.pin_memory is True
    assert slot.chunk_indices_chunk64.with_numpy is False
    assert slot.chunk_indices_chunk64.device == device
    assert slot.chunk_indices_chunk64.cpu.shape == (
        builder.vllm_config.scheduler_config.max_num_batched_tokens,
        2,
    )
    assert slot.chunk_indices_chunk64.gpu.shape == (
        builder.vllm_config.scheduler_config.max_num_batched_tokens,
        2,
    )


@pytest.mark.parametrize(
    "batch_spec",
    [
        BatchSpec(seq_lens=[1, 1, 1], query_lens=[1, 1, 1], name="decode_only"),
        BatchSpec(seq_lens=[4, 4], query_lens=[4, 4], name="spec_only"),
    ],
)
def test_builder_skips_prebuilt_meta_without_non_spec_prefill(batch_spec: BatchSpec):
    device = torch.device("cpu")
    builder = _make_builder(
        device=device,
        num_heads=32,
        num_speculative_tokens=3 if batch_spec.name == "spec_only" else 0,
    )
    common_attn_metadata = create_common_attn_metadata(
        batch_spec=batch_spec,
        block_size=16,
        device=device,
    )

    num_accepted_tokens = None
    num_decode_draft_tokens_cpu = None
    if batch_spec.name == "spec_only":
        num_accepted_tokens = torch.ones(
            batch_spec.batch_size, dtype=torch.int32, device=device
        )
        num_decode_draft_tokens_cpu = torch.full(
            (batch_spec.batch_size,), 3, dtype=torch.int32
        )

    attn_metadata = builder.build(
        0,
        common_attn_metadata,
        num_accepted_tokens=num_accepted_tokens,
        num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
    )

    assert getattr(attn_metadata, "non_spec_prefill_fallback_meta", None) is None
