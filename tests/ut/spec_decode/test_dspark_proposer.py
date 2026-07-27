# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.gdn_attn_builder import (
    AscendGDNAttentionMetadataBuilder,
)
from vllm_ascend.spec_decode.dspark_proposer import (
    AscendDSparkProposer,
    validate_temporary_dspark_config,
)
from vllm_ascend.spec_decode.llm_base_proposer import (
    AscendSpecDecodeBaseProposer,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def test_dspark_dp_padding_is_cleared_before_metadata_build():
    proposer = AscendDSparkProposer.__new__(AscendDSparkProposer)
    proposer.parallel_drafting_token_id = 99
    proposer.positions = torch.full((32,), -7, dtype=torch.int32)
    proposer.input_ids = torch.full((32,), -7, dtype=torch.int64)
    proposer._slot_mapping_buffer = torch.full(
        (32,),
        -7,
        dtype=torch.int32,
    )

    proposer._pad_draft_buffers(14, 21)

    assert torch.all(proposer.positions[14:21] == 0)
    assert torch.all(proposer.input_ids[14:21] == 99)
    assert torch.all(proposer._slot_mapping_buffer[14:21] == -1)

    source = inspect.getsource(AscendSpecDecodeBaseProposer._propose)
    sync_index = source.index("self.runner._sync_metadata_across_dp")
    pad_index = source.index("self._pad_draft_buffers")
    metadata_index = source.index("builder.build")
    assert sync_index < pad_index < metadata_index


def test_dspark_dummy_run_clears_dp_padding_before_forward():
    source = inspect.getsource(AscendDSparkProposer.dummy_run)
    sync_index = source.index("self.runner._sync_metadata_across_dp")
    pad_index = source.index("self._pad_draft_buffers")
    forward_context_index = source.index("with set_ascend_forward_context")
    assert sync_index < pad_index < forward_context_index


def test_dspark_aux_layer_ids_use_following_layer_boundaries():
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.speculative_config = SimpleNamespace(
        method="dspark",
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                target_layer_ids=[7, 23, 51, 67, 83],
            )
        ),
    )

    assert runner._get_eagle3_aux_layers_from_config() == (
        8,
        24,
        52,
        68,
        84,
    )


def test_reorder_threshold_accepts_matching_anchor_plus_k_boundary():
    runner = NPUModelRunner.__new__(NPUModelRunner)

    def group(threshold):
        builder = SimpleNamespace(reorder_batch_threshold=threshold)
        return SimpleNamespace(get_metadata_builder=lambda: builder)

    runner._attn_group_iterator = lambda: iter([group(8), group(None), group(8)])
    runner.reorder_batch_threshold = None

    runner.calculate_reorder_batch_threshold()

    assert runner.reorder_batch_threshold == 8


def test_gdn_dspark_threshold_counts_anchor_plus_k():
    source = inspect.getsource(AscendGDNAttentionMetadataBuilder._init_reorder_batch_threshold)
    assert 'method in ("dflash", "dspark")' in source
    assert "1 + num_speculative_tokens" in source


def test_reorder_threshold_rejects_mismatched_hybrid_boundaries():
    runner = NPUModelRunner.__new__(NPUModelRunner)

    def group(threshold):
        builder = SimpleNamespace(reorder_batch_threshold=threshold)
        return SimpleNamespace(get_metadata_builder=lambda: builder)

    runner._attn_group_iterator = lambda: iter([group(8), group(7)])
    runner.reorder_batch_threshold = None

    with pytest.raises(ValueError, match="threshold 7"):
        runner.calculate_reorder_batch_threshold()


def test_temporary_dspark_full_vocab_config_is_supported():
    validate_temporary_dspark_config(
        num_speculative_tokens=7,
        draft_sample_method="greedy",
        lmhead_tp_enabled=False,
        draft_vocab_size=163840,
        target_vocab_size=163840,
    )


def test_temporary_dspark_rejects_fine_grained_lmhead_tp():
    with pytest.raises(ValueError, match="LM-head tensor parallelism"):
        validate_temporary_dspark_config(
            num_speculative_tokens=7,
            draft_sample_method="greedy",
            lmhead_tp_enabled=True,
            draft_vocab_size=163840,
            target_vocab_size=163840,
        )


def test_temporary_dspark_rejects_reduced_vocab():
    with pytest.raises(ValueError, match="full-vocabulary"):
        validate_temporary_dspark_config(
            num_speculative_tokens=7,
            draft_sample_method="greedy",
            lmhead_tp_enabled=False,
            draft_vocab_size=32000,
            target_vocab_size=163840,
        )


def test_dspark_sampling_stays_in_draft_domain_then_maps_to_target():
    source = inspect.getsource(AscendSpecDecodeBaseProposer._run_merged_draft)
    compute_index = source.index("compute_draft_logits")
    map_index = source.index("map_draft_to_target")
    assert compute_index < map_index


def test_dspark_receives_all_kernel_block_sizes():
    source = inspect.getsource(NPUModelRunner.initialize_kv_cache)
    normalize_index = source.index("kernel_block_sizes = [")
    initialize_index = source.index(
        "self.drafter.initialize_attn_backend(\n"
        "                    kv_cache_config,\n"
        "                    kernel_block_sizes,"
    )
    assert normalize_index < initialize_index


def test_kimi_k3_dspark_uses_media_placeholder_for_multimodal_token():
    source = inspect.getsource(AscendSpecDecodeBaseProposer.load_model)
    kimi_k3_index = source.index('"AscendKimiK3ForConditionalGeneration"')
    media_token_index = source.index(
        "model.config.media_placeholder_token_id",
        kimi_k3_index,
    )
    generic_image_token_index = source.index(
        "model.config.image_token_index",
        media_token_index,
    )
    assert kimi_k3_index < media_token_index < generic_image_token_index
