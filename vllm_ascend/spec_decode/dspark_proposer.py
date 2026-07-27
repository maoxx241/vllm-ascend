# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.ops.triton.spec_decode.utils import (
    copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid,
)
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.utils import lmhead_tp_enable

_TEMP_DSPARK_BLOCK_SIZE = 7


def validate_temporary_dspark_config(
    *,
    num_speculative_tokens: int,
    draft_sample_method: str,
    lmhead_tp_enabled: bool,
    draft_vocab_size: int | None,
    target_vocab_size: int,
) -> None:
    if num_speculative_tokens != _TEMP_DSPARK_BLOCK_SIZE:
        raise ValueError(
            "The temporary Ascend DSpark path requires exactly "
            f"{_TEMP_DSPARK_BLOCK_SIZE} speculative tokens, got "
            f"{num_speculative_tokens}."
        )
    if lmhead_tp_enabled:
        raise ValueError("The temporary Ascend DSpark path does not support fine-grained LM-head tensor parallelism.")
    if draft_sample_method == "probabilistic":
        raise ValueError("Ascend MRV1 DSpark supports greedy draft sampling only.")
    if draft_vocab_size != target_vocab_size:
        raise ValueError(
            "The temporary Ascend DSpark path supports full-vocabulary "
            "draft checkpoints only: "
            f"draft_vocab_size={draft_vocab_size}, "
            f"target_vocab_size={target_vocab_size}."
        )


class AscendDSparkProposer(AscendDflashProposer):
    """Temporary eager MRV1 DSpark proposer for the fixed seven-token block."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config, device, runner=runner)
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        draft_model_config = speculative_config.draft_model_config
        assert draft_model_config is not None
        draft_hf_config = draft_model_config.hf_config
        draft_vocab_size = getattr(draft_hf_config, "draft_vocab_size", None) or getattr(
            draft_hf_config, "vocab_size", None
        )
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        validate_temporary_dspark_config(
            num_speculative_tokens=self.num_speculative_tokens,
            draft_sample_method=getattr(
                speculative_config,
                "draft_sample_method",
                "greedy",
            ),
            lmhead_tp_enabled=lmhead_tp_enable(),
            draft_vocab_size=draft_vocab_size,
            target_vocab_size=target_vocab_size,
        )

        block_with_seed = 1 + self.num_speculative_tokens
        self._dspark_seed_buffer = torch.zeros(
            self.max_batch_size,
            dtype=torch.int64,
            device=device,
        )
        self._dspark_draft_buffer = torch.zeros(
            (self.max_batch_size, block_with_seed),
            dtype=torch.int64,
            device=device,
        )

        # The target can still use FULL_DECODE_ONLY. The temporary draft path
        # intentionally remains eager.
        self.use_cuda_graph = False

    def _pad_draft_buffers(
        self,
        num_actual_tokens: int,
        num_input_tokens: int,
    ) -> None:
        """Clear a DP-padding tail before attention metadata consumes it."""
        if num_input_tokens <= num_actual_tokens:
            return
        self.positions[num_actual_tokens:num_input_tokens].zero_()
        self.input_ids[num_actual_tokens:num_input_tokens].fill_(self.parallel_drafting_token_id)
        self._slot_mapping_buffer[num_actual_tokens:num_input_tokens].fill_(-1)

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
    ) -> tuple[
        int,
        torch.Tensor,
        CommonAttentionMetadata,
        tuple[Any, Any] | None,
    ]:
        batch_size = cad.num_reqs
        num_query_per_req = self.num_speculative_tokens
        num_query_total = batch_size * num_query_per_req

        num_context = target_token_ids.shape[0]
        self._dflash_num_context = num_context
        self._dflash_hidden_states[:num_context] = target_hidden_states

        num_seeds = next_token_ids.shape[0]
        self._dspark_seed_buffer[:num_seeds].copy_(next_token_ids)
        self._dspark_seed_buffer[num_seeds:].fill_(0)

        token_indices_to_sample = torch.empty(
            num_query_total,
            dtype=torch.int32,
            device=self.device,
        )
        has_num_rejected = num_rejected_tokens_gpu is not None

        copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid[1,](
            next_token_ids_ptr=next_token_ids,
            target_positions_ptr=target_positions,
            context_slot_mapping_ptr=cad.slot_mapping,
            out_input_ids_ptr=self.input_ids,
            out_context_positions_ptr=self._context_positions_buffer,
            out_query_positions_ptr=self.positions,
            out_context_slot_mapping_ptr=self._context_slot_mapping_buffer,
            out_query_slot_mapping_ptr=self._slot_mapping_buffer,
            out_token_indices_ptr=token_indices_to_sample,
            block_table_ptr=cad.block_table_tensor,
            block_table_stride=cad.block_table_tensor.stride(0),
            query_start_loc_ptr=cad.query_start_loc,
            seq_lens_ptr=cad.seq_lens,
            num_rejected_tokens_ptr=(num_rejected_tokens_gpu if has_num_rejected else 0),
            parallel_drafting_token_id=self.parallel_drafting_token_id,
            block_size=self.kernel_block_size,
            num_query_per_req=num_query_per_req,
            num_speculative_tokens=self.num_speculative_tokens,
            total_input_tokens=num_context,
            batch_size=batch_size,
            HAS_NUM_REJECTED=has_num_rejected,
            SAMPLE_FROM_ANCHOR=True,
        )

        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        cad.query_start_loc = self.arange_dflash[: batch_size + 1] * num_query_per_req
        cad.seq_lens = effective_seq_lens + num_query_per_req
        cad.query_start_loc_cpu = (
            torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone() * num_query_per_req
        ).to(torch.int32)

        if hasattr(cad, "actual_seq_lengths_q"):
            cad.actual_seq_lengths_q = [num_query_per_req] * batch_size
        if hasattr(cad, "decode_token_per_req"):
            cad.decode_token_per_req = num_query_per_req

        cad.num_actual_tokens = num_query_total
        cad.num_input_tokens = num_query_total
        cad.max_query_len = num_query_per_req
        cad.max_seq_len += num_query_per_req
        cad.slot_mapping = self._slot_mapping_buffer[:num_query_total]
        # Keep the full buffer for DP padding. Attention backends slice it using
        # the DP-synchronized num_input_tokens.
        cad.positions = self.positions
        cad.causal = False
        cad.attn_mask = None
        cad.attn_state = AscendAttentionState.ChunkedPrefill

        return num_query_total, token_indices_to_sample, cad, None

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        num_reqs: int = 0,
        num_tokens_across_dp: torch.Tensor | None = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
        **kwargs,
    ) -> None:
        num_query_per_req = self.num_speculative_tokens
        num_query_total = num_reqs * num_query_per_req
        num_query_tokens = min(
            num_query_total if num_reqs > 0 else num_tokens,
            self.max_query_tokens,
        )
        (
            num_input_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(
            num_query_tokens,
            is_draft_model=True,
        )

        context_positions = self._context_positions_buffer[:num_input_tokens]
        context_states = self.hidden_states[:num_input_tokens]
        self.token_indices_to_sample.fill_(0)
        self._pad_draft_buffers(num_query_total, num_input_tokens)

        with set_ascend_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_input_tokens,
            in_profile_run=is_profile,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            is_draft_model=True,
            draft_attn_metadatas=[],
        ):
            if is_profile:
                self.model.precompute_and_store_context_kv(
                    context_states,
                    context_positions,
                )
                self.model(
                    input_ids=self.input_ids[:num_query_total],
                    positions=self._get_positions(num_query_total),
                    inputs_embeds=None,
                )
            else:
                self._dflash_num_context = num_input_tokens
                self._runnable(
                    num_input_tokens=num_input_tokens,
                    batch_size=num_reqs,
                    token_indices_to_sample=self.token_indices_to_sample[:num_query_total],
                    target_positions=self._get_positions(num_input_tokens),
                    inputs_embeds=None,
                    multi_steps_attn_metadata=[],
                    num_tokens=num_input_tokens,
                )


# Compatibility with the spelling used by the first Ascend DSpark prototype.
AscendDsparkProposer = AscendDSparkProposer
