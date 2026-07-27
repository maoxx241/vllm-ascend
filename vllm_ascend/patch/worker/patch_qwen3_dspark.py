# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

_ORIGINAL_INIT_PARALLEL_DRAFTING_PARAMS = SpecDecodeBaseProposer._init_parallel_drafting_params


def _init_parallel_drafting_params(self):
    speculative_config = self.vllm_config.speculative_config
    draft_hf_config = self.draft_model_config.hf_config
    if speculative_config.method == "dspark" and hasattr(draft_hf_config, "mask_token_id"):
        self.parallel_drafting_token_id = draft_hf_config.mask_token_id
        return
    _ORIGINAL_INIT_PARALLEL_DRAFTING_PARAMS(self)


SpecDecodeBaseProposer._init_parallel_drafting_params = _init_parallel_drafting_params
