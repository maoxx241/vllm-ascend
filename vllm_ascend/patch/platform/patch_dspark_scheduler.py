# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Temporary v0.23.0 scheduler compatibility for DSpark and P/D serving."""

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler

_ORIGINAL_SCHEDULER_INIT = Scheduler.__init__
_ORIGINAL_GET_COMPUTED_BLOCKS = KVCacheManager.get_computed_blocks


def _scheduler_init(self, *args, **kwargs) -> None:
    _ORIGINAL_SCHEDULER_INIT(self, *args, **kwargs)

    speculative_config = self.vllm_config.speculative_config
    use_dspark = speculative_config is not None and getattr(speculative_config, "method", None) == "dspark"
    self.use_dspark = use_dspark
    if use_dspark:
        # DSpark predicts exactly K tokens from the anchor position.
        self.num_lookahead_tokens = speculative_config.num_speculative_tokens
        self.kv_cache_manager._ascend_dspark_skip_local_prefix_cache = True


def _get_computed_blocks(self, request):
    if (
        getattr(self, "_ascend_dspark_skip_local_prefix_cache", False)
        and request.kv_transfer_params is not None
        and request.kv_transfer_params.get("do_remote_decode") is True
    ):
        # The P side must recompute the connector-truncated prompt so every
        # DSpark draft KV slot is populated before Mooncake transfers it.
        return self.empty_kv_cache_blocks, 0
    return _ORIGINAL_GET_COMPUTED_BLOCKS(self, request)


Scheduler.__init__ = _scheduler_init
KVCacheManager.get_computed_blocks = _get_computed_blocks
