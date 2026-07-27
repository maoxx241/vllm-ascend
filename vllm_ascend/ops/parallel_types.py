# SPDX-License-Identifier: Apache-2.0
"""Static parallelism contracts shared by model-owned Ascend operators."""

from enum import Enum


class AscendLinearParallelMode(Enum):
    """Select the communication topology owned by an Ascend linear layer."""

    DEFAULT = "default"
    TENSOR_PARALLEL = "tensor_parallel"


class AscendTokenLayout(Enum):
    """Token placement at an explicit model communication boundary."""

    GLOBAL = "global"
    TOKEN_SHARDED = "token_sharded"


__all__ = ["AscendLinearParallelMode", "AscendTokenLayout"]
