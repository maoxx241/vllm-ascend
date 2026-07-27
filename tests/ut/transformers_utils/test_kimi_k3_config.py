# SPDX-License-Identifier: Apache-2.0

from transformers import AutoConfig

from vllm_ascend.transformers_utils.configs.kimi_k3 import (
    KimiK3Config,
    register_kimi_k3_config,
)


def test_register_kimi_k3_config_uses_public_transformers_api():
    register_kimi_k3_config()
    assert type(AutoConfig.for_model("kimi_k3")) is KimiK3Config
