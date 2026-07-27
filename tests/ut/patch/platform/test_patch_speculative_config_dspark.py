from types import SimpleNamespace

import pytest
from pydantic import TypeAdapter
from transformers import Qwen3Config
from vllm.config import SpeculativeConfig

from vllm_ascend.patch.platform import patch_speculative_config
from vllm_ascend.patch.platform.patch_speculative_config import (
    _dspark_post_init,
    hf_config_override,
)


def test_ascend_override_preserves_kimi_dspark_normalization():
    config = Qwen3Config(
        architectures=["DSparkDraftModel"],
        block_size=7,
        dflash_config={
            "mask_token_id": 163824,
            "target_layer_ids": [7, 23, 51, 67, 83],
        },
        pad_token_id=163839,
    )

    normalized = hf_config_override(config)

    assert SpeculativeConfig.hf_config_override is hf_config_override
    assert normalized is config
    assert normalized.architectures == ["Qwen3DSparkModel"]
    assert normalized.mask_token_id == 163824
    assert normalized.target_layer_ids == [7, 23, 51, 67, 83]


def test_dspark_is_accepted_by_v023_speculative_config_schema():
    method_annotation = SpeculativeConfig.__pydantic_fields__["method"].annotation

    assert TypeAdapter(method_annotation).validate_python("dspark") == "dspark"


def test_dspark_uses_v023_draft_model_path_then_restores_method(monkeypatch):
    observed_methods = []

    def original_post_init(config):
        observed_methods.append(config.method)
        return config

    monkeypatch.setattr(
        patch_speculative_config,
        "_ORIGINAL_POST_INIT",
        original_post_init,
    )
    config = SimpleNamespace(
        method="dspark",
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(block_size=7),
        ),
        num_speculative_tokens=7,
        draft_sample_method="greedy",
        parallel_drafting=False,
    )

    assert _dspark_post_init(config) is config
    assert observed_methods == ["draft_model"]
    assert config.method == "dspark"
    assert config.parallel_drafting is True


@pytest.mark.parametrize(
    ("block_size", "num_speculative_tokens", "draft_sample_method", "match"),
    [
        (8, 7, "greedy", "block_size=7"),
        (7, 6, "greedy", "num_speculative_tokens"),
        (7, 7, "probabilistic", "greedy"),
    ],
)
def test_dspark_v023_patch_rejects_non_k7_or_probabilistic_config(
    block_size,
    num_speculative_tokens,
    draft_sample_method,
    match,
    monkeypatch,
):
    monkeypatch.setattr(
        patch_speculative_config,
        "_ORIGINAL_POST_INIT",
        lambda config: config,
    )
    config = SimpleNamespace(
        method="dspark",
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(block_size=block_size),
        ),
        num_speculative_tokens=num_speculative_tokens,
        draft_sample_method=draft_sample_method,
        parallel_drafting=False,
    )

    with pytest.raises(ValueError, match=match):
        _dspark_post_init(config)
