from typing import TYPE_CHECKING, Any, Literal, get_args, get_origin

from pydantic.dataclasses import rebuild_dataclass
from vllm.config.speculative import SpeculativeConfig
from vllm.utils.hashing import safe_hash
from vllm.utils.import_utils import LazyLoader

if TYPE_CHECKING:
    import vllm.model_executor.layers.quantization as me_quant
    from transformers import PretrainedConfig
else:
    PretrainedConfig = Any

    me_quant = LazyLoader("model_executor", globals(), "vllm.model_executor.layers.quantization")


def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
    initial_architecture = hf_config.architectures[0]
    if initial_architecture == "DSparkDraftModel" and hf_config.model_type == "qwen3":
        # vLLM's DSpark support normalizes the training-time checkpoint
        # architecture before model-registry inspection. Keep the Ascend
        # override in sync because this module replaces the vLLM hook.
        dflash_config = getattr(hf_config, "dflash_config", None) or {}

        def get_dflash_value(name: str) -> Any:
            if isinstance(dflash_config, dict):
                return dflash_config.get(name)
            return getattr(dflash_config, name, None)

        updates: dict[str, Any] = {"architectures": ["Qwen3DSparkModel"]}
        for name in ("mask_token_id", "target_layer_ids"):
            if (value := get_dflash_value(name)) is not None:
                updates[name] = value
        hf_config.update(updates)

    if hf_config.model_type in ("deepseek_v3", "deepseek_v32", "deepseek_v4", "glm_moe_dsa"):
        target_model_type = hf_config.model_type
        hf_config.model_type = "deepseek_mtp"
    if hf_config.model_type == "deepseek_mtp":
        if target_model_type == "deepseek_v4":
            hf_config.update({"architectures": ["DeepSeekV4MTPModel"]})
        else:
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update({"n_predict": n_predict, "architectures": ["DeepSeekMTPModel"]})
    if hf_config.model_type in ("pangu_ultra_moe"):
        hf_config.model_type = "pangu_ultra_moe_mtp"
    if hf_config.model_type == "pangu_ultra_moe_mtp":
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update({"n_predict": n_predict, "architectures": ["OpenPanguMTPModel"]})

    if hf_config.architectures[0] == "MiMoForCausalLM":
        hf_config.model_type = "mimo_mtp"
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update(
            {
                "num_hidden_layers": 0,
                "n_predict": n_predict,
                "architectures": ["MiMoMTPModel"],
            }
        )

    if hf_config.architectures[0] == "Glm4MoeForCausalLM":
        hf_config.model_type = "glm4_moe_mtp"
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update(
            {
                "n_predict": n_predict,
                "architectures": ["Glm4MoeMTPModel"],
            }
        )

    if hf_config.architectures[0] == "Glm4MoeLiteForCausalLM":
        hf_config.model_type = "glm4_moe_lite_mtp"
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update(
            {
                "num_hidden_layers": 0,
                "n_predict": n_predict,
                "architectures": ["Glm4MoeLiteMTPModel"],
            }
        )

    if hf_config.architectures[0] == "GlmOcrForConditionalGeneration":
        hf_config.model_type = "glm_ocr_mtp"
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update(
            {
                "num_hidden_layers": 0,
                "n_predict": n_predict,
                "architectures": ["GlmOcrMTPModel"],
            }
        )

    if hf_config.model_type == "ernie4_5_moe":
        hf_config.model_type = "ernie_mtp"
    if hf_config.model_type == "ernie_mtp":
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update({"n_predict": n_predict, "architectures": ["ErnieMTPModel"]})

    if (
        hf_config.model_type == "nemotron_h"
        and hasattr(hf_config, "num_nextn_predict_layers")
        and hf_config.num_nextn_predict_layers > 0
    ):
        # Check if this is an MTP variant
        hf_config.model_type = "nemotron_h_mtp"
    if hf_config.model_type == "nemotron_h_mtp":
        n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
        hf_config.update({"n_predict": n_predict, "architectures": ["NemotronHMTPModel"]})

    if hf_config.model_type == "qwen3_next":
        hf_config.model_type = "qwen3_next_mtp"
    if hf_config.model_type == "qwen3_next_mtp":
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update({"n_predict": n_predict, "architectures": ["Qwen3NextMTP"]})

    if hf_config.model_type == "exaone_moe":
        hf_config.model_type = "exaone_moe_mtp"
    if hf_config.model_type == "exaone_moe_mtp":
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update({"n_predict": n_predict, "architectures": ["ExaoneMoeMTP"]})

    if hf_config.model_type in ("qwen3_5", "qwen3_5_moe"):
        is_moe = hf_config.model_type == "qwen3_5_moe"
        hf_config.model_type = "qwen3_5_mtp"
        n_predict = getattr(hf_config, "mtp_num_hidden_layers", None)
        hf_config.update(
            {
                "n_predict": n_predict,
                "architectures": ["Qwen3_5MoeMTP" if is_moe else "Qwen3_5MTP"],
            }
        )
    if hf_config.model_type == "longcat_flash":
        hf_config.model_type = "longcat_flash_mtp"
        n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
        hf_config.update({"n_predict": n_predict, "architectures": ["LongCatFlashMTPModel"]})

    if hf_config.model_type in ("step3p5", "step3p7") or hf_config.architectures[0] in (
        "Step3p5ForCausalLM",
        "Step3p7ForConditionalGeneration",
    ):
        quantization_config = getattr(hf_config, "quantization_config", None)
        hf_config = getattr(hf_config, "text_config", hf_config)
        if quantization_config is not None and getattr(hf_config, "quantization_config", None) is None:
            hf_config.update({"quantization_config": quantization_config})
        hf_config.model_type = "step3p5_mtp"
        n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
        hf_config.update({"n_predict": n_predict, "architectures": ["Step3p5MTP"]})

    if initial_architecture == "MistralLarge3ForCausalLM":
        hf_config.update({"architectures": ["EagleMistralLarge3ForCausalLM"]})

    return hf_config


_ORIGINAL_POST_INIT = SpeculativeConfig.__post_init__
_ORIGINAL_COMPUTE_HASH = SpeculativeConfig.compute_hash
_ORIGINAL_USE_EAGLE = SpeculativeConfig.use_eagle


def _verify_dspark_args(self) -> None:
    if self.method != "dspark":
        return

    hf_config = self.draft_model_config.hf_config
    block_size = getattr(hf_config, "block_size", None)
    if block_size != 7:
        raise ValueError(f"This temporary DSpark implementation requires checkpoint block_size=7, got {block_size!r}.")
    if self.num_speculative_tokens != block_size:
        raise ValueError(
            "DSpark requires num_speculative_tokens to equal the checkpoint "
            f"block_size ({block_size}), got {self.num_speculative_tokens}."
        )
    if self.draft_sample_method != "greedy":
        raise ValueError("This temporary DSpark implementation supports only draft_sample_method='greedy'.")


def _dspark_post_init(self):
    if self.method != "dspark":
        return _ORIGINAL_POST_INIT(self)

    # v0.23.0 does not know the DSpark method. Let its existing draft-model
    # path build the ModelConfig without wrapping the checkpoint as EAGLE,
    # then restore the DSpark identity for the scheduler and proposer.
    self.method = "draft_model"
    try:
        result = _ORIGINAL_POST_INIT(self)
    except Exception:
        self.method = "dspark"
        raise

    self.method = "dspark"
    self.parallel_drafting = True
    _verify_dspark_args(self)
    return result


def _compute_hash(self) -> str:
    if self.method != "dspark":
        return _ORIGINAL_COMPUTE_HASH(self)

    factors: list[Any] = [True]
    if self.draft_model_config is not None:
        layer_ids = getattr(
            self.draft_model_config.hf_config,
            "target_layer_ids",
            None,
        )
        if layer_ids is not None:
            factors.append(tuple(layer_ids))
    return safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()


def _use_eagle(self) -> bool:
    return self.method == "dspark" or _ORIGINAL_USE_EAGLE(self)


def _use_dspark(self) -> bool:
    return self.method == "dspark"


def _add_dspark_method_to_schema() -> None:
    annotation = SpeculativeConfig.__pydantic_fields__["method"].annotation
    method_values: list[Any] = []

    def collect_literal_values(value: Any) -> None:
        if get_origin(value) is Literal:
            method_values.extend(get_args(value))
            return
        for arg in get_args(value):
            collect_literal_values(arg)

    collect_literal_values(annotation)
    if "dspark" not in method_values:
        method_values.append("dspark")
    method_literal = Literal.__getitem__(tuple(method_values))
    method_annotation = method_literal | None
    SpeculativeConfig.__annotations__["method"] = method_annotation
    SpeculativeConfig.__dataclass_fields__["method"].type = method_annotation


SpeculativeConfig.hf_config_override = hf_config_override
SpeculativeConfig.__post_init__ = _dspark_post_init
SpeculativeConfig._verify_dspark_args = _verify_dspark_args
SpeculativeConfig.compute_hash = _compute_hash
SpeculativeConfig.use_eagle = _use_eagle
SpeculativeConfig.use_dspark = _use_dspark
_add_dspark_method_to_schema()
rebuild_dataclass(SpeculativeConfig, force=True)
