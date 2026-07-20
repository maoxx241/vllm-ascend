# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from PIL import Image
from torch import nn
from vllm.multimodal.parse import MultiModalDataItems, VisionChunkProcessorItems

from vllm_ascend.models import kimi_k3, kimi_k3_text, kimi_k3_vit
from vllm_ascend.models.kimi_k3 import (
    AscendKimiK3ForConditionalGeneration,
    KimiK3MultiModalProcessor,
    _move_module_to_device,
)
from vllm_ascend.models.kimi_k3_text import (
    AscendKimiK3ForCausalLM,
    KimiK3DecoderLayer,
    KimiK3MLAAttention,
    KimiK3MoE,
    KimiK3TextModel,
    _apply_attention_residual,
    _routed_latent_quant_config,
)
from vllm_ascend.models.kimi_k3_vit import KimiK3VisionPatchEmbed
from vllm_ascend.transformers_utils.configs.kimi_k3 import KimiK3Config
from vllm_ascend.transformers_utils.processors.kimi_k3 import KimiK3Processor


def _tiny_k3_config() -> KimiK3Config:
    return KimiK3Config(
        text_config={
            "architectures": ["KimiK3ForCausalLM"],
            "hidden_size": 32,
            "num_attention_heads": 4,
            "num_hidden_layers": 4,
            "intermediate_size": 64,
            "hidden_act": "situ",
            "q_lora_rank": 16,
            "kv_lora_rank": 8,
            "qk_nope_head_dim": 8,
            "qk_rope_head_dim": 4,
            "v_head_dim": 8,
            "mla_use_nope": True,
            "mla_use_output_gate": True,
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
            "routed_expert_hidden_size": 16,
            "attn_res_block_size": 2,
            "num_experts": 8,
            "num_experts_per_token": 2,
            "num_shared_experts": 1,
            "moe_intermediate_size": 12,
            "linear_attn_config": {
                "kda_layers": [1, 2, 3],
                "full_attn_layers": [4],
                "num_heads": 4,
                "head_dim": 8,
                "short_conv_kernel_size": 4,
                "use_full_rank_gate": True,
                "gate_lower_bound": -5.0,
            },
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "mxfp4-pack-quantized",
            },
        },
        vision_config={
            "vt_hidden_size": 16,
            "vt_num_attention_heads": 2,
            "vt_num_hidden_layers": 2,
            "vt_intermediate_size": 32,
            "qkv_hidden_size": 32,
            "mm_hidden_size": 16,
            "text_hidden_size": 32,
        },
    )


def test_kimi_k3_config_preserves_model_contract():
    config = _tiny_k3_config()
    text = config.text_config
    assert config.model_type == "kimi_k3"
    assert config.hidden_size == 32
    assert config.vocab_size == text.vocab_size
    assert text.mla_use_nope is True
    assert text.mla_use_rope is False
    assert text.mla_use_output_gate is True
    assert text.qk_rope_head_dim == 4  # Keep the slice; only rotation is disabled.
    assert text.activation_situ_beta == 4.0
    assert text.activation_situ_linear_beta == 25.0
    assert text.routed_expert_hidden_size == 16
    assert text.is_kda_layer(0)
    assert not text.is_kda_layer(3)
    assert config.quantization_config["format"] == "mxfp4-pack-quantized"


def test_kimi_k3_model_cache_shape_includes_speculative_tokens(monkeypatch):
    import vllm_ascend.ops.kimi_kda_state as state_shape

    monkeypatch.setattr(state_shape, "is_conv_state_dim_first", lambda: False)
    config = _tiny_k3_config().text_config
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
        model_config=SimpleNamespace(hf_text_config=config),
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
    )

    actual = AscendKimiK3ForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    assert actual == ((6, 48), (2, 8, 8))


def test_kimi_k3_vision_config_exposes_vllm_aliases():
    vision = _tiny_k3_config().vision_config
    assert vision.hidden_size == vision.vt_hidden_size == 16
    assert vision.num_attention_heads == vision.vt_num_attention_heads == 2
    assert vision.num_hidden_layers == vision.vt_num_hidden_layers == 2
    assert vision.qkv_hidden_size == 32
    assert vision.patch_embed_proj_bias is False
    assert vision.attn_bias is False
    assert vision.linear_bias is False
    assert vision.norm_type == "rmsnorm"
    assert vision.mm_projector_type == "patchmergerv2"


def test_kimi_k3_mla_tp8_gate_width_matches_local_attention_heads(monkeypatch):
    class StubModule(nn.Module):
        pass

    captures = {}

    def fake_column_parallel(in_features, out_features, **kwargs):
        module = StubModule()
        captures[kwargs["prefix"]] = (in_features, out_features, module)
        return module

    def fake_row_parallel(in_features, out_features, **kwargs):
        module = StubModule()
        captures[kwargs["prefix"]] = (in_features, out_features, module)
        return module

    def fake_wrapper(*args, **kwargs):
        captures["wrapper_args"] = args
        captures["wrapper_kwargs"] = kwargs
        return StubModule()

    monkeypatch.setattr(kimi_k3_text, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(kimi_k3_text, "DeepSeekV2FusedQkvAProjLinear", lambda *args, **kwargs: StubModule())
    monkeypatch.setattr(kimi_k3_text, "ColumnParallelLinear", fake_column_parallel)
    monkeypatch.setattr(kimi_k3_text, "RowParallelLinear", fake_row_parallel)
    monkeypatch.setattr(kimi_k3_text, "RMSNorm", lambda *args, **kwargs: StubModule())
    monkeypatch.setattr(kimi_k3_text, "MLAModules", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(kimi_k3_text, "MultiHeadLatentAttentionWrapper", fake_wrapper)

    prefix = "model.layers.3.self_attn"
    KimiK3MLAAttention(
        _tiny_k3_config().text_config,
        hidden_size=7168,
        num_heads=96,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        q_lora_rank=1536,
        kv_lora_rank=512,
        prefix=prefix,
    )

    gate_in, gate_out, gate_module = captures[f"{prefix}.g_proj"]
    o_in, o_out, _ = captures[f"{prefix}.o_proj"]
    wrapper_args = captures["wrapper_args"]
    mla_modules = wrapper_args[8]

    assert (gate_in, gate_out) == (7168, 96 * 128)
    assert (o_in, o_out) == (96 * 128, 7168)
    assert wrapper_args[1] == 96 // 8
    assert wrapper_args[5] == 128
    assert mla_modules.g_proj is gate_module
    assert mla_modules.use_output_gate is True
    assert mla_modules.use_mla_rope is False


def test_kimi_k3_skips_explicit_move_for_meta_modules():
    module = nn.Linear(4, 4, device="meta")

    actual = _move_module_to_device(
        module,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert actual is module
    assert all(parameter.is_meta for parameter in module.parameters())


def test_kimi_k3_moves_non_meta_modules():
    module = nn.Linear(4, 4)

    actual = _move_module_to_device(
        module,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert actual is module
    assert all(parameter.device.type == "cpu" for parameter in module.parameters())
    assert all(parameter.dtype == torch.bfloat16 for parameter in module.parameters())


def test_kimi_k3_ignores_ascend_compressed_tensors_for_vision():
    quant_config = MagicMock()
    quant_config.get_name.return_value = "compressed-tensors"

    actual = AscendKimiK3ForConditionalGeneration._maybe_ignore_quant_config(quant_config)

    assert actual is None


def test_kimi_k3_loader_skips_only_modelslim_projector_rotation(monkeypatch):
    captured = {}

    class StubLoader:
        def __init__(self, model, *, skip_prefixes=None):
            captured["model"] = model
            captured["skip_prefixes"] = skip_prefixes

        def load_weights(self, weights, *, mapper=None):
            captured["weights"] = list(weights)
            captured["mapper"] = mapper
            return {"loaded"}

    monkeypatch.setattr(kimi_k3, "AutoWeightsLoader", StubLoader)
    model = AscendKimiK3ForConditionalGeneration.__new__(AscendKimiK3ForConditionalGeneration)
    nn.Module.__init__(model)
    weights = [("mm_projector.rot_proj.weight", torch.ones(1))]

    loaded = model.load_weights(iter(weights))

    assert loaded == {"loaded"}
    assert captured["model"] is model
    assert captured["skip_prefixes"] == ["mm_projector.rot_proj."]
    assert captured["weights"] == weights
    assert captured["mapper"] is model.hf_to_vllm_mapper


def test_kimi_k3_outer_mapper_covers_real_vision_and_projector_keys():
    names = [
        "vision_tower.encoder.blocks.0.wqkv.weight",
        "mm_projector.proj.0.weight",
        "mm_projector.proj.2.weight",
        "mm_projector.post_norm.weight",
        "language_model.model.layers.0.self_attn.g_proj.weight",
    ]

    assert AscendKimiK3ForConditionalGeneration.hf_to_vllm_mapper.apply_list(names) == [
        "vision_tower.encoder.blocks.0.wqkv.weight",
        "mm_projector.linear_1.weight",
        "mm_projector.linear_2.weight",
        "mm_projector.post_norm.weight",
        "language_model.model.layers.0.self_attn.g_proj.weight",
    ]


def test_kimi_k3_text_loader_maps_real_checkpoint_names_to_shards():
    calls = []

    def make_parameter(target_name):
        parameter = nn.Parameter(torch.zeros(1))

        def weight_loader(param, loaded_weight, *args, **kwargs):
            assert param is parameter
            calls.append(
                (
                    target_name,
                    int(loaded_weight.item()),
                    args,
                    kwargs,
                )
            )

        parameter.weight_loader = weight_loader
        return parameter

    params = {
        "model.layers.0.mlp.gate_up_proj.weight": make_parameter("model.layers.0.mlp.gate_up_proj.weight"),
        "model.layers.3.self_attn.fused_qkv_a_proj.weight": make_parameter(
            "model.layers.3.self_attn.fused_qkv_a_proj.weight"
        ),
        "model.layers.1.block_sparse_moe.experts.w13_weight": make_parameter(
            "model.layers.1.block_sparse_moe.experts.w13_weight"
        ),
        "model.layers.1.block_sparse_moe.experts.w2_weight": make_parameter(
            "model.layers.1.block_sparse_moe.experts.w2_weight"
        ),
        "model.layers.0.self_attn.g_proj.weight": make_parameter("model.layers.0.self_attn.g_proj.weight"),
    }
    model = KimiK3TextModel.__new__(KimiK3TextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_experts=1,
        num_hidden_layers=5,
        num_nextn_predict_layers=0,
    )
    model.named_parameters = lambda: iter(params.items())
    weights = [
        ("model.layers.0.mlp.gate_proj.weight", torch.tensor([10.0])),
        ("model.layers.0.mlp.up_proj.weight", torch.tensor([11.0])),
        ("model.layers.3.self_attn.q_a_proj.weight", torch.tensor([20.0])),
        (
            "model.layers.3.self_attn.kv_a_proj_with_mqa.weight",
            torch.tensor([21.0]),
        ),
        (
            "model.layers.1.block_sparse_moe.experts.0.w1.weight",
            torch.tensor([30.0]),
        ),
        (
            "model.layers.1.block_sparse_moe.experts.0.w3.weight",
            torch.tensor([31.0]),
        ),
        (
            "model.layers.1.block_sparse_moe.experts.0.w2.weight",
            torch.tensor([32.0]),
        ),
        ("model.layers.0.self_attn.g_proj.weight", torch.tensor([40.0])),
    ]

    loaded = model.load_weights(iter(weights))

    assert loaded == set(params)
    assert calls == [
        ("model.layers.0.mlp.gate_up_proj.weight", 10, (0,), {}),
        ("model.layers.0.mlp.gate_up_proj.weight", 11, (1,), {}),
        ("model.layers.3.self_attn.fused_qkv_a_proj.weight", 20, (0,), {}),
        ("model.layers.3.self_attn.fused_qkv_a_proj.weight", 21, (1,), {}),
        (
            "model.layers.1.block_sparse_moe.experts.w13_weight",
            30,
            ("model.layers.1.block_sparse_moe.experts.w13_weight",),
            {"expert_id": 0, "shard_id": "w1"},
        ),
        (
            "model.layers.1.block_sparse_moe.experts.w13_weight",
            31,
            ("model.layers.1.block_sparse_moe.experts.w13_weight",),
            {"expert_id": 0, "shard_id": "w3"},
        ),
        (
            "model.layers.1.block_sparse_moe.experts.w2_weight",
            32,
            ("model.layers.1.block_sparse_moe.experts.w2_weight",),
            {"expert_id": 0, "shard_id": "w2"},
        ),
        ("model.layers.0.self_attn.g_proj.weight", 40, (), {}),
    ]


def test_attention_residual_matches_reference_math():
    torch.manual_seed(7)
    prefix_sum = torch.randn(3, 4, dtype=torch.bfloat16)
    block_residual = torch.randn(3, 2, 4, dtype=torch.bfloat16)
    norm = nn.Module()
    norm.register_parameter("weight", nn.Parameter(torch.ones(4)))
    norm.variance_epsilon = 1e-5
    projection = nn.Linear(4, 1, bias=False)

    actual = _apply_attention_residual(prefix_sum, block_residual, projection, norm)

    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    values_fp32 = values.float()
    normalized = values_fp32 * torch.rsqrt(values_fp32.square().mean(-1, keepdim=True) + 1e-5)
    score_weight = norm.weight.float() * projection.weight.squeeze(0).float()
    probabilities = (normalized * score_weight).sum(-1).softmax(-1).unsqueeze(1)
    expected = torch.matmul(probabilities, values_fp32).squeeze(1).to(values.dtype)
    torch.testing.assert_close(actual, expected)


def test_kimi_k3_latent_moe_wiring_quantizes_only_fused_routed_experts(monkeypatch):
    class StubModule(nn.Module):
        pass

    replicated_quant_configs = {}
    fused_moe_kwargs = {}
    shared_mlp_kwargs = {}

    def fake_replicated_linear(*args, **kwargs):
        replicated_quant_configs[kwargs["prefix"]] = kwargs["quant_config"]
        return StubModule()

    def fake_fused_moe(**kwargs):
        fused_moe_kwargs.update(kwargs)
        return StubModule()

    def fake_shared_mlp(*args, **kwargs):
        shared_mlp_kwargs.update(kwargs)
        return StubModule()

    monkeypatch.setattr(kimi_k3_text, "ReplicatedLinear", fake_replicated_linear)
    monkeypatch.setattr(kimi_k3_text, "FusedMoE", fake_fused_moe)
    monkeypatch.setattr(kimi_k3_text, "KimiK3MLP", fake_shared_mlp)
    monkeypatch.setattr(kimi_k3_text, "RMSNorm", lambda *args, **kwargs: StubModule())
    quant_config = MagicMock(name="compressed_tensors_mxfp4")
    quant_config.get_name.return_value = "compressed-tensors"

    KimiK3MoE(
        _tiny_k3_config().text_config,
        quant_config=quant_config,
        prefix="model.layers.1.block_sparse_moe",
    )

    assert replicated_quant_configs == {
        "model.layers.1.block_sparse_moe.gate": None,
        "model.layers.1.block_sparse_moe.routed_expert_down_proj": None,
        "model.layers.1.block_sparse_moe.routed_expert_up_proj": None,
    }
    assert fused_moe_kwargs["quant_config"] is quant_config
    assert fused_moe_kwargs["prefix"] == "model.layers.1.block_sparse_moe.experts"
    assert fused_moe_kwargs["activation"] == "situ"
    assert shared_mlp_kwargs["quant_config"] is quant_config
    assert shared_mlp_kwargs["prefix"] == "model.layers.1.block_sparse_moe.shared_experts"


def test_kimi_k3_modelslim_quantizes_latent_moe_projections():
    modelslim = MagicMock()
    modelslim.get_name.return_value = "ascend"
    compressed_tensors = MagicMock()
    compressed_tensors.get_name.return_value = "compressed-tensors"

    assert _routed_latent_quant_config(modelslim) is modelslim
    assert _routed_latent_quant_config(compressed_tensors) is None
    assert _routed_latent_quant_config(None) is None


def test_decoder_registers_moe_under_checkpoint_module_name(monkeypatch):
    class StubModule(nn.Module):
        def __init__(self, with_marker: bool = False):
            super().__init__()
            if with_marker:
                self.marker = nn.Parameter(torch.ones(1))

    monkeypatch.setattr(kimi_k3_text, "KimiGatedDeltaNetAttention", lambda *args, **kwargs: StubModule())
    monkeypatch.setattr(kimi_k3_text, "KimiK3MoE", lambda *args, **kwargs: StubModule(with_marker=True))
    monkeypatch.setattr(kimi_k3_text, "RMSNorm", lambda *args, **kwargs: StubModule())
    monkeypatch.setattr(kimi_k3_text, "ReplicatedLinear", lambda *args, **kwargs: StubModule())

    vllm_config = MagicMock()
    vllm_config.quant_config = None
    layer = KimiK3DecoderLayer(
        _tiny_k3_config().text_config,
        vllm_config,
        prefix="model.layers.1",
    )

    assert hasattr(layer, "block_sparse_moe")
    assert not hasattr(layer, "mlp")
    assert "block_sparse_moe.marker" in dict(layer.named_parameters())


def test_processor_injects_k3_image_resolution():
    text = "a<|media_begin|>image<|media_content|><|media_pad|><|media_end|>b"
    chunks = [{"type": "image", "image": Image.new("RGB", (320, 240))}]
    actual = KimiK3Processor._inject_image_sizes(text, chunks)
    assert "image 320x240<|media_content|>" in actual


def test_cached_prompt_update_matches_k3_image_size_contract():
    class FakeTokenizer:
        def encode(self, text, *, add_special_tokens=False):
            assert add_special_tokens is False
            return {
                "<|media_begin|>image<|media_content|>": [10, 11],
                "<|media_begin|>image 320x240<|media_content|>": [10, 20, 21, 11],
                "<|media_end|>": [12],
            }[text]

        def __call__(self, texts):
            assert texts == ["<|media_begin|>image 320x240<|media_content|><|media_pad|><|media_end|>"]
            return {"input_ids": [[10, 20, 21, 11, 99, 12]]}

    class FakeImageProcessor:
        @staticmethod
        def preprocess(items, return_tensors=None):
            del items, return_tensors
            return {}

        @staticmethod
        def media_tokens_calculator(item):
            del item
            return 3

    processor = object.__new__(KimiK3MultiModalProcessor)
    tokenizer = FakeTokenizer()
    processor.info = SimpleNamespace(
        media_token_id=99,
        get_tokenizer=lambda: tokenizer,
        media_tokens_calculator=lambda item: 3,
    )
    item = {"type": "image", "image": Image.new("RGB", (320, 240))}
    mm_items = MultiModalDataItems({"vision_chunk": VisionChunkProcessorItems([item])})

    update = processor._get_prompt_updates(mm_items, {}, MagicMock())[0]
    replacement = update.replacement(0)

    assert update.target == [10, 11, 99, 12]
    assert replacement.full == [10, 20, 21, 11, 99, 99, 99, 12]
    assert replacement.is_embed is not None
    assert replacement.is_embed(None, replacement.full).tolist() == [
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        False,
    ]

    # The uncached joint HF-processor path injects the same WxH text and then
    # expands the media token. It must produce exactly the cached replacement.
    joint_processor = KimiK3Processor(FakeImageProcessor(), tokenizer, 99)
    joint = joint_processor(
        text="<|media_begin|>image<|media_content|><|media_pad|><|media_end|>",
        vision_chunks=[item],
    )
    assert joint["input_ids"][0] == replacement.full


def test_cached_prompt_update_preserves_multi_image_size_order():
    class FakeTokenizer:
        def encode(self, text, *, add_special_tokens=False):
            assert add_special_tokens is False
            tokens = {
                "<|media_begin|>image<|media_content|>": [10, 11],
                "<|media_begin|>image 320x240<|media_content|>": [10, 320, 240, 11],
                "<|media_begin|>image 640x480<|media_content|>": [10, 640, 480, 11],
                "<|media_end|>": [12],
            }
            return tokens[text]

    items = [
        {"type": "image", "image": Image.new("RGB", (320, 240))},
        {"type": "image", "image": Image.new("RGB", (640, 480))},
    ]
    processor = object.__new__(KimiK3MultiModalProcessor)
    processor.info = SimpleNamespace(
        media_token_id=99,
        get_tokenizer=FakeTokenizer,
        media_tokens_calculator=lambda item: 3 if item["image"].size[0] == 320 else 2,
    )
    mm_items = MultiModalDataItems({"vision_chunk": VisionChunkProcessorItems(items)})

    update = processor._get_prompt_updates(mm_items, {}, MagicMock())[0]

    assert update.replacement(0).full == [10, 320, 240, 11, 99, 99, 99, 12]
    assert update.replacement(1).full == [10, 640, 480, 11, 99, 99, 12]


def test_kimi_k3_vision_patch_embed_uses_reference_bicubic(monkeypatch):
    captured = {}

    def fake_pos_emb(**kwargs):
        captured.update(kwargs)
        return nn.Identity()

    monkeypatch.setattr(
        kimi_k3_vit,
        "Learnable2DInterpPosEmbDivided_fixed",
        fake_pos_emb,
    )

    KimiK3VisionPatchEmbed(_tiny_k3_config().vision_config)

    assert captured["interpolation_mode"] == "bicubic"


def test_k3_state_shape_inputs_are_full_rank():
    config = _tiny_k3_config().text_config
    linear = config.linear_attn_config
    assert linear["use_full_rank_gate"] is True
    # K3's full-rank KDA gate projects hidden_size independently to every
    # state head.  Its output width is not required to equal hidden_size.
    assert linear["num_heads"] == config.num_attention_heads
    assert linear["head_dim"] == config.v_head_dim
    assert linear["gate_lower_bound"] == -5.0
