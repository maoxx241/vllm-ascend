# SPDX-License-Identifier: Apache-2.0
"""Static regressions for the migrated SiTU quantization operators."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEQUANT_ROOT = ROOT / "csrc/moe/dequant_situ_quant"
MX_ROOT = ROOT / "csrc/moe/situ_mx_quant"
K3_SHARED_TP_WIDTHS = (12288, 6144, 3072, 1536, 768)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_dequant_situ_quant_is_registered_for_a2_and_a3():
    cmake = _read(DEQUANT_ROOT / "op_host/CMakeLists.txt")
    op_def = _read(DEQUANT_ROOT / "op_host/dequant_situ_quant_def.cpp")

    assert "ascend910b ascend910_93" in cmake
    assert 'AddConfig("ascend910b")' in op_def
    assert 'AddConfig("ascend910_93")' in op_def


def test_situ_mx_quant_is_gated_to_a5():
    cmake = _read(MX_ROOT / "op_host/CMakeLists.txt")
    op_def = _read(MX_ROOT / "op_host/situ_mx_quant_def.cpp")

    assert 'if(NOT "ascend950" IN_LIST ASCEND_COMPUTE_UNIT)' in cmake
    assert 'AddConfig("ascend950", regbaseCfg)' in op_def
    assert "situ_mx_quant_apt" in op_def


def test_build_script_selects_each_op_only_for_supported_socs():
    build_script = _read(ROOT / "csrc/build_aclnn.sh")

    a2_block = build_script.split('elif [[ "$SOC_VERSION" =~ ^ascend910_93 ]]')[0]
    a3_block = build_script.split('elif [[ "$SOC_VERSION" =~ ^ascend910_93 ]]')[1].split(
        'elif [[ "$SOC_VERSION" =~ ^ascend950 ]]'
    )[0]
    a5_block = build_script.split('elif [[ "$SOC_VERSION" =~ ^ascend950 ]]')[1].split("else")[0]

    assert '"dequant_situ_quant"' in a2_block
    assert '"dequant_situ_quant"' in a3_block
    assert '"situ_mx_quant"' in a5_block
    assert '"situ_mx_quant"' not in a2_block + a3_block
    assert '"dequant_situ_quant"' not in a5_block


def test_editable_build_does_not_reuse_generated_custom_op_artifacts():
    build_script = _read(ROOT / "csrc/build_aclnn.sh")

    assert "rm -rf -- build output build_out" in build_script
    assert "preserving csrc/build" not in build_script


def test_situ_mx_quant_binary_manifest_covers_both_fp8_formats():
    manifest_path = MX_ROOT / "op_host/config/ascend950/situ_mx_quant_binary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["op_type"] == "SituMxQuant"
    assert {item["bin_filename"] for item in manifest["op_list"]} == {
        "SituMxQuant_bf16_e4m3fn",
        "SituMxQuant_bf16_e5m2",
    }


def test_torch_bindings_and_meta_kernels_are_registered():
    binding = _read(ROOT / "csrc/torch_binding.cpp")
    meta = _read(ROOT / "csrc/torch_binding_meta.cpp")

    assert '"dequant_situ_quant(Tensor x, "' in binding
    assert 'ops.impl("dequant_situ_quant", torch::kPrivateUse1' in binding
    assert '"situ_mx_quant(Tensor x, "' in binding
    assert 'ops.impl("situ_mx_quant", torch::kPrivateUse1' in binding
    assert 'ops.impl("dequant_situ_quant", &vllm_ascend::meta::dequant_situ_quant_meta)' in meta
    assert 'ops.impl("situ_mx_quant", &vllm_ascend::meta::situ_mx_quant_meta)' in meta
    assert "Tensor? weight_scale=None" in binding
    assert "Tensor? activation_scale=None" in binding
    assert "Tensor? bias=None" in binding
    assert "Tensor? group_index=None" in binding


def test_framework_keeps_situ_in_one_dedicated_branch():
    moe_mlp = _read(ROOT / "vllm_ascend/ops/fused_moe/moe_mlp.py")
    contracts = _read(ROOT / "vllm_ascend/ops/fused_moe/moe_stage_contracts.py")

    assert "def _w4a8_situ_apply_mlp(" in moe_mlp
    assert "torch.ops._C_ascend.dequant_situ_quant(" in moe_mlp
    assert "torch.ops._C_ascend.situ_mx_quant(" in moe_mlp
    assert "SituActivationConfig" in contracts
    assert "situ_beta:" not in contracts
    assert "situ_linear_beta:" not in contracts


def test_torch_adapters_preserve_output_contracts():
    dequant_adapter = _read(DEQUANT_ROOT / "dequant_situ_quant_torch_adpt.h")
    mx_adapter = _read(MX_ROOT / "situ_mx_quant_torch_adpt.h")

    assert "K3_ROUTED_SITU_INPUT_WIDTH" not in dequant_adapter
    assert "is_k3_shared_situ_input_width" not in dequant_adapter
    assert "input_width / 2" in dequant_adapter
    assert "aclnnDequantSituQuant" in dequant_adapter
    assert "y_shape.back() /= 2" in mx_adapter
    assert "MX_BLOCK_SPAN = 64" in mx_adapter
    assert "at::kFloat8_e8m0fnu" in mx_adapter
    assert "aclnnSituMxQuant" in mx_adapter


def test_dequant_dynamic_scale_is_packed_after_int8_output():
    kernel = _read(DEQUANT_ROOT / "op_kernel/dequant_situ_quant.h")

    assert kernel.count(
        "int64_t scaleIdx = (outputWidth_ + static_cast<int64_t>(sizeof(float)) - 1) / sizeof(float);"
    ) == 2
    assert "scaleValue <= 0.0f" in kernel
    assert "scaleValue = 1.0f" in kernel
    assert "WholeReduceMax" in kernel


def test_dequant_k3_kernel_follows_latest_operator_int32_fix():
    kernel = _read(DEQUANT_ROOT / "op_kernel/dequant_situ_quant.h")
    k3_kernel = kernel.split("class DequantSituQuantK3Kernel", 1)[1]
    compute_row = k3_kernel.split("__aicore__ inline void ComputeRow", 1)[1].split(
        "__aicore__ inline void DynamicQuant", 1
    )[0]

    # Both the numeric INT32 accumulator and BF16 input must be converted to
    # FP32 values before SiTU. INT32 reuses its input storage for the result.
    assert "xLocalF32 = xLocal.template ReinterpretCast<float>();" in compute_row
    assert compute_row.count("Cast(xLocalF32, xLocal, RoundMode::CAST_NONE, inputWidth_);") == 1
    bf16_branch = compute_row.split("} else {", 1)[1].split("}", 1)[0]
    assert "Cast(xLocalF32, xLocal, RoundMode::CAST_NONE, inputWidth_);" not in bf16_branch
    assert (
        "}\n        Cast(xLocalF32, xLocal, RoundMode::CAST_NONE, inputWidth_);"
        in compute_row
    )


def test_dequant_host_and_kernel_share_group_and_width_tiling_contract():
    tiling_header = _read(DEQUANT_ROOT / "op_host/dequant_situ_quant_tiling.h")
    tiling = _read(DEQUANT_ROOT / "op_host/dequant_situ_quant_tiling.cpp")
    infer_shape = _read(DEQUANT_ROOT / "op_host/dequant_situ_quant_infershape.cpp")
    kernel = _read(DEQUANT_ROOT / "op_kernel/dequant_situ_quant.h")
    meta = _read(ROOT / "csrc/torch_binding_meta.cpp")

    for field in ("inputWidth", "outputWidth", "expertNum", "hasGroupIndex"):
        assert f"TILING_DATA_FIELD_DEF(uint32_t, {field});" in tiling_header
        assert f"tilingData.set_{field}" in tiling
        assert f"tilingData_->{field}" in kernel
    assert "TILING_DATA_FIELD_DEF(uint32_t, dequantBiasIsEmpty);" in tiling_header
    assert "tilingData.set_dequantBiasIsEmpty" in tiling
    assert "tilingData_->dequantBiasIsEmpty" in kernel
    assert "DSQ_INT32_DYNAMIC = 30000" in tiling_header
    assert "DSQ_BF16_DYNAMIC = 40000" in tiling_header
    assert "return DSQ_INT32_DYNAMIC;" in tiling
    assert "return DSQ_BF16_DYNAMIC;" in tiling
    assert "requestedRows > remainingRows ? remainingRows : requestedRows" in kernel
    assert "ProcessGroup(0, rowLen_, 0)" in kernel
    assert "hasGroupIndex_ ?" in tiling
    assert "ValidateInt32Contract()" in tiling
    assert "CheckInputShapesBF16()" in tiling
    assert "K3_ROUTED_INPUT_WIDTH" not in tiling
    assert "IsK3SharedInputWidth" not in tiling
    assert "K3_ROUTED_INPUT_WIDTH" not in infer_shape
    assert "IsK3SharedInputWidth" not in infer_shape
    assert "K3_ROUTED_INPUT_WIDTH" not in meta
    assert "is_k3_shared_input_width" not in meta
    assert "const c10::SymInt input_width = x.sym_size(1);" in meta
    assert "input_width % 2 == 0" in meta
    assert "y_shape.back() = input_width / 2;" in meta


def test_dequant_shared_tp_widths_are_safe_for_vector_reduce_and_ub_plan():
    kernel = _read(DEQUANT_ROOT / "op_kernel/dequant_situ_quant.h")

    for input_width in K3_SHARED_TP_WIDTHS:
        output_width = input_width // 2
        vector_cycles = output_width // 64
        assert output_width % 64 == 0
        assert 1 <= vector_cycles - 1 <= 255

        # INT32 x + FP32 weight + FP32 temporary + INT8 y/scale padding,
        # matching the bias-free shared-expert runtime contract.
        required_ub_bytes = 4 * input_width + 4 * input_width + 4 * input_width
        required_ub_bytes += output_width + 32 + 1024
        assert required_ub_bytes < 192 * 1024

    assert "static_cast<uint8_t>(vectorCycles - 1)" in kernel
    assert "inputWidth_ * static_cast<int64_t>(sizeof(float))" in kernel


def test_migrated_sources_only_use_resolvable_includes():
    dequant_header = _read(DEQUANT_ROOT / "op_host/dequant_situ_quant_tiling.h")
    dequant_tiling = _read(DEQUANT_ROOT / "op_host/dequant_situ_quant_tiling.cpp")
    tiling = _read(MX_ROOT / "op_host/arch35/situ_mx_quant_tiling_arch35.cpp")
    tiling_header = _read(MX_ROOT / "op_host/arch35/situ_mx_quant_tiling_arch35.h")
    infer_shape = _read(MX_ROOT / "op_host/situ_mx_quant_infershape.cpp")
    common = _read(MX_ROOT / "op_kernel/arch35/situ_mx_quant_common.h")
    kernel = common + _read(MX_ROOT / "op_kernel/arch35/situ_mx_quant_axis_last.h")

    assert '"../../dequant_swiglu_quant/tiling_base/tiling_base.h"' in dequant_header
    assert '"../../dequant_swiglu_quant/tiling_base/tiling_templates_registry.h"' in dequant_header
    assert '"../../dequant_swiglu_quant/tiling_base/tiling_templates_registry.h"' in dequant_tiling
    assert '"../../dequant_swiglu_quant/tiling_base/tiling_util.h"' in dequant_tiling
    assert "Ops::NN::OpTiling::IsRegbaseSocVersion(context_)" in dequant_tiling
    assert '"op_host/tiling_' not in dequant_header + dequant_tiling
    assert '"quant/situ_mx_quant/' not in tiling
    assert '"op_host/tiling_' not in tiling_header + tiling
    assert '"log/log.h"' in tiling
    assert '"../../op_kernel/arch35/situ_mx_quant_tiling_key.h"' in tiling
    assert '"../inc/platform.h"' in common
    assert '"../inc/kernel_utils.h"' in common
    assert '"op_kernel/math_util.h"' not in kernel
    assert '"op_kernel/platform_util.h"' not in kernel
    assert "Ops::Base::IsUnknownRank" not in infer_shape
    assert "Ops::Base::SetUnknownRank" not in infer_shape
    assert (MX_ROOT / "op_kernel/inc/platform.h").is_file()
    assert (MX_ROOT / "op_kernel/inc/kernel_utils.h").is_file()
