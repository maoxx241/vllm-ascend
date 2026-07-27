# SPDX-License-Identifier: Apache-2.0
"""Static regressions for the recurrent KDA device-metadata contract."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OP_ROOT = ROOT / "csrc/attention/recurrent_kda"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_recurrent_kda_uses_vllm_ascend_apache_headers():
    source_suffixes = {".cpp", ".h", ".py", ".txt"}
    sources = [OP_ROOT / "CMakeLists.txt", OP_ROOT / "op_host/CMakeLists.txt"]
    sources.extend(path for path in OP_ROOT.rglob("*") if path.suffix in source_suffixes)

    for path in sources:
        source = _read(path)
        assert "SPDX-License-Identifier: Apache-2.0" in source, path
        assert "CANN Open Software License Agreement" not in source, path


def test_aclnn_uses_device_cu_seqlens_and_mutable_state():
    header = _read(OP_ROOT / "op_host/op_api/aclnn_recurrent_kda.h")
    l0_source = _read(OP_ROOT / "op_host/op_api/recurrent_kda.cpp")

    assert "aclTensor *stateRef" in header
    assert "const aclTensor *cuSeqlens" in header
    assert "aclIntArray" not in header
    assert "const aclTensor *finalState" not in header
    assert "OP_OUTPUT(out, stateRef)" in l0_source


def test_tiling_processor_owns_context_and_supports_2d_slots():
    source = _read(OP_ROOT / "op_host/recurrent_kda_tiling_processor.h")

    assert "RecurrentKdaTilingContext ctx_;" in source
    assert "const RecurrentKdaTilingContext &ctx_;" not in source
    assert "speculative [seq_num,max_step]" in source
    assert "ssmStateStride" in source


def test_kernel_skips_empty_sequences_before_state_metadata_access():
    source = _read(OP_ROOT / "op_kernel/recurrent_kda.h")

    empty_skip = source.index("if (seqLen64 == 0)")
    slot_validation = source.index("ValidateStateSlots(batch_i, seq0, seqLen)")
    state_prefetch = source.index("PrefetchState(nextStateOffset, nextSingleV)")
    assert empty_skip < slot_validation < state_prefetch
    assert "batchIdx * ssmStateStride_" in source
    assert "stateSlot >= static_cast<int64_t>(stateCapacity_)" in source


def test_cu_seqlens_uses_fla_prefix_sum_semantics():
    kernel_paths = [
        OP_ROOT / "op_kernel/recurrent_kda.h",
        OP_ROOT / "op_kernel/arch35/recurrent_kda.h",
    ]

    for kernel_path in kernel_paths:
        source = _read(kernel_path)
        assert "int64_t seq0 = cuSeqlensGm_.GetValue(batch_i)" in source
        assert "int64_t seq1 = cuSeqlensGm_.GetValue(batch_i + 1)" in source
        assert "int64_t seqLen64 = seq1 - seq0" in source
        assert "if (seq0 != 0)" in source
        assert "return seq0 <= static_cast<int64_t>(T_)" in source
        assert "return seq0 == static_cast<int64_t>(T_)" not in source


def test_kernel_uses_generated_state_dtype_macro():
    source = _read(OP_ROOT / "op_kernel/recurrent_kda.cpp")

    assert "DTYPE_STATE" in source
    assert "DTYPE_INITIAL_STATE" not in source


def test_arch35_kernel_has_dedicated_micro_api_implementation():
    source = _read(OP_ROOT / "op_kernel/arch35/recurrent_kda.h")

    assert '#include "../recurrent_kda.h"' not in source
    assert "using namespace AscendC::MicroAPI;" in source
    assert "__VEC_SCOPE__" in source
    assert "inline void MatVecMul" in source
    assert "inline void ProcessKQ" in source
    assert "inline void ReduceSumDispatch" in source


def test_torch_binding_preserves_mutation_and_accepts_tnd():
    adapter = _read(OP_ROOT / "recurrent_kda_torch_adpt.h")
    schema = _read(ROOT / "csrc/torch_binding.cpp")

    assert 'const char* layout = is_tnd ? "TND" : "BSND";' in adapter
    assert "        layout," in adapter
    assert "speculative [seq_num,max_step]" in adapter
    assert "at::Tensor final_state = initial_state" in adapter
    assert "Tensor(a!) initial_state" in schema
    assert "Tensor cu_seqlens" in schema
    assert "-> Tensor output" in schema
    assert "Tensor(a!) final_state" not in schema
