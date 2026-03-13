from __future__ import annotations

import shutil

import pytest

from vllm_ascend.agent_runtime import (
    RawRequest,
    deployment_artifact_packager,
    pack,
    resolve,
    vllm_ascend_assistant,
)
from vllm_ascend.agent_runtime.contracts import ContractError, run_contract_checks
from vllm_ascend.agent_runtime.kb import build_local, doctor


@pytest.fixture()
def a3_resolve_result(agent_repo_root):
    return resolve(
        agent_repo_root,
        request_id="req-p6-a3",
        overrides={
            "soc": "A3",
            "cann": "8.5.0",
            "torch": "2.9.0",
            "torch_npu": "2.9.0",
            "python": "3.11",
        },
    )


def test_h601_resolve_includes_substrate_shards_when_available(a3_resolve_result) -> None:
    assert "hw_soc_detail" in a3_resolve_result["selected_shards"]
    assert "hw_runtime_caps" in a3_resolve_result["selected_shards"]
    assert "cann_op_constraints" in a3_resolve_result["selected_shards"]
    assert "torch_npu_bindings" in a3_resolve_result["selected_shards"]


def test_h601_repo_only_fallback_still_works_without_substrate(agent_repo_root, tmp_path) -> None:
    result = resolve(
        agent_repo_root,
        request_id="req-p6-fallback",
        overrides={
            "soc": "A3",
            "cann": "8.5.0",
            "torch": "2.9.0",
            "torch_npu": "unknown",
            "python": "3.11",
        },
    )
    emit_sqlite = tmp_path / "fallback.sqlite"
    build_local(agent_repo_root, resolve_result=result, emit_sqlite=emit_sqlite)
    response = pack(
        agent_repo_root,
        request={
            "schema_version": "kb-pack-request/v2",
            "request_id": "req-p6-fallback-pack",
            "created_at": "2026-03-13T14:40:00Z",
            "intent": "deployment_lookup",
            "repo_root": ".",
            "resolve_policy": "auto",
            "logical_domains": ["deployment_config", "validation_evidence"],
            "physical_shard_hints": ["repo_semantics", "validation"],
            "selectors": {
                "files": [],
                "symbols": [],
                "entities": [],
                "errors": [],
                "models": ["qwen3-32b-w8a8"],
                "features": ["tp4"],
                "hw": ["A3"],
                "commits": [],
                "prs": [],
                "versions": [],
                "configs": [],
            },
            "must_have": ["deployment baseline"],
            "nice_to_have": ["validation anchor"],
            "evidence_refs": [],
            "budget_token_cap": 1500,
            "max_atoms": 10,
            "max_hops": 1,
            "include_evidence_stubs": True,
            "stop_after_first_sufficient": True,
            "emit_path": ".agents/kb/local/capsules/req-p6-fallback-pack.json",
        },
        resolve_result=result,
        merged_pack=emit_sqlite,
    )
    assert response["capsule_text"]
    assert result["match_level"] in {"compatible", "unknown"}


def test_h602_doctor_and_contract_checks_agree_on_broken_example(tmp_path, agent_repo_root) -> None:
    temp_root = tmp_path / "broken-root"
    shutil.copytree(agent_repo_root / ".agents", temp_root / ".agents")
    broken = temp_root / ".agents" / "kb" / "examples" / "selector-plan.performance.atomic.json"
    text = broken.read_text(encoding="utf-8").replace('"task_family": "performance_analysis"', '"task_family": 7')
    broken.write_text(text, encoding="utf-8")
    with pytest.raises(ContractError):
        run_contract_checks(root=temp_root)
    with pytest.raises(ContractError):
        doctor(root=temp_root)


def test_p6_deepseek_a3_request_routes_to_performance_expectation() -> None:
    result = vllm_ascend_assistant(
        RawRequest(
            request_id="req-p6-dsv3",
            user_text="帮我分析deepseekv3在A3单机上的理论性能",
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=[],
            created_at_hint="2026-03-13T14:40:01Z",
        )
    )
    assert result["selector_plan"]["task_family"] == "performance_analysis"
    assert result["selector_plan"]["consumer_id"] == "model-expected-performance-estimator"
    assert result["selector_seed"]["normalized_entities"]["models"] == ["deepseek-v3"]
    assert result["selector_seed"]["normalized_entities"]["hw"] == ["A3"]


def test_p6_qwen3_a3_deployment_request_routes_to_deployment() -> None:
    result = vllm_ascend_assistant(
        RawRequest(
            request_id="req-p6-qwen-deploy",
            user_text="帮我看下A3上单卡跑qwen3 32b w8a8的部署命令",
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=[],
            created_at_hint="2026-03-13T14:40:02Z",
        )
    )
    assert result["selector_plan"]["task_family"] == "deployment_execution"
    assert result["selector_seed"]["normalized_entities"]["models"] == ["qwen3-32b-w8a8"]
    assert result["selector_seed"]["normalized_entities"]["hw"] == ["A3"]


def test_p6_qwen3_dense_a3_single_card_request_stays_qwen3_32b() -> None:
    result = vllm_ascend_assistant(
        RawRequest(
            request_id="req-p6-qwen-dense-deploy",
            user_text="给我一个A3上单卡部署qwen3 32B的部署脚本",
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=[],
            created_at_hint="2026-03-13T14:40:02Z",
        )
    )
    assert result["selector_plan"]["task_family"] == "deployment_execution"
    assert result["selector_seed"]["normalized_entities"]["models"] == ["qwen3-32b"]
    assert result["selector_seed"]["normalized_entities"]["hw"] == ["A3"]


def test_p6_model_expectation_pack_uses_deepseek_a3_envelope(a3_resolve_result, agent_repo_root, tmp_path) -> None:
    emit_sqlite = tmp_path / "a3.sqlite"
    build_local(agent_repo_root, resolve_result=a3_resolve_result, emit_sqlite=emit_sqlite)
    response = pack(
        agent_repo_root,
        request={
            "schema_version": "kb-pack-request/v2",
            "request_id": "req-p6-deepseek-pack",
            "created_at": "2026-03-13T14:40:03Z",
            "intent": "model_expectation",
            "repo_root": ".",
            "resolve_policy": "auto",
            "logical_domains": ["validation_evidence", "deployment_config", "ascend_foundation"],
            "physical_shard_hints": ["validation", "repo_semantics", "hw_runtime_caps", "hw_soc_detail", "cann_op_constraints"],
            "selectors": {
                "files": [],
                "symbols": [],
                "entities": [],
                "errors": [],
                "models": ["deepseek-v3"],
                "features": ["prefill", "decode"],
                "hw": ["A3"],
                "commits": [],
                "prs": [],
                "versions": [],
                "configs": [],
            },
            "must_have": ["expected TTFT range", "expected throughput range", "memory headroom assumptions"],
            "nice_to_have": ["graph mode sensitivity"],
            "evidence_refs": [],
            "budget_token_cap": 1500,
            "max_atoms": 10,
            "max_hops": 1,
            "include_evidence_stubs": True,
            "stop_after_first_sufficient": True,
            "emit_path": ".agents/kb/local/capsules/req-p6-deepseek-pack.json",
        },
        resolve_result=a3_resolve_result,
        merged_pack=emit_sqlite,
    )
    assert "deepseek-v3" in response["capsule_text"].lower()
    assert "a3" in response["capsule_text"].lower()
    assert "qwen3-next-32b" not in response["capsule_text"].lower()


def test_p6_deployment_pack_uses_qwen3_a3_constraints(a3_resolve_result, agent_repo_root, tmp_path) -> None:
    emit_sqlite = tmp_path / "a3.sqlite"
    build_local(agent_repo_root, resolve_result=a3_resolve_result, emit_sqlite=emit_sqlite)
    response = pack(
        agent_repo_root,
        request={
            "schema_version": "kb-pack-request/v2",
            "request_id": "req-p6-qwen-pack",
            "created_at": "2026-03-13T14:40:04Z",
            "intent": "deployment_lookup",
            "repo_root": ".",
            "resolve_policy": "auto",
            "logical_domains": ["deployment_config", "validation_evidence", "ascend_foundation"],
            "physical_shard_hints": ["validation", "repo_semantics", "hw_runtime_caps", "hw_soc_detail"],
            "selectors": {
                "files": [],
                "symbols": [],
                "entities": [],
                "errors": [],
                "models": ["qwen3-32b-w8a8"],
                "features": ["tp4"],
                "hw": ["A3"],
                "commits": [],
                "prs": [],
                "versions": [],
                "configs": [],
            },
            "must_have": ["deployment baseline", "single-card support"],
            "nice_to_have": ["launch command"],
            "evidence_refs": [],
            "budget_token_cap": 1500,
            "max_atoms": 10,
            "max_hops": 1,
            "include_evidence_stubs": True,
            "stop_after_first_sufficient": True,
            "emit_path": ".agents/kb/local/capsules/req-p6-qwen-pack.json",
        },
        resolve_result=a3_resolve_result,
        merged_pack=emit_sqlite,
    )
    lower = response["capsule_text"].lower()
    assert "qwen3-32b-w8a8" in lower
    assert "a3" in lower
    assert "tp4" in lower or "4 npu" in lower or "single-card" in lower


def test_p6_qwen3_dense_pack_rejects_single_card_and_points_to_tp4(a3_resolve_result, agent_repo_root, tmp_path) -> None:
    emit_sqlite = tmp_path / "a3-dense.sqlite"
    build_local(agent_repo_root, resolve_result=a3_resolve_result, emit_sqlite=emit_sqlite)
    response = pack(
        agent_repo_root,
        request={
            "schema_version": "kb-pack-request/v2",
            "request_id": "req-p6-qwen-dense-pack",
            "created_at": "2026-03-13T14:40:05Z",
            "intent": "deployment_lookup",
            "repo_root": ".",
            "resolve_policy": "auto",
            "logical_domains": ["deployment_config", "validation_evidence", "ascend_foundation"],
            "physical_shard_hints": ["validation", "repo_semantics", "hw_runtime_caps", "hw_soc_detail"],
            "selectors": {
                "files": [],
                "symbols": [],
                "entities": [],
                "errors": [],
                "models": ["qwen3-32b"],
                "features": ["single_card"],
                "hw": ["A3"],
                "commits": [],
                "prs": [],
                "versions": [],
                "configs": [],
            },
            "must_have": ["deployment baseline", "single-card support"],
            "nice_to_have": ["launch command"],
            "evidence_refs": [],
            "budget_token_cap": 1500,
            "max_atoms": 10,
            "max_hops": 1,
            "include_evidence_stubs": True,
            "stop_after_first_sufficient": True,
            "emit_path": ".agents/kb/local/capsules/req-p6-qwen-dense-pack.json",
        },
        resolve_result=a3_resolve_result,
        merged_pack=emit_sqlite,
    )
    card = deployment_artifact_packager(
        {
            "request_id": "req-p6-qwen-dense-pack",
            "plan_id": "plan-req-p6-qwen-dense-pack",
            "task_family": "deployment_execution",
            "work_package_id": "wp-qwen3-dense-a3-script",
            "selectors": {
                "files": [],
                "symbols": [],
                "entities": [],
                "errors": [],
                "models": ["qwen3-32b"],
                "features": ["single_card"],
                "hw": ["A3"],
                "commits": [],
                "prs": [],
                "versions": [],
                "configs": [],
            },
        },
        response,
    )
    lower = response["capsule_text"].lower()
    assert "qwen3-32b" in lower
    assert "tp4" in lower
    assert "single-card" in lower or "单卡" in response["capsule_text"]
    assert card["notes"] is not None
    assert "vllm serve" in card["notes"]
    assert "--tensor-parallel-size 4" in card["notes"]
    assert "--quantization ascend" not in card["notes"]
    assert "single-card" in card["notes"].lower() or "单卡" in card["notes"]


def test_p6_skill_docs_force_runtime_first_for_deployment(agent_repo_root) -> None:
    assistant_skill = (
        agent_repo_root / ".agents" / "skills" / "vllm-ascend-assistant" / "SKILL.md"
    ).read_text(encoding="utf-8")
    deployment_skill = (
        agent_repo_root / ".agents" / "skills" / "deployment_execution" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert "runtime.py" in assistant_skill
    assert "Do not grep raw docs first" in assistant_skill
    assert "deployment-intake" in assistant_skill
    assert "deployment-artifact-packager" in deployment_skill
    assert "Do not fabricate" in deployment_skill
    assert "single-card" in deployment_skill
