from __future__ import annotations

from pathlib import Path

import pytest

from vllm_ascend.agent_runtime import (
    RawRequest,
    deployment_artifact_packager,
    pack,
    resolve,
    vllm_ascend_assistant,
)
from vllm_ascend.agent_runtime.kb import build_local
from vllm_ascend.agent_runtime.skill_lint import lint_runtime_first_skills


def _pack_request(
    *,
    request_id: str,
    intent: str,
    selectors: dict,
    domains: list[str] | None = None,
    must_have: list[str] | None = None,
    nice_to_have: list[str] | None = None,
) -> dict:
    return {
        "schema_version": "kb-pack-request/v2",
        "request_id": request_id,
        "created_at": "2026-03-13T18:00:00Z",
        "intent": intent,
        "repo_root": ".",
        "resolve_policy": "auto",
        "logical_domains": domains or ["deployment_config", "validation_evidence", "ascend_foundation"],
        "physical_shard_hints": ["validation", "repo_semantics", "hw_runtime_caps", "hw_soc_detail"],
        "selectors": selectors,
        "must_have": must_have or ["deployment baseline"],
        "nice_to_have": nice_to_have or ["launch command"],
        "evidence_refs": [],
        "budget_token_cap": 1500,
        "max_atoms": 10,
        "max_hops": 1,
        "include_evidence_stubs": True,
        "stop_after_first_sufficient": True,
        "emit_path": f".agents/kb/local/capsules/{request_id}.json",
    }


def _atom(response: dict, prefix: str) -> dict:
    for atom in response["atoms"]:
        if atom["atom_id"].startswith(prefix):
            return atom
    raise AssertionError(f"expected atom with prefix {prefix}")


@pytest.fixture()
def a3_fp8_resolve_result(agent_repo_root):
    return resolve(
        agent_repo_root,
        request_id="req-p6-fp8-a3",
        overrides={
            "soc": "A3",
            "cann": "8.5.0",
            "torch": "2.9.0",
            "torch_npu": "2.9.0",
            "python": "3.11",
        },
    )


@pytest.fixture()
def a3_fp8_sqlite(agent_repo_root, a3_fp8_resolve_result, tmp_path):
    emit_sqlite = tmp_path / "a3-fp8.sqlite"
    build_local(agent_repo_root, resolve_result=a3_fp8_resolve_result, emit_sqlite=emit_sqlite)
    return emit_sqlite


@pytest.mark.parametrize(
    "prompt",
    [
        "我想要在A3单卡上跑Qwen3 32b的fp8权重，给我一个拉起命令吧",
        "A3 1卡怎么跑 qwen3 32b 原生 fp8 权重",
        "single-card deployment for qwen3 32b native fp8 weights on A3",
    ],
)
def test_h701_native_fp8_requests_route_to_design_analysis(prompt: str) -> None:
    result = vllm_ascend_assistant(
        RawRequest(
            request_id="req-h701",
            user_text=prompt,
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=[],
            created_at_hint="2026-03-13T18:00:01Z",
        )
    )
    selectors = result["selector_seed"]["normalized_entities"]
    assert result["selector_plan"]["task_family"] == "design_analysis"
    assert result["selector_plan"]["execution_mode"] == "spec_plan_workflow"
    assert result["selector_plan"]["query_stage"] == "spec_plan"
    assert selectors["models"] == ["qwen3-32b"]
    assert selectors["hw"] == ["A3"]
    assert "weight_fp8_native" in selectors["features"]
    assert "quant_w8a8" not in selectors["features"]


def test_h702_modelslim_quantized_weight_path_stays_in_deployment() -> None:
    result = vllm_ascend_assistant(
        RawRequest(
            request_id="req-h702",
            user_text="我已经拿到了 ModelSlim 量化后的 qwen3 32b w8a8 权重，给我 A3 单卡部署命令",
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=[],
            created_at_hint="2026-03-13T18:00:02Z",
        )
    )
    selectors = result["selector_seed"]["normalized_entities"]
    assert result["selector_plan"]["task_family"] == "deployment_execution"
    assert "weight_quantized" in selectors["features"]
    assert "quant_w8a8" in selectors["features"]
    assert "weight_fp8_native" not in selectors["features"]


def test_h703_design_lookup_pack_exposes_native_fp8_unsupported_artifact(
    agent_repo_root, a3_fp8_resolve_result, a3_fp8_sqlite
) -> None:
    response = pack(
        agent_repo_root,
        request=_pack_request(
            request_id="req-h703",
            intent="design_lookup",
            selectors={
                "files": [],
                "symbols": [],
                "entities": [],
                "errors": [],
                "models": ["qwen3-32b"],
                "features": ["weight_fp8_native", "single_card", "cards_1", "topology_locked", "priority_keep_topology"],
                "hw": ["A3"],
                "commits": [],
                "prs": [],
                "versions": [],
                "configs": ["parallelism_unspecified"],
            },
            domains=["deployment_config", "ascend_foundation"],
            must_have=["native fp8 support status", "route choice"],
            nice_to_have=["conversion path", "serving constraints"],
        ),
        resolve_result=a3_fp8_resolve_result,
        merged_pack=a3_fp8_sqlite,
    )
    artifact = _atom(response, "artifact:selected|")
    assert "kind=unsupported_requires_choice" in artifact["atom_id"]
    assert "native fp8" in response["capsule_text"].lower()
    assert any("unsupported" in item.lower() or "不支持" in item for item in response["unknowns"] + response["warnings"])


def test_h704_deployment_packager_requires_artifact_and_strategy_atoms() -> None:
    card = deployment_artifact_packager(
        {
            "request_id": "req-h704",
            "plan_id": "plan-req-h704",
            "task_family": "deployment_execution",
            "work_package_id": "wp-h704",
            "selectors": {
                "files": [],
                "symbols": [],
                "entities": [],
                "errors": [],
                "models": ["qwen3-32b"],
                "features": ["weight_quantized", "quant_w8a8", "single_card", "cards_1", "topology_locked"],
                "hw": ["A3"],
                "commits": [],
                "prs": [],
                "versions": [],
                "configs": ["parallelism_unspecified"],
            },
        },
        {
            "schema_version": "kb-pack-response/v1",
            "request_id": "req-h704",
            "pack_id": "pack-req-h704",
            "created_at": "2026-03-13T18:00:04Z",
            "match_level": "exact",
            "selected_shards": ["deployment_config", "ascend_foundation"],
            "warnings": [],
            "unknowns": [],
            "budget_token_cap": 1500,
            "estimated_tokens": 280,
            "capsule_type": "atomic_capsule",
            "logical_domains": ["deployment_config", "ascend_foundation"],
            "capsule_text": "missing artifact selection",
            "atoms": [
                {
                    "atom_id": "strategy:selected|kind=inferred_preserve_topology|model=qwen3-32b|traits=quant_w8a8|hw=A3|cards=1|logical=2|tp=2|dp=1|ep=1|family=tp|preset=qwen3_32b_a3_w8a8|confidence=medium|documented=0|unvalidated=1",
                    "atom_kind": "constraint",
                    "summary": "preserving the requested smaller topology with a conservative TP-only inference",
                    "source_refs": ["validation:qwen3-32b-a3"],
                }
            ],
            "deep_reference_stubs": [],
            "cache_hit": False,
            "capsule_path": None,
        },
    )
    assert card["notes"] is None


def test_h705_single_card_a3_quantized_runbook_keeps_local_comm_env(
    agent_repo_root, a3_fp8_resolve_result, a3_fp8_sqlite
) -> None:
    response = pack(
        agent_repo_root,
        request=_pack_request(
            request_id="req-h705",
            intent="deployment_lookup",
            selectors={
                "files": [],
                "symbols": [],
                "entities": [],
                "errors": [],
                "models": ["qwen3-32b"],
                "features": [
                    "weight_quantized",
                    "quant_w8a8",
                    "single_card",
                    "cards_1",
                    "topology_locked",
                    "priority_keep_topology",
                    "artifact_modelslim",
                ],
                "hw": ["A3"],
                "commits": [],
                "prs": [],
                "versions": [],
                "configs": ["parallelism_unspecified"],
            },
            must_have=["preserve requested topology", "serving constraints"],
            nice_to_have=["launch command"],
        ),
        resolve_result=a3_fp8_resolve_result,
        merged_pack=a3_fp8_sqlite,
    )
    card = deployment_artifact_packager(
        {
            "request_id": "req-h705",
            "plan_id": "plan-req-h705",
            "task_family": "deployment_execution",
            "work_package_id": "wp-h705",
            "selectors": {
                "files": [],
                "symbols": [],
                "entities": [],
                "errors": [],
                "models": ["qwen3-32b"],
                "features": [
                    "weight_quantized",
                    "quant_w8a8",
                    "single_card",
                    "cards_1",
                    "topology_locked",
                    "priority_keep_topology",
                    "artifact_modelslim",
                ],
                "hw": ["A3"],
                "commits": [],
                "prs": [],
                "versions": [],
                "configs": ["parallelism_unspecified"],
            },
        },
        response,
    )
    assert card["notes"] is not None
    assert "ASCEND_RT_VISIBLE_DEVICES=0,1" in card["notes"]
    assert "HCCL_OP_EXPANSION_MODE" in card["notes"]


def test_h706_runtime_first_skill_lint(agent_repo_root: Path) -> None:
    findings = lint_runtime_first_skills(agent_repo_root)
    assert findings == []
