from __future__ import annotations

import pytest

from vllm_ascend.agent_runtime import (
    RawRequest,
    deployment_artifact_packager,
    pack,
    perf_intake,
    resolve,
    vllm_ascend_assistant,
)
from vllm_ascend.agent_runtime.kb import build_local


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
        "created_at": "2026-03-13T15:10:00Z",
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


def _strategy_atom(response: dict) -> dict:
    for atom in response["atoms"]:
        if atom["atom_id"].startswith("strategy:selected|"):
            return atom
    raise AssertionError("expected a selected strategy atom")


@pytest.fixture()
def a3_generalize_resolve_result(agent_repo_root):
    return resolve(
        agent_repo_root,
        request_id="req-p6-generalize-a3",
        overrides={
            "soc": "A3",
            "cann": "8.5.0",
            "torch": "2.9.0",
            "torch_npu": "2.9.0",
            "python": "3.11",
        },
    )


@pytest.fixture()
def a3_generalize_sqlite(agent_repo_root, a3_generalize_resolve_result, tmp_path):
    emit_sqlite = tmp_path / "generalize-a3.sqlite"
    build_local(agent_repo_root, resolve_result=a3_generalize_resolve_result, emit_sqlite=emit_sqlite)
    return emit_sqlite


@pytest.mark.parametrize(
    ("prompt", "expected_features"),
    [
        (
            "帮我看下 A3四卡 部署 qwen3 32b w8a8 的命令",
            {"cards_4", "topology_locked", "priority_keep_topology", "quant_w8a8"},
        ),
        (
            "4 cards on A3 部署 qwen3-32b-w8a8",
            {"cards_4", "topology_locked", "priority_keep_topology", "quant_w8a8"},
        ),
        (
            "A3 4卡怎么部署 Qwen3-32B-W8A8",
            {"cards_4", "topology_locked", "priority_keep_topology", "quant_w8a8"},
        ),
    ],
)
def test_g601_prompt_variants_normalize_to_same_selector_seed(prompt: str, expected_features: set[str]) -> None:
    result = vllm_ascend_assistant(
        RawRequest(
            request_id="req-g601",
            user_text=prompt,
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=[],
            created_at_hint="2026-03-13T15:10:01Z",
        )
    )
    selectors = result["selector_seed"]["normalized_entities"]
    assert result["selector_plan"]["task_family"] == "deployment_execution"
    assert selectors["models"] == ["qwen3-32b"]
    assert selectors["hw"] == ["A3"]
    assert expected_features.issubset(set(selectors["features"]))
    assert "parallelism_unspecified" in selectors["configs"]


def test_g602_unknown_model_alias_keeps_deployment_family_with_partial_normalization() -> None:
    result = vllm_ascend_assistant(
        RawRequest(
            request_id="req-g602",
            user_text="帮我看下 foo-32b 在 A3 4卡 上怎么部署",
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=[],
            created_at_hint="2026-03-13T15:10:02Z",
        )
    )
    selectors = result["selector_seed"]["normalized_entities"]
    assert result["selector_plan"]["task_family"] == "deployment_execution"
    assert selectors["models"] == []
    assert selectors["hw"] == ["A3"]
    assert "cards_4" in selectors["features"]
    assert "topology_locked" in selectors["features"]


def test_g603_pack_selects_documented_best_perf_strategy_when_topology_unspecified(
    agent_repo_root, a3_generalize_resolve_result, a3_generalize_sqlite
) -> None:
    response = pack(
        agent_repo_root,
        request=_pack_request(
            request_id="req-g603",
            intent="deployment_lookup",
            selectors={
                "files": [],
                "symbols": [],
                "entities": [],
                "errors": [],
                "models": ["qwen3-32b"],
                "features": ["quant_w8a8", "priority_best_perf"],
                "hw": ["A3"],
                "commits": [],
                "prs": [],
                "versions": [],
                "configs": ["parallelism_unspecified"],
            },
            must_have=["documented deployment baseline"],
        ),
        resolve_result=a3_generalize_resolve_result,
        merged_pack=a3_generalize_sqlite,
    )
    selected = _strategy_atom(response)
    assert "kind=best_perf_default" in selected["atom_id"]
    assert "cards=2" in selected["atom_id"]
    assert "logical=4" in selected["atom_id"]
    assert "tp=4" in selected["atom_id"]
    assert all("unvalidated" not in item.lower() for item in response["unknowns"])


def test_g604_pack_preserves_locked_single_card_topology_as_inferred_strategy(
    agent_repo_root, a3_generalize_resolve_result, a3_generalize_sqlite
) -> None:
    response = pack(
        agent_repo_root,
        request=_pack_request(
            request_id="req-g604",
            intent="deployment_lookup",
            selectors={
                "files": [],
                "symbols": [],
                "entities": [],
                "errors": [],
                "models": ["qwen3-32b"],
                "features": ["quant_w8a8", "single_card", "cards_1", "topology_locked", "priority_keep_topology"],
                "hw": ["A3"],
                "commits": [],
                "prs": [],
                "versions": [],
                "configs": ["parallelism_unspecified"],
            },
            must_have=["preserve requested topology"],
        ),
        resolve_result=a3_generalize_resolve_result,
        merged_pack=a3_generalize_sqlite,
    )
    selected = _strategy_atom(response)
    assert "kind=inferred_preserve_topology" in selected["atom_id"]
    assert "cards=1" in selected["atom_id"]
    assert "logical=2" in selected["atom_id"]
    assert "tp=2" in selected["atom_id"]
    assert any("unvalidated" in item.lower() or "未验证" in item for item in response["unknowns"])


def test_g605_pack_marks_four_card_request_as_ambiguous_instead_of_forcing_tp(
    agent_repo_root, a3_generalize_resolve_result, a3_generalize_sqlite
) -> None:
    response = pack(
        agent_repo_root,
        request=_pack_request(
            request_id="req-g605",
            intent="deployment_lookup",
            selectors={
                "files": [],
                "symbols": [],
                "entities": [],
                "errors": [],
                "models": ["qwen3-32b"],
                "features": ["cards_4", "topology_locked", "priority_keep_topology"],
                "hw": ["A3"],
                "commits": [],
                "prs": [],
                "versions": [],
                "configs": ["parallelism_unspecified"],
            },
            must_have=["preserve requested topology"],
        ),
        resolve_result=a3_generalize_resolve_result,
        merged_pack=a3_generalize_sqlite,
    )
    selected = _strategy_atom(response)
    assert "kind=unknown_or_reroute" in selected["atom_id"]
    assert "tp=8" not in selected["atom_id"]
    assert any("tp" in item.lower() and "dp" in item.lower() for item in response["unknowns"])


def test_g606_deployment_artifact_packager_requires_strategy_atom_for_script_notes() -> None:
    card = deployment_artifact_packager(
        {
            "request_id": "req-g606",
            "plan_id": "plan-req-g606",
            "task_family": "deployment_execution",
            "work_package_id": "wp-g606",
            "selectors": {
                "files": [],
                "symbols": [],
                "entities": [],
                "errors": [],
                "models": ["qwen3-32b"],
                "features": ["single_card", "cards_1", "topology_locked"],
                "hw": ["A3"],
                "commits": [],
                "prs": [],
                "versions": [],
                "configs": ["parallelism_unspecified"],
            },
        },
        {
            "schema_version": "kb-pack-response/v1",
            "request_id": "req-g606",
            "pack_id": "pack-req-g606",
            "created_at": "2026-03-13T15:10:06Z",
            "match_level": "exact",
            "selected_shards": ["repo_semantics", "validation"],
            "warnings": [],
            "unknowns": [],
            "budget_token_cap": 1500,
            "estimated_tokens": 320,
            "capsule_type": "atomic_capsule",
            "logical_domains": ["deployment_config"],
            "capsule_text": "generic deployment capsule without strategy atom",
            "atoms": [
                {
                    "atom_id": "atom-generic",
                    "atom_kind": "fact",
                    "summary": "generic deployment fact",
                    "source_refs": ["repo_semantics:generic"],
                }
            ],
            "deep_reference_stubs": [],
            "cache_hit": False,
            "capsule_path": None,
        },
    )
    assert card["notes"] is None


def test_g607_deployment_artifact_packager_reroutes_ambiguous_strategy(
    agent_repo_root, a3_generalize_resolve_result, a3_generalize_sqlite
) -> None:
    response = pack(
        agent_repo_root,
        request=_pack_request(
            request_id="req-g607",
            intent="deployment_lookup",
            selectors={
                "files": [],
                "symbols": [],
                "entities": [],
                "errors": [],
                "models": ["qwen3-32b"],
                "features": ["cards_4", "topology_locked", "priority_keep_topology"],
                "hw": ["A3"],
                "commits": [],
                "prs": [],
                "versions": [],
                "configs": ["parallelism_unspecified"],
            },
            must_have=["preserve requested topology"],
        ),
        resolve_result=a3_generalize_resolve_result,
        merged_pack=a3_generalize_sqlite,
    )
    card = deployment_artifact_packager(
        {
            "request_id": "req-g607",
            "plan_id": "plan-req-g607",
            "task_family": "deployment_execution",
            "work_package_id": "wp-g607",
            "selectors": {
                "files": [],
                "symbols": [],
                "entities": [],
                "errors": [],
                "models": ["qwen3-32b"],
                "features": ["cards_4", "topology_locked", "priority_keep_topology"],
                "hw": ["A3"],
                "commits": [],
                "prs": [],
                "versions": [],
                "configs": ["parallelism_unspecified"],
            },
        },
        response,
    )
    assert card["result_status"] == "needs_reroute"
    assert card["reroute"] is not None
    assert card["reroute"]["target_family"] == "design_analysis"
    assert card["flush_required"] is True


def test_g608_model_expectation_reuses_topology_facts(
    agent_repo_root, a3_generalize_resolve_result, a3_generalize_sqlite
) -> None:
    intake = perf_intake(
        RawRequest(
            request_id="req-g608",
            user_text="估算 qwen3 32b 在 A3 单卡下的预期 TTFT、吞吐和显存范围",
            attachment_refs=[],
            inline_paths=[],
            inline_symbols=[],
            inline_errors=[],
            created_at_hint="2026-03-13T15:10:08Z",
        )
    )
    response = pack(
        agent_repo_root,
        request=_pack_request(
            request_id="req-g608",
            intent="model_expectation",
            selectors=intake["selector_plan"]["selectors"],
            domains=["validation_evidence", "deployment_config", "ascend_foundation"],
            must_have=["expected TTFT range", "expected throughput range", "memory headroom assumptions"],
            nice_to_have=["topology sensitivity"],
        ),
        resolve_result=a3_generalize_resolve_result,
        merged_pack=a3_generalize_sqlite,
    )
    selected = _strategy_atom(response)
    assert "cards=1" in selected["atom_id"]
    assert "logical=2" in selected["atom_id"]
    assert "topology" in response["capsule_text"].lower()
    assert any("graph mode" in item.lower() for item in response["unknowns"])


def test_g609_skill_docs_stop_being_hardware_truth_source(agent_repo_root) -> None:
    assistant_skill = (
        agent_repo_root / ".agents" / "skills" / "vllm-ascend-assistant" / "SKILL.md"
    ).read_text(encoding="utf-8")
    deployment_skill = (
        agent_repo_root / ".agents" / "skills" / "deployment_execution" / "SKILL.md"
    ).read_text(encoding="utf-8")
    synth_skill = (
        agent_repo_root / ".agents" / "skills" / "deployment-config-synthesizer" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert "Use `runtime.py` first" in assistant_skill
    assert "Do not grep raw docs first" in assistant_skill
    assert "capsule" in deployment_skill.lower()
    assert "Use the capsule as the source of truth" in synth_skill
    assert "1 card = 2 logical NPUs" not in assistant_skill
    assert "1 card = 2 logical NPUs" not in deployment_skill
    assert "1 card = 2 logical NPUs" not in synth_skill
