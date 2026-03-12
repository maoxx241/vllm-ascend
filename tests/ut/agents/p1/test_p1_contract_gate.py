from __future__ import annotations

import copy

import pytest

from vllm_ascend.agent_runtime.contracts import (
    ContractError,
    copy_example,
    load_schema,
    validate_contract_docs,
    validate_examples,
    validate_instance,
    validate_sql,
)


def test_a1_load_all_schemas(agent_repo_root) -> None:
    schema_dir = agent_repo_root / ".agents" / "kb" / "schema"
    for schema_path in sorted(schema_dir.glob("*.json")):
        schema = load_schema(schema_path.name, root=agent_repo_root)
        assert schema["$id"]


def test_a2_validate_all_examples(agent_repo_root) -> None:
    messages = validate_examples(root=agent_repo_root)
    assert messages
    assert any("selector-plan.deployment.intake.json" in message for message in messages)


def test_a3_init_empty_sqlite(agent_repo_root) -> None:
    messages = validate_sql(root=agent_repo_root)
    assert "OK SQL init smoke" in messages


def test_a4_schema_vs_docs_key_names(agent_repo_root) -> None:
    messages = validate_contract_docs(root=agent_repo_root)
    assert "OK governor single source of truth" in messages


def test_a5_invalid_intake_plan_with_deep_refs_rejected(agent_repo_root) -> None:
    plan = copy_example("selector-plan.deployment.intake.json", root=agent_repo_root)
    plan["max_deep_refs"] = 1
    with pytest.raises(ContractError):
        validate_instance(plan, "selector-plan.schema.json", root=agent_repo_root)


def test_a6_invalid_reroute_card_without_reroute_rejected(agent_repo_root) -> None:
    card = copy_example("atomic-result-card.reroute.json", root=agent_repo_root)
    card["reroute"] = None
    with pytest.raises(ContractError):
        validate_instance(card, "atomic-result-card.schema.json", root=agent_repo_root)


def test_a7_invalid_confirmation_seed_rejected(agent_repo_root) -> None:
    seed = copy_example("selector-seed.adaptation.pending-confirmation.json", root=agent_repo_root)
    seed["confirmation_status"] = "not_needed"
    with pytest.raises(ContractError):
        validate_instance(seed, "selector-seed.schema.json", root=agent_repo_root)


def test_a8_invalid_intake_origin_atomic_spec_plan_workflow_rejected(agent_repo_root) -> None:
    plan = copy.deepcopy(copy_example("selector-plan.performance.atomic.json", root=agent_repo_root))
    plan["execution_mode"] = "spec_plan_workflow"
    with pytest.raises(ContractError):
        validate_instance(plan, "selector-plan.schema.json", root=agent_repo_root)


def test_a9_invalid_non_full_bundle_rejected(agent_repo_root) -> None:
    state = copy_example("continuation-state.upstream-sync.json", root=agent_repo_root)
    state["persistence_mode"] = "light_bundle"
    with pytest.raises(ContractError):
        validate_instance(state, "continuation-state.schema.json", root=agent_repo_root)
