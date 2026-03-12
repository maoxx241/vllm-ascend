from __future__ import annotations

import copy
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
import yaml

from .paths import kb_root, repo_root


class ContractError(RuntimeError):
    pass


def now_utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def kb_path(*parts: str, root: Path | None = None) -> Path:
    return kb_root(root or repo_root()).joinpath(*parts)


def schema_for_example(example_name: str) -> str:
    prefixes = {
        "selector-seed.": "selector-seed.schema.json",
        "selector-plan.": "selector-plan.schema.json",
        "atomic-result-card.": "atomic-result-card.schema.json",
        "continuation-state.": "continuation-state.schema.json",
        "governor-decision.": "governor-decision.schema.json",
        "kb-resolve-result.": "kb-resolve-result.schema.json",
        "kb-pack-request.": "kb-pack-request.schema.json",
        "kb-pack-response.": "kb-pack-response.schema.json",
    }
    for prefix, schema_name in prefixes.items():
        if example_name.startswith(prefix):
            return schema_name
    raise KeyError(f"no schema mapping for example: {example_name}")


def load_schema(schema_name: str, root: Path | None = None) -> dict[str, Any]:
    return load_json(kb_path("schema", schema_name, root=root))


def load_example(example_name: str, root: Path | None = None) -> dict[str, Any]:
    return load_json(kb_path("examples", example_name, root=root))


def copy_example(example_name: str, root: Path | None = None) -> dict[str, Any]:
    return copy.deepcopy(load_example(example_name, root=root))


def validator_for(schema_name: str, root: Path | None = None) -> Draft202012Validator:
    return Draft202012Validator(load_schema(schema_name, root=root))


def validate_instance(instance: dict[str, Any], schema_name: str | None = None, root: Path | None = None) -> None:
    schema_name = schema_name or schema_for_example(instance["schema_version"].replace("/", ".") + ".json")
    errors = sorted(validator_for(schema_name, root=root).iter_errors(instance), key=lambda err: list(err.path))
    if errors:
        detail = "; ".join(f"{list(err.path)}: {err.message}" for err in errors)
        raise ContractError(detail)


def validate_examples(root: Path | None = None) -> list[str]:
    root = root or repo_root()
    messages: list[str] = []
    for example_path in sorted(kb_path("examples", root=root).glob("*.json")):
        schema_name = schema_for_example(example_path.name)
        validate_instance(load_json(example_path), schema_name=schema_name, root=root)
        messages.append(f"OK example {example_path.name}")
    return messages


def validate_negative_cases(root: Path | None = None) -> list[str]:
    root = root or repo_root()
    messages: list[str] = []

    seed_schema = validator_for("selector-seed.schema.json", root=root)
    seed = load_example("selector-seed.adaptation.pending-confirmation.json", root=root)
    invalid_seed = copy.deepcopy(seed)
    invalid_seed["confirmation_status"] = "not_needed"
    assert list(seed_schema.iter_errors(invalid_seed))
    messages.append("OK negative seed")

    plan_schema = validator_for("selector-plan.schema.json", root=root)
    plan = load_example("selector-plan.deployment.intake.json", root=root)
    invalid_plan = copy.deepcopy(plan)
    invalid_plan["max_deep_refs"] = 1
    assert list(plan_schema.iter_errors(invalid_plan))
    messages.append("OK negative plan deep refs")

    perf_plan = load_example("selector-plan.performance.atomic.json", root=root)
    invalid_plan2 = copy.deepcopy(perf_plan)
    invalid_plan2["execution_mode"] = "spec_plan_workflow"
    assert list(plan_schema.iter_errors(invalid_plan2))
    messages.append("OK negative plan execution mode")

    invalid_plan3 = copy.deepcopy(plan)
    invalid_plan3["budget_class"] = "atomic"
    invalid_plan3["capsule_type"] = "atomic_capsule"
    invalid_plan3["query_stage"] = "intake"
    assert list(plan_schema.iter_errors(invalid_plan3))
    messages.append("OK negative plan intake/atomic combo")

    card_schema = validator_for("atomic-result-card.schema.json", root=root)
    card = load_example("atomic-result-card.reroute.json", root=root)
    invalid_card = copy.deepcopy(card)
    invalid_card["reroute"] = None
    assert list(card_schema.iter_errors(invalid_card))
    messages.append("OK negative reroute payload")

    invalid_card2 = copy.deepcopy(card)
    invalid_card2["next_action"]["kind"] = "continue_atomic"
    assert list(card_schema.iter_errors(invalid_card2))
    messages.append("OK negative reroute action")

    cont_schema = validator_for("continuation-state.schema.json", root=root)
    cont = load_example("continuation-state.upstream-sync.json", root=root)
    invalid_cont = copy.deepcopy(cont)
    invalid_cont["persistence_mode"] = "none"
    assert list(cont_schema.iter_errors(invalid_cont))
    messages.append("OK negative continuation persistence")

    invalid_code_change = copy.deepcopy(load_example("selector-plan.performance.atomic.json", root=root))
    invalid_code_change["task_family"] = "adaptation"
    invalid_code_change["deliverable_contract"] = "code_change_pack"
    invalid_code_change["analysis_depth"] = "none"
    assert list(plan_schema.iter_errors(invalid_code_change))
    messages.append("OK negative code-change analysis depth")
    return messages


def validate_sql(root: Path | None = None) -> list[str]:
    root = root or repo_root()
    messages: list[str] = []
    sql_text = kb_path("sql", "merged_pack.sql", root=root).read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.sqlite"
        conn = sqlite3.connect(db_path)
        try:
            conn.executescript(sql_text)
            tables = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            }
            required = {
                "pack_meta",
                "sources",
                "entities",
                "facts",
                "edges",
                "symbol_index",
                "validations",
                "capsules",
            }
            missing = required - tables
            if missing:
                raise ContractError(f"missing tables after init: {sorted(missing)}")
            messages.append("OK SQL init smoke")
        finally:
            conn.close()
    return messages


def validate_backlog(root: Path | None = None) -> list[str]:
    root = root or repo_root()
    data = yaml.safe_load(kb_path("tasks", "codex_backlog.yaml", root=root).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("phases"), list):
        raise ContractError("backlog YAML missing phases list")
    return ["OK backlog parse"]


def validate_contract_docs(root: Path | None = None) -> list[str]:
    root = root or repo_root()
    design_docs = root / ".agents" / "design" / "v3_3-final"
    interface_text = (design_docs / "docs" / "06-interface-contracts.md").read_text(encoding="utf-8")
    governor_text = (design_docs / "docs" / "04-context-governor-and-persistence.md").read_text(encoding="utf-8")
    acceptance_text = kb_path("tasks", "acceptance_matrix.md", root=root).read_text(encoding="utf-8")
    selector_plan_schema = load_schema("selector-plan.schema.json", root=root)
    governor_schema = load_schema("governor-decision.schema.json", root=root)

    assert 'stage: Literal["public_entry", "intake", "spec_plan", "atomic"]' not in interface_text
    assert "selector_plan.query_stage" in interface_text
    assert "governor-decision.stage" in interface_text
    assert "不接受任何外部 `stage` override" in governor_text
    assert "derived only from `selector_plan.query_stage`" in acceptance_text
    assert "only formal stage input" in selector_plan_schema["properties"]["query_stage"]["description"]
    assert "Derived effective stage" in governor_schema["properties"]["stage"]["description"]
    return ["OK governor single source of truth"]


def run_contract_checks(root: Path | None = None) -> list[str]:
    root = root or repo_root()
    messages: list[str] = []
    messages.extend(validate_examples(root=root))
    messages.extend(validate_negative_cases(root=root))
    messages.extend(validate_contract_docs(root=root))
    messages.extend(validate_sql(root=root))
    messages.extend(validate_backlog(root=root))
    return messages
