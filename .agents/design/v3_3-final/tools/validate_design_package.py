#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "schema"
EXAMPLE_DIR = ROOT / "examples"
SQL_FILE = ROOT / "sql" / "merged_pack.sql"
BACKLOG_FILE = ROOT / "tasks" / "codex_backlog.yaml"
REQUIREMENTS_FILE = ROOT / "requirements.txt"


def missing_dependency_exit(pkg: str) -> int:
    print(f"FAIL missing dependency: {pkg}")
    print("Bootstrap with:")
    print("  python -m venv .venv")
    print("  . .venv/bin/activate")
    print("  pip install -r requirements.txt")
    return 2


try:
    from jsonschema import Draft202012Validator
except Exception:
    raise SystemExit(missing_dependency_exit("jsonschema"))

try:
    import yaml  # type: ignore
except Exception:
    raise SystemExit(missing_dependency_exit("PyYAML"))


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
    raise KeyError(f"No schema mapping for example: {example_name}")


def validate_examples() -> list[str]:
    messages: list[str] = []
    for example_path in sorted(EXAMPLE_DIR.glob("*.json")):
        schema_name = schema_for_example(example_path.name)
        schema = load_json(SCHEMA_DIR / schema_name)
        validator = Draft202012Validator(schema)
        instance = load_json(example_path)
        errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.path))
        if errors:
            detail = "; ".join(f"{list(e.path)}: {e.message}" for e in errors)
            raise AssertionError(f"Example failed validation: {example_path.name}: {detail}")
        messages.append(f"OK  example -> schema  {example_path.name} -> {schema_name}")
    return messages


def validate_negative_cases() -> list[str]:
    messages: list[str] = []

    seed_schema = Draft202012Validator(load_json(SCHEMA_DIR / "selector-seed.schema.json"))
    seed = load_json(EXAMPLE_DIR / "selector-seed.adaptation.pending-confirmation.json")
    invalid_seed = copy.deepcopy(seed)
    invalid_seed["confirmation_status"] = "not_needed"
    assert list(seed_schema.iter_errors(invalid_seed)), "invalid confirmation seed unexpectedly passed"
    messages.append("OK  negative seed: required=true + status=not_needed rejected")

    plan_schema = Draft202012Validator(load_json(SCHEMA_DIR / "selector-plan.schema.json"))
    plan = load_json(EXAMPLE_DIR / "selector-plan.deployment.intake.json")
    invalid_plan = copy.deepcopy(plan)
    invalid_plan["max_deep_refs"] = 1
    assert list(plan_schema.iter_errors(invalid_plan)), "invalid intake plan unexpectedly passed"
    messages.append("OK  negative plan: intake plan with deep refs rejected")

    perf_plan = load_json(EXAMPLE_DIR / "selector-plan.performance.atomic.json")
    invalid_plan2 = copy.deepcopy(perf_plan)
    invalid_plan2["execution_mode"] = "spec_plan_workflow"
    assert list(plan_schema.iter_errors(invalid_plan2)), "invalid intake-origin atomic spec plan unexpectedly passed"
    messages.append("OK  negative plan: intake-origin atomic plan with spec_plan_workflow rejected")

    invalid_plan3 = copy.deepcopy(plan)
    invalid_plan3["budget_class"] = "atomic"
    invalid_plan3["capsule_type"] = "atomic_capsule"
    invalid_plan3["query_stage"] = "intake"
    assert list(plan_schema.iter_errors(invalid_plan3)), "invalid intake/atomic combo unexpectedly passed"
    messages.append("OK  negative plan: intake query_stage with atomic budget rejected")

    card_schema = Draft202012Validator(load_json(SCHEMA_DIR / "atomic-result-card.schema.json"))
    card = load_json(EXAMPLE_DIR / "atomic-result-card.reroute.json")
    invalid_card = copy.deepcopy(card)
    invalid_card["reroute"] = None
    assert list(card_schema.iter_errors(invalid_card)), "invalid reroute card unexpectedly passed"
    messages.append("OK  negative card: reroute without payload rejected")

    invalid_card2 = copy.deepcopy(card)
    invalid_card2["next_action"]["kind"] = "continue_atomic"
    assert list(card_schema.iter_errors(invalid_card2)), "invalid reroute action unexpectedly passed"
    messages.append("OK  negative card: reroute without reroute_task rejected")

    cont_schema = Draft202012Validator(load_json(SCHEMA_DIR / "continuation-state.schema.json"))
    cont = load_json(EXAMPLE_DIR / "continuation-state.upstream-sync.json")
    invalid_cont = copy.deepcopy(cont)
    invalid_cont["persistence_mode"] = "none"
    assert list(cont_schema.iter_errors(invalid_cont)), "invalid continuation persistence unexpectedly passed"
    messages.append("OK  negative continuation: non-full_bundle persistence rejected")

    return messages


def validate_sql() -> list[str]:
    messages: list[str] = []
    sql_text = SQL_FILE.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.sqlite"
        conn = sqlite3.connect(db_path)
        try:
            conn.executescript(sql_text)
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = {row[0] for row in cursor.fetchall()}
            required = {"pack_meta", "sources", "entities", "facts", "edges", "symbol_index", "validations", "capsules"}
            missing = required - tables
            if missing:
                raise AssertionError(f"Missing tables after SQL init: {sorted(missing)}")
            messages.append("OK  SQL init smoke test")
        finally:
            conn.close()
    return messages


def validate_backlog() -> list[str]:
    messages: list[str] = []
    data = yaml.safe_load(BACKLOG_FILE.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "Backlog YAML did not parse to a mapping"
    assert "phases" in data and isinstance(data["phases"], list), "Backlog YAML missing phases list"
    messages.append("OK  codex_backlog.yaml parse")
    return messages


def validate_bootstrap_files() -> list[str]:
    messages: list[str] = []
    assert REQUIREMENTS_FILE.exists(), "requirements.txt missing"
    reqs = REQUIREMENTS_FILE.read_text(encoding="utf-8")
    assert "jsonschema" in reqs and "PyYAML" in reqs, "requirements.txt missing validator dependencies"
    messages.append("OK  bootstrap requirements present")
    return messages



def validate_contract_docs() -> list[str]:
    messages: list[str] = []
    interface_text = (ROOT / "docs" / "06-interface-contracts.md").read_text(encoding="utf-8")
    governor_text = (ROOT / "docs" / "04-context-governor-and-persistence.md").read_text(encoding="utf-8")
    acceptance_text = (ROOT / "tasks" / "acceptance_matrix.md").read_text(encoding="utf-8")
    selector_plan_schema = load_json(SCHEMA_DIR / "selector-plan.schema.json")
    governor_schema = load_json(SCHEMA_DIR / "governor-decision.schema.json")

    assert 'stage: Literal["public_entry", "intake", "spec_plan", "atomic"]' not in interface_text, "interface contract still exposes explicit stage input"
    assert 'selector_plan.query_stage' in interface_text, "interface contract missing query_stage-derived stage rule"
    assert 'governor-decision.stage' in interface_text, "interface contract missing derived output-stage rule"
    assert '不接受任何外部 `stage` override' in governor_text, "governor spec missing single-stage-source rule"
    assert 'derived only from `selector_plan.query_stage`' in acceptance_text, "acceptance matrix missing single-stage-source case"
    assert 'only formal stage input consumed by context-governor' in selector_plan_schema['properties']['query_stage'].get('description', ''), "selector-plan schema missing query_stage single-source description"
    assert 'Derived effective stage' in governor_schema['properties']['stage'].get('description', ''), "governor-decision schema missing derived-stage description"
    messages.append("OK  contract docs/schema lint: governor stage has a single source of truth")
    return messages

def main() -> int:
    all_messages: list[str] = []
    try:
        all_messages.extend(validate_bootstrap_files())
        all_messages.extend(validate_examples())
        all_messages.extend(validate_negative_cases())
        all_messages.extend(validate_contract_docs())
        all_messages.extend(validate_sql())
        all_messages.extend(validate_backlog())
    except Exception as exc:
        print(f"FAIL {exc}")
        return 1
    for line in all_messages:
        print(line)
    print(f"PASS validated {len(list(EXAMPLE_DIR.glob('*.json')))} examples + critical negative cases + contract lint + SQL smoke + backlog parse")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
