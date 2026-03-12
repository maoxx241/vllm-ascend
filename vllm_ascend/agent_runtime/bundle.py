from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import copy_example, dump_json, now_utc, validate_instance
from .paths import repo_root, tasks_root


def ensure_task_bundle(task_id: str, persistence_mode: str, root: Path | None = None) -> str:
    if persistence_mode != "full_bundle":
        raise ValueError("continuation_state only supports full_bundle")
    root = root or repo_root()
    bundle_root = tasks_root(root) / task_id
    bundle_root.mkdir(parents=True, exist_ok=True)
    for filename, title in {
        "spec.md": "# Spec\n",
        "plan.md": "# Plan\n",
        "checklist.md": "# Checklist\n",
        "progress.md": "# Progress\n",
    }.items():
        path = bundle_root / filename
        if not path.exists():
            path.write_text(title, encoding="utf-8")
    return str(bundle_root.relative_to(root))


def append_progress_entry(task_id: str, entry_markdown: str, root: Path | None = None) -> None:
    root = root or repo_root()
    bundle_root = Path(ensure_task_bundle(task_id, "full_bundle", root=root))
    progress_file = root / bundle_root / "progress.md"
    with progress_file.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {now_utc()}\n{entry_markdown.strip()}\n")


def update_plan_section(task_id: str, patch: dict[str, Any], root: Path | None = None) -> None:
    root = root or repo_root()
    bundle_root = Path(ensure_task_bundle(task_id, "full_bundle", root=root))
    plan_file = root / bundle_root / "plan.md"
    payload = json.dumps(patch, ensure_ascii=False, indent=2)
    plan_file.write_text("# Plan\n\n```json\n" + payload + "\n```\n", encoding="utf-8")


def save_selector_plan(task_id: str, selector_plan: dict[str, Any], root: Path | None = None) -> str:
    root = root or repo_root()
    bundle_root = Path(ensure_task_bundle(task_id, "full_bundle", root=root))
    output = root / bundle_root / "runtime" / "plans" / f"{selector_plan['plan_id']}.json"
    validate_instance(selector_plan, "selector-plan.schema.json", root=root)
    dump_json(output, selector_plan)
    return str(output.relative_to(root))


def save_atomic_card(task_id: str, card: dict[str, Any], root: Path | None = None) -> str:
    root = root or repo_root()
    bundle_root = Path(ensure_task_bundle(task_id, "full_bundle", root=root))
    output = root / bundle_root / "runtime" / "atomic-cards" / f"{card['card_id']}.json"
    validate_instance(card, "atomic-result-card.schema.json", root=root)
    dump_json(output, card)
    return str(output.relative_to(root))


def save_continuation_state(task_id: str, state: dict[str, Any], root: Path | None = None) -> str:
    root = root or repo_root()
    bundle_root = Path(ensure_task_bundle(task_id, "full_bundle", root=root))
    output = root / bundle_root / "runtime" / "continuation-state.json"
    validate_instance(state, "continuation-state.schema.json", root=root)
    dump_json(output, state)
    return str(output.relative_to(root))


def flush_atomic_result(selector_plan: dict[str, Any], card: dict[str, Any], root: Path | None = None) -> dict[str, str]:
    root = root or repo_root()
    task_id = card["task_id"]
    plan_ref = save_selector_plan(task_id, selector_plan, root=root)
    card_ref = save_atomic_card(task_id, card, root=root)
    append_progress_entry(
        task_id,
        "\n".join(
            [
                f"- atomic_skill: {card['atomic_skill']}",
                f"- result_status: {card['result_status']}",
                f"- resolution_code: {card['resolution_code']}",
                f"- summary: {card['finding_summary']}",
                f"- plan_ref: {plan_ref}",
                f"- card_ref: {card_ref}",
            ]
        ),
        root=root,
    )
    return {"plan_ref": plan_ref, "card_ref": card_ref}


def refresh_continuation_state(task_id: str, state: dict[str, Any], root: Path | None = None) -> str:
    root = root or repo_root()
    bundle_root = Path(ensure_task_bundle(task_id, "full_bundle", root=root))
    progress_file = root / bundle_root / "progress.md"
    progress_text = progress_file.read_text(encoding="utf-8")
    if progress_text.strip() == "# Progress":
        raise ValueError("bundle must be flushed before continuation refresh")
    return save_continuation_state(task_id, state, root=root)


def build_continuation_state(
    *,
    task_id: str,
    request_id: str,
    selector_plan: dict[str, Any],
    goal: str,
    selected_atomics: list[str] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    root = root or repo_root()
    bundle_root = ensure_task_bundle(task_id, "full_bundle", root=root)
    save_selector_plan(task_id, selector_plan, root=root)
    state = copy_example("continuation-state.upstream-sync.json", root=root)
    state.update(
        {
            "task_id": task_id,
            "request_id": request_id,
            "updated_at": now_utc(),
            "goal": goal,
            "task_family": selector_plan["task_family"],
            "execution_mode": selector_plan["execution_mode"],
            "analysis_depth": selector_plan["analysis_depth"],
            "deliverable_contract": selector_plan["deliverable_contract"],
            "current_stage": selector_plan["query_stage"],
            "bundle_root": bundle_root,
            "canonical_source_files": [
                f"{bundle_root}/spec.md",
                f"{bundle_root}/plan.md",
                f"{bundle_root}/checklist.md",
                f"{bundle_root}/progress.md",
            ],
            "selected_atomics": selected_atomics or [selector_plan["consumer_id"]],
            "execution_order": [selector_plan["work_package_id"]],
            "completed_work_packages": [],
            "pending_work_packages": [selector_plan["work_package_id"]],
            "success_criteria": selector_plan["must_have"],
            "stop_conditions": selector_plan["query_trigger_codes"],
            "reroute_conditions": ["若 atomic 判定越过 family 边界，则返回 needs_reroute"],
            "unresolved_risks": selector_plan["nice_to_have"] or ["暂无"],
            "open_questions": selector_plan["notes"] and [selector_plan["notes"]] or [],
            "latest_selector_plan_refs": [f"{bundle_root}/runtime/plans/{selector_plan['plan_id']}.json"],
            "latest_atomic_card_refs": [],
            "compacted_history_refs": [],
            "session_budget_used": 0,
            "last_flush_at": now_utc(),
            "notes": "文件 bundle 是唯一 source of truth；continuation_state 只用于续跑。",
        }
    )
    validate_instance(state, "continuation-state.schema.json", root=root)
    return state
