from __future__ import annotations

from vllm_ascend.agent_runtime import pack, resolve
from vllm_ascend.agent_runtime.kb import build_local


def _pack_request(request_id: str) -> dict:
    return {
        "schema_version": "kb-pack-request/v2",
        "request_id": request_id,
        "created_at": "2026-03-14T03:00:00Z",
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
            "features": ["quant_w8a8", "priority_best_perf"],
            "hw": ["A3"],
            "commits": [],
            "prs": [],
            "versions": [],
            "configs": ["parallelism_unspecified"],
        },
        "must_have": ["documented deployment baseline"],
        "nice_to_have": ["launch command"],
        "evidence_refs": [],
        "budget_token_cap": 1500,
        "max_atoms": 12,
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


def _shadow_atom(response: dict) -> dict | None:
    for atom in response["atoms"]:
        if atom["atom_id"].startswith("shadow:selected|"):
            return atom
    return None


def test_s601_shadow_gate_disabled_keeps_pack_passthrough(agent_repo_root, monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("VLLM_ASCEND_TYPED_KB_SELECTOR_RUNTIME_ADAPTER", raising=False)
    resolve_result = resolve(
        agent_repo_root,
        request_id="req-s601",
        overrides={
            "soc": "A3",
            "cann": "8.5.0",
            "torch": "2.9.0",
            "torch_npu": "2.9.0",
            "python": "3.11",
        },
    )
    emit_sqlite = tmp_path / "shadow-disabled.sqlite"
    build_local(agent_repo_root, resolve_result=resolve_result, emit_sqlite=emit_sqlite)
    response = pack(
        agent_repo_root,
        request=_pack_request("req-s601"),
        resolve_result=resolve_result,
        merged_pack=emit_sqlite,
    )
    assert _shadow_atom(response) is None
    assert "strategy:selected|" in _strategy_atom(response)["atom_id"]


def test_s602_shadow_gate_enabled_attaches_diagnostics_without_mutating_strategy(
    agent_repo_root, monkeypatch, tmp_path
) -> None:
    resolve_result = resolve(
        agent_repo_root,
        request_id="req-s602",
        overrides={
            "soc": "A3",
            "cann": "8.5.0",
            "torch": "2.9.0",
            "torch_npu": "2.9.0",
            "python": "3.11",
        },
    )
    emit_sqlite = tmp_path / "shadow-enabled.sqlite"
    build_local(agent_repo_root, resolve_result=resolve_result, emit_sqlite=emit_sqlite)

    monkeypatch.delenv("VLLM_ASCEND_TYPED_KB_SELECTOR_RUNTIME_ADAPTER", raising=False)
    baseline = pack(
        agent_repo_root,
        request=_pack_request("req-s602-baseline"),
        resolve_result=resolve_result,
        merged_pack=emit_sqlite,
    )

    monkeypatch.setenv("VLLM_ASCEND_TYPED_KB_SELECTOR_RUNTIME_ADAPTER", "1")
    response = pack(
        agent_repo_root,
        request=_pack_request("req-s602-shadow"),
        resolve_result=resolve_result,
        merged_pack=emit_sqlite,
    )

    assert _strategy_atom(response)["atom_id"] == _strategy_atom(baseline)["atom_id"]
    shadow_atom = _shadow_atom(response)
    assert shadow_atom is not None
    assert "status=shadow_" in shadow_atom["atom_id"]
    assert "主决策保持不变" in shadow_atom["summary"]
