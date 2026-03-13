from __future__ import annotations

import sqlite3

from vllm_ascend.agent_runtime.kb import build_local, pack, resolve


def _request(intent: str, request_id: str) -> dict:
    return {
        "schema_version": "kb-pack-request/v2",
        "request_id": request_id,
        "created_at": "2026-03-13T14:10:00Z",
        "intent": intent,
        "repo_root": ".",
        "resolve_policy": "auto",
        "logical_domains": ["vllm_upstream", "vllm_ascend_core", "integration_core"],
        "physical_shard_hints": ["vllm_semantics", "vllm_symbols", "vllm_release_delta", "repo_semantics"],
        "selectors": {
            "files": [],
            "symbols": [],
            "entities": [],
            "errors": [],
            "models": ["qwen3-next-32b"],
            "features": ["prefill", "decode", "tp4", "bf16"],
            "hw": ["A2"],
            "commits": [],
            "prs": [],
            "versions": ["vllm-ascend@feature/agent-runtime-v33-p3-mvp"],
            "configs": ["tp4_bf16_ctx8k"],
        },
        "must_have": ["upstream semantics", "affected symbols"],
        "nice_to_have": ["release delta summary"],
        "evidence_refs": [],
        "budget_token_cap": 2400,
        "max_atoms": 10,
        "max_hops": 2,
        "include_evidence_stubs": True,
        "stop_after_first_sufficient": False,
        "emit_path": f".agents/kb/local/capsules/{request_id}.json",
    }


def test_k401_resolve_can_select_vllm_pair_shards(agent_repo_root) -> None:
    result = resolve(
        agent_repo_root,
        request_id="req-p4-resolve",
        overrides={
            "soc": "A2",
            "cann": "8.5.0",
            "torch": "2.9.0",
            "torch_npu": "2.9.0",
            "python": "3.11",
        },
    )
    assert "vllm_semantics" in result["selected_shards"]
    assert "vllm_symbols" in result["selected_shards"]
    assert "vllm_release_delta" in result["selected_shards"]


def test_k402_build_local_contains_upstream_sources_and_symbols(agent_repo_root, exact_resolve_result, tmp_path) -> None:
    emit_sqlite = tmp_path / "pair.sqlite"
    build_local(agent_repo_root, resolve_result=exact_resolve_result, emit_sqlite=emit_sqlite)
    conn = sqlite3.connect(emit_sqlite)
    try:
        source_ids = {row[0] for row in conn.execute("SELECT source_id FROM sources")}
        assert "source-vllm-engine-arg-utils" in source_ids
        assert "source-vllm-release-md" in source_ids
        qualnames = {row[0] for row in conn.execute("SELECT qualname FROM symbol_index")}
        assert "EngineArgs" in qualnames
        assert "EngineArgs.create_engine_config" in qualnames
    finally:
        conn.close()


def test_k403_release_delta_rows_record_version_scopes(agent_repo_root, exact_resolve_result, tmp_path) -> None:
    emit_sqlite = tmp_path / "pair.sqlite"
    build_local(agent_repo_root, resolve_result=exact_resolve_result, emit_sqlite=emit_sqlite)
    conn = sqlite3.connect(emit_sqlite)
    try:
        rows = list(
            conn.execute(
                "SELECT metadata_json FROM facts WHERE shard_family = 'vllm_release_delta' ORDER BY fact_id"
            )
        )
        assert rows
        assert '"from_version"' in rows[0][0]
        assert '"to_version"' in rows[0][0]
        assert '"impact_tags"' in rows[0][0]
    finally:
        conn.close()


def test_p4_design_lookup_uses_pair_facts(exact_resolve_result, agent_repo_root, tmp_path) -> None:
    emit_sqlite = tmp_path / "pair.sqlite"
    build_local(agent_repo_root, resolve_result=exact_resolve_result, emit_sqlite=emit_sqlite)
    request = _request("design_lookup", "req-p4-design")
    response = pack(agent_repo_root, request=request, resolve_result=exact_resolve_result, merged_pack=emit_sqlite)
    assert not any("requires deferred family support" in item for item in response["unknowns"])
    assert response["atoms"]
    assert any("vllm_symbols" in ref or "vllm_semantics" in ref for atom in response["atoms"] for ref in atom["source_refs"])


def test_p4_upstream_delta_uses_release_delta_pack(exact_resolve_result, agent_repo_root, tmp_path) -> None:
    emit_sqlite = tmp_path / "pair.sqlite"
    build_local(agent_repo_root, resolve_result=exact_resolve_result, emit_sqlite=emit_sqlite)
    request = _request("upstream_delta", "req-p4-delta")
    request["logical_domains"] = ["vllm_upstream", "integration_core"]
    request["must_have"] = ["delta summary", "affected surfaces"]
    response = pack(agent_repo_root, request=request, resolve_result=exact_resolve_result, merged_pack=emit_sqlite)
    assert response["atoms"]
    assert any("release delta" in atom["summary"].lower() or "upstream" in atom["summary"].lower() for atom in response["atoms"])
