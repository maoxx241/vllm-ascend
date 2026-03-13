from __future__ import annotations

from vllm_ascend.agent_runtime import (
    RawRequest,
    compile_pack_request,
    cross_log_correlation,
    debug_intake,
    evaluate_governor,
    log_triage,
    pack,
    resolve,
)


def test_e5_runtime_error_triage_routes_to_debugging(agent_repo_root) -> None:
    result = debug_intake(
        RawRequest(
            request_id="req-p3b-e5",
            user_text="服务报 RuntimeError 161001，帮我做日志分诊",
            attachment_refs=[],
            inline_paths=["vllm_ascend/worker/worker.py"],
            inline_symbols=["NPUWorker"],
            inline_errors=["RuntimeError 161001", "aclnnApplyRotaryPosEmbV2 failed"],
            created_at_hint="2026-03-13T14:50:00Z",
        )
    )
    plan = result["selector_plan"]
    assert plan["task_family"] == "debugging"
    assert plan["consumer_id"] == "log-triage"
    assert plan["execution_mode"] == "direct_atomic_workflow"
    request = compile_pack_request(
        plan,
        evaluate_governor(
            selector_seed=result["selector_seed"],
            selector_plan=plan,
            continuation_state=None,
            progress_state={
                "bundle_exists": False,
                "has_unflushed_findings": False,
                "query_count_in_stage": 0,
                "opened_deep_refs_in_stage": 0,
                "seen_dedupe_keys": [],
                "last_flush_at": None,
                "session_budget_used": 0,
            },
            root=agent_repo_root,
        ),
        root=agent_repo_root,
    )
    assert request["intent"] == "debug_triage"


def test_p3b_log_triage_surfaces_exact_vs_compatible_evidence(agent_repo_root, tmp_path) -> None:
    triage = debug_intake(
        RawRequest(
            request_id="req-p3b-compatible",
            user_text="日志里出现 RuntimeError 161001，帮我排查兼容性问题",
            attachment_refs=[],
            inline_paths=["vllm_ascend/worker/worker.py"],
            inline_symbols=["NPUWorker"],
            inline_errors=["RuntimeError 161001"],
            created_at_hint="2026-03-13T14:50:01Z",
        )
    )
    compatible = resolve(
        agent_repo_root,
        request_id="req-p3b-compatible-resolve",
        overrides={
            "soc": "A3",
            "cann": "8.5.0",
            "torch": "2.9.0",
            "torch_npu": "unknown",
            "python": "3.11",
        },
    )
    emit_sqlite = tmp_path / "debug.sqlite"
    from vllm_ascend.agent_runtime.kb import build_local

    build_local(agent_repo_root, resolve_result=compatible, emit_sqlite=emit_sqlite)
    response = pack(
        agent_repo_root,
        request={
            "schema_version": "kb-pack-request/v2",
            "request_id": "req-p3b-compatible-pack",
            "created_at": "2026-03-13T14:50:01Z",
            "intent": "debug_triage",
            "repo_root": ".",
            "resolve_policy": "auto",
            "logical_domains": ["troubleshooting", "vllm_ascend_core"],
            "physical_shard_hints": ["repo_semantics", "validation"],
            "selectors": triage["selector_plan"]["selectors"],
            "must_have": ["error-signature", "workaround"],
            "nice_to_have": ["related-validation"],
            "evidence_refs": [],
            "budget_token_cap": 1500,
            "max_atoms": 10,
            "max_hops": 1,
            "include_evidence_stubs": True,
            "stop_after_first_sufficient": True,
            "emit_path": ".agents/kb/local/capsules/req-p3b-compatible-pack.json",
        },
        resolve_result=compatible,
        merged_pack=emit_sqlite,
    )
    card = log_triage(triage["selector_plan"], response)
    assert card["task_family"] == "debugging"
    assert card["confidence"] == "medium"
    assert card["residual_unknowns"]


def test_p3b_cross_log_correlation_requires_flush_before_second_source() -> None:
    result = debug_intake(
        RawRequest(
            request_id="req-p3b-cross",
            user_text="对照两份错误日志，做 cross log correlation",
            attachment_refs=["log:current", "log:baseline"],
            inline_paths=["vllm_ascend/worker/worker.py"],
            inline_symbols=["NPUWorker"],
            inline_errors=["RuntimeError 161001"],
            created_at_hint="2026-03-13T14:50:02Z",
        )
    )
    plan = result["selector_plan"]
    assert plan["consumer_id"] == "cross-log-correlation"
    blocked = evaluate_governor(
        selector_seed=result["selector_seed"],
        selector_plan=plan,
        continuation_state=None,
        progress_state={
            "bundle_exists": True,
            "has_unflushed_findings": True,
            "query_count_in_stage": 1,
            "opened_deep_refs_in_stage": 0,
            "seen_dedupe_keys": [],
            "last_flush_at": None,
            "session_budget_used": 0,
        },
    )
    assert blocked["allow_query"] is False
    assert blocked["denial_reason_code"] == "pending_flush"
    followup = cross_log_correlation(
        plan,
        {
            "capsule_text": "已对照两份日志，异常签名与 shape 条件一致。",
            "atoms": [
                {
                    "atom_id": "atom-debug-cross",
                    "atom_kind": "validation",
                    "summary": "两份日志共享同一失败签名。",
                    "source_refs": ["validation:known-failure-161001"],
                }
            ],
            "unknowns": [],
            "match_level": "compatible",
            "estimated_tokens": 360,
        },
    )
    assert followup["atomic_skill"] == "cross-log-correlation"
