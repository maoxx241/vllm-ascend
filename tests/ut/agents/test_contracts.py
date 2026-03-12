from vllm_ascend.agent_runtime.contracts import run_contract_checks


def test_contract_checks_pass() -> None:
    messages = run_contract_checks()
    assert any("OK SQL init smoke" in message for message in messages)
    assert any("OK governor single source of truth" in message for message in messages)
