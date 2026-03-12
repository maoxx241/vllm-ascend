#!/usr/bin/env python3
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_ascend.agent_runtime.contracts import kb_path, run_contract_checks


def main() -> int:
    try:
        messages = run_contract_checks()
    except Exception as exc:
        print(f"FAIL {exc}")
        return 1
    for line in messages:
        print(line)
    example_count = len(list(kb_path("examples").glob("*.json")))
    print(f"PASS validated {example_count} examples + critical negative cases + contract lint + SQL smoke + backlog parse")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
