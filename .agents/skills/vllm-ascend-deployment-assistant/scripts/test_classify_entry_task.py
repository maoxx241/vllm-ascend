#!/usr/bin/env python3
"""Regression tests for deployment entry routing."""

from __future__ import annotations

from classify_entry_task import classify_entry_task


def main() -> int:
    cases = [
        (
            "我要分析profiling，然后用kimik2模型看看瓶颈",
            "performance_analysis",
            "vllm-ascend-developer-assistant",
            False,
        ),
        (
            "帮我部署 kimi-k2 服务",
            "deployment",
            "vllm-ascend-deployment-assistant",
            True,
        ),
        (
            "先装环境，把 CANN 和依赖弄好",
            "env_bootstrap",
            "vllm-ascend-deployment-assistant",
            True,
        ),
        (
            "分析一下 crash 日志和 OOM",
            "debugging",
            "vllm-ascend-developer-assistant",
            False,
        ),
        (
            "我想做 graph mode 的设计分析",
            "design_analysis",
            "vllm-ascend-developer-assistant",
            False,
        ),
    ]

    for text, task_type, entry_skill, should_continue in cases:
        result = classify_entry_task(text)
        assert result["task_type"] == task_type, (text, result)
        assert result["entry_skill"] == entry_skill, (text, result)
        assert (
            result["should_continue_in_deployment_assistant"] == should_continue
        ), (text, result)

    print("PASS: entry routing classification")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
