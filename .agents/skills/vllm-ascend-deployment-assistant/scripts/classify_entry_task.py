#!/usr/bin/env python3
"""Cheap first-hop task classification for entry routing."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskRule:
    task_type: str
    entry_skill: str
    keywords: tuple[str, ...]
    reason: str


RULES: tuple[TaskRule, ...] = (
    TaskRule(
        "performance_analysis",
        "vllm-ascend-developer-assistant",
        (
            "profiling",
            "profile",
            "perf",
            "性能",
            "吞吐",
            "时延",
            "延迟",
            "ttft",
            "tpot",
            "回归归因",
            "瓶颈",
            "benchmark",
            "trace",
        ),
        "profiling/performance signals require the developer entry chain",
    ),
    TaskRule(
        "debugging",
        "vllm-ascend-developer-assistant",
        (
            "报错",
            "错误",
            "崩溃",
            "crash",
            "oom",
            "hang",
            "卡住",
            "日志",
            "异常",
            "回放失败",
            "replay mismatch",
        ),
        "debug/crash signals require the developer entry chain",
    ),
    TaskRule(
        "model_adaptation",
        "vllm-ascend-developer-assistant",
        (
            "模型适配",
            "适配模型",
            "接入新模型",
            "register model",
            "model adaptation",
            "新模型",
            "接模型",
        ),
        "model adaptation requests require the developer entry chain",
    ),
    TaskRule(
        "design_analysis",
        "vllm-ascend-developer-assistant",
        (
            "设计分析",
            "架构分析",
            "原理分析",
            "实现分析",
            "设计",
            "架构",
            "源码分析",
            "code path",
        ),
        "design/architecture requests require the developer entry chain",
    ),
    TaskRule(
        "upstream_sync",
        "vllm-ascend-developer-assistant",
        (
            "上游同步",
            "sync",
            "main2main",
            "rebase",
            "upstream",
        ),
        "upstream sync requests require the developer entry chain",
    ),
    TaskRule(
        "release_analysis",
        "vllm-ascend-developer-assistant",
        (
            "release note",
            "release notes",
            "发布说明",
            "发版",
            "changelog",
            "release",
        ),
        "release analysis requests require the developer entry chain",
    ),
    TaskRule(
        "op_development",
        "vllm-ascend-developer-assistant",
        (
            "算子",
            "operator",
            "kernel",
            "triton",
            "自定义算子",
        ),
        "operator/kernel requests require the developer entry chain",
    ),
    TaskRule(
        "env_bootstrap",
        "vllm-ascend-deployment-assistant",
        (
            "环境",
            "安装",
            "bootstrap",
            "setup env",
            "初始化",
            "依赖",
            "cann",
            "驱动",
        ),
        "environment setup belongs to the deployment entry chain",
    ),
    TaskRule(
        "deployment",
        "vllm-ascend-deployment-assistant",
        (
            "部署",
            "上线",
            "serve",
            "启动服务",
            "launch",
            "拉起服务",
            "起服务",
        ),
        "deployment requests belong to the deployment entry chain",
    ),
)


def classify_entry_task(text: str) -> dict[str, object]:
    raw = text.strip()
    lowered = raw.lower()
    for rule in RULES:
        matched = [kw for kw in rule.keywords if kw.lower() in lowered]
        if matched:
            return {
                "task_type": rule.task_type,
                "entry_skill": rule.entry_skill,
                "should_continue_in_deployment_assistant": rule.entry_skill
                == "vllm-ascend-deployment-assistant",
                "matched_signals": matched,
                "reason": rule.reason,
            }

    return {
        "task_type": "deployment",
        "entry_skill": "vllm-ascend-deployment-assistant",
        "should_continue_in_deployment_assistant": True,
        "matched_signals": [],
        "reason": "no specialist engineering signal detected; keep deployment entry as default",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="Natural language user request")
    args = parser.parse_args()
    print(json.dumps(classify_entry_task(args.text), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
