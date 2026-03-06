---
topic_id: vllm.arg.kernel_config
canonical_term: --kernel-config
topic_kind: parameter
---

# --kernel-config

## Core

- topic_id: `vllm.arg.kernel_config`
- canonical_term: `--kernel-config`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `general_runtime`
- status/confidence: `needs_manual_review` / `0.76`
- source: `code` / source_tags: code
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `--kernel-config`, `kernel-config`, `kernel_config`, `kernel config`, `kernelconfig`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 错误组合可能影响稳定性
- combo_effects: N/A

## Development View

- definition_ref: vllm/engine/arg_utils.py:1210
- read_ref: vllm/vllm/config/vllm.py:133, vllm/vllm/config/vllm.py:152, vllm/vllm/config/vllm.py:171
- effect_ref: vllm/vllm/config/vllm.py:774, vllm/vllm/engine/arg_utils.py:603, vllm/vllm/engine/arg_utils.py:1742
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-06
