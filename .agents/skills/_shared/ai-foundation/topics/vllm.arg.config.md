---
topic_id: vllm.arg.config
canonical_term: --config
topic_kind: parameter
---

# --config

## Core

- topic_id: `vllm.arg.config`
- canonical_term: `--config`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `general_runtime`
- status/confidence: `upstream_delta` / `0.68`
- source: `code` / source_tags: code
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `--config`, `config`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: Read CLI options from a config file. Must be a YAML with the following options: https://docs.vllm.ai/en/latest/configuration/serve_args.html
- value_shape: `free_form`
- accepted_values: string value
- constraints: 错误组合可能影响稳定性; Must be a YAML with the following options: https://docs.vllm.ai/en/latest/configuration/serve_args.html
- combo_effects: N/A

## Development View

- definition_ref: vllm/entrypoints/openai/cli_args.py:301
- read_ref: vllm/vllm/_aiter_ops.py:1555, vllm/vllm/_aiter_ops.py:1572, vllm/vllm/_aiter_ops.py:1572
- effect_ref: vllm/vllm/benchmarks/datasets.py:894, vllm/vllm/benchmarks/datasets.py:896, vllm/vllm/benchmarks/datasets.py:899
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: 启动失败; 行为与预期不符
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-06
