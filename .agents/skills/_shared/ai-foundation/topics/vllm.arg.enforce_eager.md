---
topic_id: vllm.arg.enforce_eager
canonical_term: --enforce-eager
topic_kind: parameter
---

# --enforce-eager

## Core

- topic_id: `vllm.arg.enforce_eager`
- canonical_term: `--enforce-eager`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `graph_mode`
- status/confidence: `needs_manual_review` / `0.86`
- semantics: 控制 eager/graph 执行策略，通常优化吞吐与时延抖动。
- aliases: `--enforce-eager`, `enforce-eager`, `enforce_eager`, `enforce eager`, `enforceeager`, `graph_mode`, `graph mode`, `graph-mode`

## Foundation

- 图模式通过稳定执行图降低调度抖动，提升吞吐稳定性。
- 推荐结合 feature: `graph_mode` 查看稳定原理。

## Deployment View

- default_behavior: 默认 disabled（False）。
- value_shape: `binary_toggle`
- accepted_values: enabled, disabled
- constraints: 开启后 compilation_config.cudagraph_mode 会被覆盖为 NONE。; bitsandbytes 8bit 或部分 ROCm 场景可能被自动切到 eager。
- combo_effects: 与 --compilation-config / --cudagraph-* 强耦合，开启 eager 后相关图参数失效。

## Development View

- definition_ref: vllm/engine/arg_utils.py:675
- read_ref: vllm/vllm/config/model.py:194, vllm/vllm/config/model.py:336, vllm/vllm/config/model.py:960
- effect_ref: vllm/vllm/config/model.py:960, vllm/vllm/config/vllm.py:844, vllm/vllm/entrypoints/llm.py:166
- web_refs: 7

## Details/Edge Cases

- failure_modes: 图编译失败; 服务启动后首轮请求异常
- value_failure_signals: warning: Enforce eager set, overriding optimization level to -O0; info: Cudagraph is disabled under eager mode
- recommendation: 先小流量验证 FULL_DECODE_ONLY，再放量。
- updated_at: 2026-03-05
