---
topic_id: vllm.arg.compilation_config
canonical_term: --compilation-config
topic_kind: parameter
---

# --compilation-config

## Core

- topic_id: `vllm.arg.compilation_config`
- canonical_term: `--compilation-config`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `graph_mode`
- status/confidence: `aligned` / `0.98`
- source: `code` / source_tags: code
- semantics: 控制图编译细节（如 cudagraph_mode），决定 eager/graph 行为。
- aliases: `--compilation-config`, `compilation-config`, `compilation_config`, `compilation config`, `compilationconfig`, `graph_mode`, `graph mode`, `graph-mode`

## Foundation

- Ascend 图模式由 ACLGraph 与 Xlite 图配置共同作用，FULL_DECODE_ONLY 常用于先稳态加速 decode。
- 推荐结合 feature: `graph_mode` 查看稳定原理。

## Deployment View

- default_behavior: 默认空对象，系统按 optimization_level 自动补全 mode/cudagraph 默认值。
- value_shape: `json_object`
- accepted_values: mode, backend, custom_ops, cudagraph_mode, cudagraph_capture_sizes, max_cudagraph_capture_size, cudagraph_num_of_warmups, pass_config, use_inductor_graph_partition
- constraints: --cudagraph-capture-sizes 与 compilation_config.cudagraph_capture_sizes 互斥。; --max-cudagraph-capture-size 与 compilation_config.max_cudagraph_capture_size 互斥。; 若 cudagraph_mode 需要 piecewise，但 mode 非 VLLM_COMPILE，会被覆盖到 NONE。; enforce-eager=true 时 cudagraph_mode 会被覆盖为 NONE。
- combo_effects: 与 --enforce-eager 冲突：eager 打开会清空 cudagraph 相关设置。; 与 --optimization-level 叠加决定最终编译策略。; 与 --additional-config.xlite_graph_config/ascend_compilation_config 协同决定 ACLGraph/Xlite 行为边界。

## Development View

- definition_ref: vllm/engine/arg_utils.py:1209
- read_ref: vllm/vllm/benchmarks/lib/utils.py:18, vllm/vllm/benchmarks/lib/utils.py:31, vllm/vllm/benchmarks/lib/utils.py:61
- effect_ref: vllm/vllm/compilation/backends.py:94, vllm/vllm/compilation/backends.py:107, vllm/vllm/compilation/backends.py:148
- web_refs: 7

## Details/Edge Cases

- failure_modes: 图编译失败; 服务启动后首轮请求异常
- value_failure_signals: ValueError: cudagraph_capture_sizes ... mutually exclusive; ValueError: max_cudagraph_capture_size ... mutually exclusive
- recommendation: 先小流量验证 FULL_DECODE_ONLY，再放量。
- updated_at: 2026-03-11
