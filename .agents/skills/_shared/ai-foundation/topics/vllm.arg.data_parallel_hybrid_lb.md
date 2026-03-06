---
topic_id: vllm.arg.data_parallel_hybrid_lb
canonical_term: --data-parallel-hybrid-lb
topic_kind: parameter
---

# --data-parallel-hybrid-lb

## Core

- topic_id: `vllm.arg.data_parallel_hybrid_lb`
- canonical_term: `--data-parallel-hybrid-lb`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `data_parallel`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code
- semantics: 通过副本扩展吞吐能力，并依赖 DP 地址和 RPC 协调。
- aliases: `--data-parallel-hybrid-lb`, `data-parallel-hybrid-lb`, `data_parallel_hybrid_lb`, `data parallel hybrid lb`, `dataparallelhybridlb`, `data_parallel`, `data parallel`, `data-parallel`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `data_parallel` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 错误地址/端口会导致调度与健康检查失败
- combo_effects: N/A

## Development View

- definition_ref: vllm/engine/arg_utils.py:860
- read_ref: vllm/vllm/config/parallel.py:126, vllm/vllm/config/parallel.py:380, vllm/vllm/config/parallel.py:519
- effect_ref: vllm/vllm/config/parallel.py:380, vllm/vllm/engine/arg_utils.py:1546, vllm/vllm/entrypoints/cli/serve.py:147
- web_refs: 4

## Details/Edge Cases

- failure_modes: RPC 连接失败; 请求分发不均衡
- value_failure_signals: RPC 连接失败; 请求分发不均衡
- recommendation: 固定 DP 地址和端口后再迭代性能参数。
- updated_at: 2026-03-06
