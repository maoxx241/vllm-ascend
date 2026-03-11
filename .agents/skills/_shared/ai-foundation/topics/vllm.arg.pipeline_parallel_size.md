---
topic_id: vllm.arg.pipeline_parallel_size
canonical_term: --pipeline-parallel-size
topic_kind: parameter
---

# --pipeline-parallel-size

## Core

- topic_id: `vllm.arg.pipeline_parallel_size`
- canonical_term: `--pipeline-parallel-size`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `general_runtime`
- status/confidence: `aligned` / `0.95`
- source: `code` / source_tags: code
- semantics: 通用运行时控制项，需要结合上下文确认语义。
- aliases: `--pipeline-parallel-size`, `pipeline-parallel-size`, `pipeline_parallel_size`, `pipeline parallel size`, `pipelineparallelsize`, `general_runtime`, `general runtime`, `general-runtime`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `general_runtime` 查看稳定原理。

## Deployment View

- default_behavior: 默认 1（不开启 PP）。
- value_shape: `numeric`
- accepted_values: int >= 1
- constraints: 模型必须支持 PP（SupportsPP），否则抛 NotImplementedError。; 非 Ray/MP/external_launcher 的后端下，PP>1 可能被判定为不支持。
- combo_effects: 与 tensor_parallel_size、data_parallel_size 共同决定 world_size。

## Development View

- definition_ref: vllm/engine/arg_utils.py:788
- read_ref: vllm/vllm/benchmarks/sweep/plot_pareto.py:83, vllm/vllm/benchmarks/sweep/plot_pareto.py:357, vllm/vllm/config/model.py:1063
- effect_ref: vllm/vllm/config/model.py:1064, vllm/vllm/config/parallel.py:205, vllm/vllm/config/vllm.py:973
- web_refs: 4

## Details/Edge Cases

- failure_modes: 启动失败; 行为与预期不符
- value_failure_signals: NotImplementedError: Pipeline parallelism is not supported for this model; unsupported: Pipeline Parallelism without Ray/mp/external launcher
- recommendation: 先查证代码与文档证据再启用。
- updated_at: 2026-03-11
