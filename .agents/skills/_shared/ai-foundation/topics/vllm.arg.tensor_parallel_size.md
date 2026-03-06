---
topic_id: vllm.arg.tensor_parallel_size
canonical_term: --tensor-parallel-size
topic_kind: parameter
---

# --tensor-parallel-size

## Core

- topic_id: `vllm.arg.tensor_parallel_size`
- canonical_term: `--tensor-parallel-size`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `tensor_parallel`
- status/confidence: `aligned` / `0.98`
- source: `code` / source_tags: code
- semantics: 设置 TP 并行度，直接影响通信拓扑与单模型吞吐。
- aliases: `--tensor-parallel-size`, `tensor-parallel-size`, `tensor_parallel_size`, `tensor parallel size`, `tensorparallelsize`, `tensor_parallel`, `tensor parallel`, `tensor-parallel`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `tensor_parallel` 查看稳定原理。

## Deployment View

- default_behavior: 默认 1（不开启 TP）。
- value_shape: `numeric`
- accepted_values: int >= 1
- constraints: 模型 attention head 总数必须可被 TP 整除。; decode_context_parallel_size 必须整除 TP（tp_size % dcp_size == 0）。
- combo_effects: 与 data/pipeline/expert parallel 联动决定并行拓扑。; 与 all2all_backend、enable_sp 等编译路径联动影响图模式可用性。

## Development View

- definition_ref: vllm/engine/arg_utils.py:798
- read_ref: vllm/vllm/benchmarks/lib/utils.py:81, vllm/vllm/benchmarks/lib/utils.py:82, vllm/vllm/benchmarks/lib/utils.py:83
- effect_ref: vllm/vllm/benchmarks/lib/utils.py:82, vllm/vllm/benchmarks/lib/utils.py:83, vllm/vllm/config/compilation.py:1115
- web_refs: 6

## Details/Edge Cases

- failure_modes: HCCL/NCCL 初始化失败; 跨卡通信超时
- value_failure_signals: ValueError: Total number of attention heads ... must be divisible by tensor parallel size; ValueError: tp_size must be divisible by dcp_size
- recommendation: TP 变更后同步检查 max_model_len 与通信环境变量。
- updated_at: 2026-03-06
