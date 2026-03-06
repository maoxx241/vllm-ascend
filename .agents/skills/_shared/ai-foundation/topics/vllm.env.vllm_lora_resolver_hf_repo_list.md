---
topic_id: vllm.env.vllm_lora_resolver_hf_repo_list
canonical_term: VLLM_LORA_RESOLVER_HF_REPO_LIST
topic_kind: parameter
---

# VLLM_LORA_RESOLVER_HF_REPO_LIST

## Core

- topic_id: `vllm.env.vllm_lora_resolver_hf_repo_list`
- canonical_term: `VLLM_LORA_RESOLVER_HF_REPO_LIST`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `expert_parallel`
- status/confidence: `aligned` / `0.91`
- source: `code` / source_tags: code_definition
- semantics: MoE 专家并行，提升大规模专家模型吞吐。
- aliases: `VLLM_LORA_RESOLVER_HF_REPO_LIST`, `vllm_lora_resolver_hf_repo_list`, `vllm-lora-resolver-hf-repo-list`, `vllm lora resolver hf repo list`, `expert_parallel`, `expert parallel`, `expert-parallel`

## Foundation

- EP 面向 MoE 专家路由，Dense 模型没有专家层时不成立。
- 推荐结合 feature: `expert_parallel` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `list`
- accepted_values: list value
- constraints: Dense 模型不适用
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:855
- read_ref: vllm/vllm/envs.py:90, vllm/vllm/envs.py:855, vllm/vllm/envs.py:856
- effect_ref: vllm/vllm/envs.py:90, vllm/vllm/envs.py:855, vllm/vllm/envs.py:856
- web_refs: 3

## Details/Edge Cases

- failure_modes: 启动报模型不支持 EP; 专家路由异常
- value_failure_signals: 启动报模型不支持 EP; 专家路由异常
- recommendation: 仅在 MoE profile 启用，并配合 TP/DP 校验。
- updated_at: 2026-03-06
