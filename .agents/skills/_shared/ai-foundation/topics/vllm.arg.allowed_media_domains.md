---
topic_id: vllm.arg.allowed_media_domains
canonical_term: --allowed-media-domains
topic_kind: parameter
---

# --allowed-media-domains

## Core

- topic_id: `vllm.arg.allowed_media_domains`
- canonical_term: `--allowed-media-domains`
- kind/scope: `arg` / `vllm`
- stage: `startup`
- primary_feature: `multimodal`
- status/confidence: `aligned` / `0.88`
- source: `code` / source_tags: code
- semantics: 控制多模态输入处理和缓存策略。
- aliases: `--allowed-media-domains`, `allowed-media-domains`, `allowed_media_domains`, `allowed media domains`, `allowedmediadomains`, `multimodal`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `multimodal` 查看稳定原理。

## Deployment View

- default_behavior: 未显式设置时使用系统默认行为。
- value_shape: `free_form`
- accepted_values: string value
- constraints: 不支持多模态的模型无法启用相关参数
- combo_effects: N/A

## Development View

- definition_ref: vllm/engine/arg_utils.py:663
- read_ref: vllm/vllm/config/model.py:159, vllm/vllm/config/model.py:330, vllm/vllm/config/speculative.py:393
- effect_ref: vllm/vllm/entrypoints/chat_utils.py:495, vllm/vllm/entrypoints/llm.py:124, vllm/vllm/multimodal/media/connector.py:54
- web_refs: 3

## Details/Edge Cases

- failure_modes: 输入解析失败; 处理时延过高
- value_failure_signals: 输入解析失败; 处理时延过高
- recommendation: 先限制每请求多模态资源，再放开。
- updated_at: 2026-03-11
