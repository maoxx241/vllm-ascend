---
topic_id: vllm.env.vllm_max_audio_clip_filesize_mb
canonical_term: VLLM_MAX_AUDIO_CLIP_FILESIZE_MB
topic_kind: parameter
---

# VLLM_MAX_AUDIO_CLIP_FILESIZE_MB

## Core

- topic_id: `vllm.env.vllm_max_audio_clip_filesize_mb`
- canonical_term: `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB`
- kind/scope: `env` / `vllm`
- stage: `runtime`
- primary_feature: `multimodal`
- status/confidence: `aligned` / `0.91`
- semantics: 控制多模态输入处理和缓存策略。
- aliases: `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB`, `vllm_max_audio_clip_filesize_mb`, `vllm-max-audio-clip-filesize-mb`, `vllm max audio clip filesize mb`, `multimodal`

## Foundation

- 该条目属于部署/推理配置知识，基础语义以代码证据为主。
- 推荐结合 feature: `multimodal` 查看稳定原理。

## Deployment View

- default_behavior: 使用默认值。
- value_shape: `numeric`
- accepted_values: int value
- constraints: 不支持多模态的模型无法启用相关参数
- combo_effects: N/A

## Development View

- definition_ref: vllm/envs.py:754
- read_ref: vllm/vllm/entrypoints/openai/realtime/connection.py:53, vllm/vllm/entrypoints/openai/speech_to_text/speech_to_text.py:106, vllm/vllm/envs.py:70
- effect_ref: vllm/vllm/entrypoints/openai/realtime/connection.py:53, vllm/vllm/entrypoints/openai/speech_to_text/speech_to_text.py:106, vllm/vllm/envs.py:70
- web_refs: 2

## Details/Edge Cases

- failure_modes: 输入解析失败; 处理时延过高
- value_failure_signals: 输入解析失败; 处理时延过高
- recommendation: 先限制每请求多模态资源，再放开。
- updated_at: 2026-03-05
