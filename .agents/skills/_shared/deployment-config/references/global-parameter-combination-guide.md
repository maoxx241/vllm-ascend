---
knowledge_id: deployment-config.global-parameter-combination-guide
domain: deployment-config
knowledge_type: procedure
summary: Global combination guidance for vLLM and vLLM-Ascend parameters and env vars.
applicable_vllm_versions: [">=0.15.0", "<0.17.0"]
applicable_cann_versions: [">=8.0.0"]
last_verified: "2026-03-05"
watch_files:
  - "../vllm-foundation/references/vllm-inputs-and-envs-global.md"
  - "../vllm-ascend-core/references/vllm-ascend-inputs-and-envs-global.md"
  - "../vllm-ascend-core/concepts/model-feature-compatibility-matrix.md"
depends_on:
  - "../../INDEX.md"
source_commit: "workspace-head"
freshness: "fresh"
---

# Global Parameter Combination Guide

## Global decision order

1. Resolve intent to canonical features.
2. Check profile-level hard blocks before rendering commands.
3. Select core arg stack and env stack by feature tags.
4. Generate start/validate/rollback package.

## High-impact feature stacks

1. Quantized throughput stack
- `--quantization` + `--model` + `--tensor-parallel-size` + `--max-num-batched-tokens`.
2. Graph acceleration stack
- `--compilation-config` + `--enforce-eager` (for A/B fallback) + `--max-model-len`.
3. Parallel scale-out stack
- `--tensor-parallel-size` + `--data-parallel-size` + `--distributed-executor-backend`.
4. Long context stack
- `--prefill-context-parallel-size` + `--decode-context-parallel-size` + `--max-model-len`.
5. Prefill/decode split stack
- `--kv-transfer-config` + DP addressing + decode/prefill endpoint args.

## Co-occurrence evidence (from docs/examples/tests)

1. `--max-model-len` + `--tensor-parallel-size` (co-occurrence files: 65)
2. `--max-model-len` + `--port` (co-occurrence files: 64)
3. `--port` + `--trust-remote-code` (co-occurrence files: 64)
4. `--gpu-memory-utilization` + `--port` (co-occurrence files: 63)
5. `--max-model-len` + `--trust-remote-code` (co-occurrence files: 63)
6. `--port` + `--tensor-parallel-size` (co-occurrence files: 63)
7. `--gpu-memory-utilization` + `--max-model-len` (co-occurrence files: 62)
8. `--gpu-memory-utilization` + `--tensor-parallel-size` (co-occurrence files: 62)
9. `--max-num-batched-tokens` + `--port` (co-occurrence files: 62)
10. `--tensor-parallel-size` + `--trust-remote-code` (co-occurrence files: 62)
11. `--gpu-memory-utilization` + `--max-num-batched-tokens` (co-occurrence files: 61)
12. `--max-model-len` + `--max-num-batched-tokens` (co-occurrence files: 61)
13. `--max-num-batched-tokens` + `--tensor-parallel-size` (co-occurrence files: 61)
14. `--gpu-memory-utilization` + `--trust-remote-code` (co-occurrence files: 60)
15. `--max-num-batched-tokens` + `--trust-remote-code` (co-occurrence files: 59)
16. `--gpu-memory-utilization` + `--max-num-seqs` (co-occurrence files: 51)
17. `--max-model-len` + `--max-num-seqs` (co-occurrence files: 51)
18. `--max-num-seqs` + `--port` (co-occurrence files: 51)
19. `--max-num-batched-tokens` + `--max-num-seqs` (co-occurrence files: 50)
20. `--max-num-seqs` + `--tensor-parallel-size` (co-occurrence files: 50)
21. `--max-num-seqs` + `--trust-remote-code` (co-occurrence files: 50)
22. `--enable-expert-parallel` + `--tensor-parallel-size` (co-occurrence files: 49)
23. `--enable-expert-parallel` + `--trust-remote-code` (co-occurrence files: 48)
24. `--host` + `--port` (co-occurrence files: 48)
25. `--no-enable-prefix-caching` + `--port` (co-occurrence files: 47)
26. `--no-enable-prefix-caching` + `--trust-remote-code` (co-occurrence files: 46)
27. `--enable-expert-parallel` + `--max-model-len` (co-occurrence files: 45)
28. `--max-num-batched-tokens` + `--no-enable-prefix-caching` (co-occurrence files: 45)
29. `--no-enable-prefix-caching` + `--tensor-parallel-size` (co-occurrence files: 45)
30. `--enable-expert-parallel` + `--gpu-memory-utilization` (co-occurrence files: 44)

## Hard blocks in demo profiles

- `qwen3-32b-w8a8 + int4_quantization`: blocked.
- `qwen3-32b-w8a8 + expert_parallel`: blocked.

## Error-case handling

- If user asks `qwen3 32b w8a8 开 int4`: return blocked reason + suggest int4-ready artifact/profile switch.
- If user asks `qwen3 32b w8a8 开 ep`: return dense-model incompatibility + suggest TP/DP path.

## Weak model guardrails

- Never execute ambiguous request directly; ask one clarification with <=3 candidates.
- Keep one-decision-per-step output: params table -> commands -> validation -> rollback.

Back to [INDEX](../../INDEX.md).
