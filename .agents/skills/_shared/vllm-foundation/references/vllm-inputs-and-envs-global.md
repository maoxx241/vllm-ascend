---
knowledge_id: vllm-foundation.inputs-and-envs-global
domain: vllm-foundation
knowledge_type: reference
summary: Global inventory of vLLM serve arguments and environment variables.
applicable_vllm_versions: [">=0.15.0", "<0.17.0"]
applicable_cann_versions: [">=8.0.0"]
last_verified: "2026-03-05"
watch_files:
  - "../vllm/vllm/envs.py"
  - "../vllm/vllm/entrypoints/openai/cli_args.py"
  - "../vllm/vllm/engine/arg_utils.py"
depends_on:
  - "../../INDEX.md"
source_commit: "workspace-head"
freshness: "fresh"
---

# vLLM Global Inputs and Envs

Generated at: `2026-03-05`

- vLLM env vars discovered: **219**
- vLLM serve args discovered: **214**

## vLLM Serve Arguments (inventory)

| Argument | Kind | Source preview |
| --- | --- | --- |
| `--additional-config` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--aggregate-engine-logging` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--all2all-backend` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--allow-credentials` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--allow-deprecated-quantization` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--allowed-headers` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--allowed-local-media-path` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--allowed-media-domains` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--allowed-methods` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--allowed-origins` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--api-key` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--api-server-count` | vllm_arg | vllm/entrypoints/openai/cli_args.py |
| `--async-scheduling` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--attention-backend` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--attention-config` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--block-size` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--calculate-kv-scales` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--chat-template` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--chat-template-content-format` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--code-revision` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--collect-detailed-traces` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--compilation-config` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--config` | vllm_arg | vllm/entrypoints/openai/cli_args.py |
| `--config-format` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--convert` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--cp-kv-cache-interleave-size` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--cpu-offload-gb` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--cudagraph-capture-sizes` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--cudagraph-metrics` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--data-parallel-address` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--data-parallel-backend` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--data-parallel-external-lb` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--data-parallel-hybrid-lb` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--data-parallel-rank` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--data-parallel-rpc-port` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--data-parallel-size` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--data-parallel-size-local` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--data-parallel-start-rank` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--dbo-decode-token-threshold` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--dbo-prefill-token-threshold` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--dcp-kv-cache-interleave-size` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--decode-context-parallel-size` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--default-chat-template-kwargs` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--default-mm-loras` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--disable-access-log-for-endpoints` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--disable-cascade-attn` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--disable-chunked-mm-input` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--disable-custom-all-reduce` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--disable-fastapi-docs` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--disable-frontend-multiprocessing` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--disable-hybrid-kv-cache-manager` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--disable-log-requests` | vllm_arg | vllm/engine/arg_utils.py |
| `--disable-log-stats` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--disable-nccl-for-dp-synchronization` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--disable-sliding-window` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--disable-uvicorn-access-log` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--distributed-executor-backend` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--download-dir` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--dtype` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--ec-transfer-config` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--enable-auto-tool-choice` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--enable-chunked-prefill` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--enable-dbo` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--enable-eplb` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--enable-expert-parallel` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--enable-flashinfer-autotune` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--enable-force-include-usage` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--enable-layerwise-nvtx-tracing` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--enable-log-deltas` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--enable-log-outputs` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--enable-log-requests` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:AsyncEngineArgs |
| `--enable-logging-iteration-details` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--enable-lora` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--enable-mfu-metrics` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--enable-mm-embeds` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--enable-mm-processor-stats` | vllm_arg | vllm/engine/arg_utils.py:EngineArgs |
| `--enable-offline-docs` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--enable-prefix-caching` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--enable-prompt-embeds` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--enable-prompt-tokens-details` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--enable-request-id-headers` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--enable-return-routed-experts` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--enable-server-load-tracking` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--enable-sleep-mode` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--enable-ssl-refresh` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--enable-tokenizer-info-endpoint` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--enable-tower-connector-lora` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--enforce-eager` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--eplb-config` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--exclude-tools-when-tool-choice-none` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--expert-placement-strategy` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--fully-sharded-loras` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--generation-config` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--gpu-memory-utilization` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--h11-max-header-count` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--h11-max-incomplete-event-size` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--headless` | vllm_arg | vllm/entrypoints/openai/cli_args.py |
| `--hf-config-path` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--hf-overrides` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--hf-token` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--host` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--ignore-patterns` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--interleave-mm-strings` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--io-processor-plugin` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--kernel-config` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--kv-cache-dtype` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--kv-cache-memory-bytes` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--kv-cache-metrics` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--kv-cache-metrics-sample` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--kv-events-config` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--kv-offloading-backend` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--kv-offloading-size` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--kv-sharing-fast-prefill` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--kv-transfer-config` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--limit-mm-per-prompt` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--load-format` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--log-config-file` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--log-error-stack` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--logits-processor-pattern` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--logits-processors` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--logprobs-mode` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--long-prefill-token-threshold` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--lora-dtype` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--lora-modules` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--mamba-block-size` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--mamba-cache-dtype` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--mamba-cache-mode` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--mamba-ssm-cache-dtype` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--master-addr` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--master-port` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--max-cpu-loras` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--max-cudagraph-capture-size` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--max-log-len` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--max-logprobs` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--max-long-partial-prefills` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--max-lora-rank` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--max-loras` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--max-model-len` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--max-num-batched-tokens` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--max-num-partial-prefills` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--max-num-seqs` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--max-parallel-loading-workers` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--media-io-kwargs` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--middleware` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--mm-encoder-attn-backend` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--mm-encoder-only` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--mm-encoder-tp-mode` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--mm-processor-cache-gb` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--mm-processor-cache-type` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--mm-processor-kwargs` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--mm-shm-cache-max-object-size-mb` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--model` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--model-impl` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--model-loader-extra-config` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--model-weights` | vllm_arg | vllm/engine/arg_utils.py:EngineArgs |
| `--nnodes` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--node-rank` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--num-gpu-blocks-override` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--optimization-level` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--otlp-traces-endpoint` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--override-attention-dtype` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--override-generation-config` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--pipeline-parallel-size` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--pooler-config` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--port` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--prefill-context-parallel-size` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--prefix-caching-hash-algo` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--profiler-config` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--pt-load-map-location` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--quantization` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--ray-workers-use-nsight` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--reasoning-parser` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--reasoning-parser-plugin` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--response-role` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--return-tokens-as-token-ids` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--revision` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--root-path` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--runner` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--safetensors-load-strategy` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--scheduler-cls` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--scheduling-policy` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--seed` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--served-model-name` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--show-hidden-metrics-for-version` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--skip-mm-profiling` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--skip-tokenizer-init` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--specialize-active-lora` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--speculative-config` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--ssl-ca-certs` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--ssl-cert-reqs` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--ssl-certfile` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--ssl-ciphers` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--ssl-keyfile` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--stream-interval` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--structured-outputs-config` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--swap-space` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--tensor-parallel-size` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--tokenizer` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--tokenizer-mode` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--tokenizer-revision` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--tokens-only` | vllm_arg | vllm/engine/arg_utils.py:EngineArgs, vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--tool-call-parser` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--tool-parser-plugin` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--tool-server` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--trust-remote-code` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--trust-request-chat-template` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--ubatch-size` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--uds` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--use-tqdm-on-load` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--uvicorn-log-level` | vllm_arg | vllm/entrypoints/openai/cli_args.py:FrontendArgs |
| `--video-pruning-rate` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--weight-transfer-config` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--worker-cls` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |
| `--worker-extension-cls` | vllm_arg | vllm/engine/arg_utils.py, vllm/engine/arg_utils.py:EngineArgs |

## vLLM Environment Variables (inventory)

| Variable | Kind | Source preview |
| --- | --- | --- |
| `CMAKE_BUILD_TYPE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `CUDA_HOME` | vllm_env | vllm/envs.py:environment_variables |
| `CUDA_VISIBLE_DEVICES` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `K_SCALE_CONSTANT` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `LD_LIBRARY_PATH` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `LOCAL_RANK` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `MAX_JOBS` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `NOTE` | vllm_env | vllm/envs.py:type_checking |
| `NO_COLOR` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `NVCC_THREADS` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `Q_SCALE_CONSTANT` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `S3_ACCESS_KEY_ID` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `S3_ENDPOINT_URL` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `S3_SECRET_ACCESS_KEY` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VERBOSE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ALLOW_INSECURE_SERIALIZATION` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ALLOW_RUNTIME_LORA_UPDATING` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ALLREDUCE_USE_SYMM_MEM` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_API_KEY` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ASSETS_CACHE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ASSETS_CACHE_MODEL_CLEAN` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_AUDIO_FETCH_TIMEOUT` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_CACHE_ROOT` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_CI_USE_S3` | vllm_env | vllm/envs.py:environment_variables |
| `VLLM_COMPILE_CACHE_SAVE_FORMAT` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_COMPUTE_NANS_IN_LOGITS` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_CONFIGURE_LOGGING` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_CONFIG_ROOT` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_CPU_KVCACHE_SPACE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_CPU_NUM_OF_RESERVED_CPU` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_CPU_OMP_THREADS_BIND` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_CPU_SGL_KERNEL` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_CUDART_SO_PATH` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_CUSTOM_SCOPES_FOR_PROFILING` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_DBO_COMM_SMS` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_DEBUG_DUMP_PATH` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_DEBUG_LOG_API_SERVER_RESPONSE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_DEBUG_MFU_METRICS` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_DEBUG_WORKSPACE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_DEEPEPLL_NVFP4_DISPATCH` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_DEEPEP_BUFFER_SIZE_MB` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_DEEP_GEMM_WARMUP` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_DISABLED_KERNELS` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_DISABLE_COMPILE_CACHE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_DISABLE_LOG_LOGO` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_DISABLE_PYNCCL` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_DISABLE_SHARED_EXPERTS_STREAM` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_DOCKER_BUILD_CONTEXT` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_DO_NOT_TRACK` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_DP_MASTER_IP` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_DP_MASTER_PORT` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_DP_RANK` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_DP_RANK_LOCAL` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_DP_SIZE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ENABLE_CUDAGRAPH_GC` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ENABLE_MOE_DP_CHUNK` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ENABLE_RESPONSES_API_STORE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ENGINE_ITERATION_TIMEOUT_S` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ENGINE_READY_TIMEOUT_S` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB` | vllm_env | vllm/envs.py:environment_variables |
| `VLLM_FLASHINFER_MOE_BACKEND` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_FLOAT32_MATMUL_PRECISION` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_FORCE_AOT_LOAD` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_FUSED_MOE_CHUNK_SIZE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_GC_DEBUG` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_HAS_FLASHINFER_CUBIN` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_HOST_IP` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_HTTP_TIMEOUT_KEEP_ALIVE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_IMAGE_FETCH_TIMEOUT` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_KEEP_ALIVE_ON_ENGINE_DEATH` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_KV_CACHE_LAYOUT` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_LOGGING_COLOR` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_LOGGING_CONFIG_PATH` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_LOGGING_LEVEL` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_LOGGING_PREFIX` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_LOGGING_STREAM` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_LOG_BATCHSIZE_INTERVAL` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_LOG_MODEL_INSPECTION` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_LOG_STATS_INTERVAL` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_LOOPBACK_IP` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_LORA_DISABLE_PDL` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_LORA_RESOLVER_CACHE_DIR` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_LORA_RESOLVER_HF_REPO_LIST` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_MAIN_CUDA_VERSION` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_MARLIN_INPUT_DTYPE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_MARLIN_USE_ATOMIC_ADD` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_MEDIA_CONNECTOR` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_MEDIA_LOADING_THREAD_COUNT` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_MEDIA_URL_ALLOW_REDIRECTS` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_MLA_DISABLE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_MM_HASHER_ALGORITHM` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_MODEL_REDIRECT_PATH` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_MOE_DP_CHUNK_SIZE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_MOE_ROUTING_SIMULATION_STRATEGY` | vllm_env | vllm/envs.py:environment_variables |
| `VLLM_MOE_USE_DEEP_GEMM` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_MOONCAKE_BOOTSTRAP_PORT` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_MORIIO_CONNECTOR_READ_MODE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_MORIIO_NUM_WORKERS` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_MORIIO_POST_BATCH_SIZE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_MORIIO_QP_PER_TRANSFER` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_MQ_MAX_CHUNK_BYTES_MB` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_MSGPACK_ZERO_COPY_THRESHOLD` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_MXFP4_USE_MARLIN` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_NCCL_INCLUDE_PATH` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_NCCL_SO_PATH` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_NIXL_ABORT_REQUEST_TIMEOUT` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_NIXL_SIDE_CHANNEL_HOST` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_NIXL_SIDE_CHANNEL_PORT` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_NO_USAGE_STATS` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_NVFP4_GEMM_BACKEND` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_NVTX_SCOPES_FOR_PROFILING` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_PATTERN_MATCH_DEBUG` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_PLUGINS` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_PORT` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_PP_LAYER_PARTITION` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_PROCESS_NAME_PREFIX` | vllm_env | vllm/envs.py:environment_variables |
| `VLLM_RANDOMIZE_DP_DUMMY_INPUTS` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_RAY_BUNDLE_INDICES` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_RAY_DP_PACK_STRATEGY` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_RAY_PER_WORKER_GPUS` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_RINGBUFFER_WARNING_INTERVAL` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ROCM_CUSTOM_PAGED_ATTN` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ROCM_FP8_MFMA_PAGE_ATTN` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ROCM_FP8_PADDING` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ROCM_MOE_PADDING` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ROCM_USE_AITER` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ROCM_USE_AITER_FP4BMM` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ROCM_USE_AITER_FP8BMM` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ROCM_USE_AITER_LINEAR` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ROCM_USE_AITER_MHA` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ROCM_USE_AITER_MLA` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ROCM_USE_AITER_MOE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ROCM_USE_AITER_PAGED_ATTN` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ROCM_USE_AITER_RMSNORM` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ROCM_USE_AITER_TRITON_GEMM` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ROCM_USE_AITER_TRITON_ROPE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_ROCM_USE_SKINNY_GEMM` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_RPC_BASE_PATH` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_RPC_TIMEOUT` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_SERVER_DEV_MODE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_SKIP_P2P_CHECK` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_SLEEP_WHEN_IDLE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_TARGET_DEVICE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_TEST_FORCE_FP8_MARLIN` | vllm_env | vllm/envs.py:environment_variables |
| `VLLM_TEST_FORCE_LOAD_FORMAT` | vllm_env | vllm/envs.py:environment_variables |
| `VLLM_TOOL_JSON_ERROR_AUTOMATIC_RETRY` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_TPU_BUCKET_PADDING_GAP` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_TPU_MOST_MODEL_LEN` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_TPU_USING_PATHWAYS` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_TRACE_FUNCTION` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_TUNED_CONFIG_FOLDER` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_USAGE_SOURCE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_USAGE_STATS_SERVER` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_USE_AOT_COMPILE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_USE_BYTECODE_HOOK` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_USE_DEEP_GEMM` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_USE_DEEP_GEMM_E8M0` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_USE_FBGEMM` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_USE_FLASHINFER_MOE_FP16` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_USE_FLASHINFER_MOE_FP4` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_USE_FLASHINFER_MOE_FP8` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_USE_FLASHINFER_MOE_INT4` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_USE_FLASHINFER_MOE_MXFP4_BF16` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_USE_FLASHINFER_SAMPLER` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_USE_FUSED_MOE_GROUPED_TOPK` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_USE_MEGA_AOT_ARTIFACT` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_USE_MODELSCOPE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_USE_NCCL_SYMM_MEM` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_USE_NVFP4_CT_EMULATIONS` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_USE_PRECOMPILED` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_USE_RAY_WRAPPED_PP_COMM` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_USE_STANDALONE_COMPILE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_USE_TRITON_AWQ` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_USE_V2_MODEL_RUNNER` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_V1_USE_OUTLINES_CACHE` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_VIDEO_FETCH_TIMEOUT` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_VIDEO_LOADER_BACKEND` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_WORKER_MULTIPROC_METHOD` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_XGRAMMAR_CACHE_MB` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_XLA_CACHE_PATH` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_XLA_CHECK_RECOMPILATION` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `VLLM_XLA_USE_SPMD` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |
| `V_SCALE_CONSTANT` | vllm_env | vllm/envs.py:environment_variables, vllm/envs.py:type_checking |

Detailed semantics and combinations:
- `../../deployment-config/references/global-parameter-feature-map.md`
- `../../deployment-config/references/global-parameter-combination-guide.md`

Machine-readable artifacts:
- `generated/vllm_args_inventory.json`
- `generated/vllm_env_inventory.json`

Back to [INDEX](../../INDEX.md).
