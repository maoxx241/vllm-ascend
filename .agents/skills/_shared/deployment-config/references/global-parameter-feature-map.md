---
knowledge_id: deployment-config.global-parameter-feature-map
domain: deployment-config
knowledge_type: reference
summary: Global semantic map for vLLM and vLLM-Ascend args/envs with usage and combination hints.
applicable_vllm_versions: [">=0.15.0", "<0.17.0"]
applicable_cann_versions: [">=8.0.0"]
last_verified: "2026-03-05"
watch_files:
  - "../vllm-foundation/references/vllm-inputs-and-envs-global.md"
  - "../vllm-ascend-core/references/vllm-ascend-inputs-and-envs-global.md"
depends_on:
  - "../../INDEX.md"
source_commit: "workspace-head"
freshness: "fresh"
---

# Global Parameter Feature Map

This document gives a global view for weak-model execution: every discovered parameter/env is mapped to feature semantics, usage intent, and common combinations.

## Coverage

- vLLM args: **214**
- vLLM envs: **219**
- vLLM-Ascend args (observed): **159**
- vLLM-Ascend envs: **24**

## Feature tags

`quantization`, `graph_mode`, `tensor_parallel`, `data_parallel`, `expert_parallel`, `context_parallel`, `prefill_decode_disaggregation`, `prefix_cache`, `lora`, `speculative_decode`, `weight_prefetch`, `sleep_mode`, `throughput_tuning`, `memory_tuning`, `network_serving`, `security_auth`, `multimodal`, `logging_debug`, `profiling_observability`, `model_selection`, `general_runtime`.

## vLLM Serve Args -> Semantics

| Parameter | Primary feature | Secondary features | Usage | Common combinations |
| --- | --- | --- | --- | --- |
| `--additional-config` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--aggregate-engine-logging` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `--all2all-backend` | `tensor_parallel` | - | Splits model tensors across NPUs/GPUs for scale-out inference. | `--tensor-parallel-size`, `--data-parallel-size`, `--distributed-executor-backend` |
| `--allow-credentials` | `security_auth` | - | Controls authentication, TLS, and request trust boundaries. | `--api-key`, `--ssl-certfile`, `--allowed-origins` |
| `--allow-deprecated-quantization` | `quantization` | - | Controls model precision and quantized weight loading path. | `--model`, `--dtype`, `--tensor-parallel-size` |
| `--allowed-headers` | `network_serving` | `security_auth` | Controls API host/port/endpoints and serving interface. | `--host`, `--port`, `--served-model-name` |
| `--allowed-local-media-path` | `network_serving` | `multimodal` | Controls API host/port/endpoints and serving interface. | `--host`, `--port`, `--served-model-name` |
| `--allowed-media-domains` | `network_serving` | `multimodal` | Controls API host/port/endpoints and serving interface. | `--host`, `--port`, `--served-model-name` |
| `--allowed-methods` | `network_serving` | `security_auth` | Controls API host/port/endpoints and serving interface. | `--host`, `--port`, `--served-model-name` |
| `--allowed-origins` | `network_serving` | `security_auth` | Controls API host/port/endpoints and serving interface. | `--host`, `--port`, `--served-model-name` |
| `--api-key` | `security_auth` | - | Controls authentication, TLS, and request trust boundaries. | `--ssl-certfile`, `--allowed-origins` |
| `--api-server-count` | `network_serving` | - | Controls API host/port/endpoints and serving interface. | `--data-parallel-rpc-port` |
| `--async-scheduling` | `throughput_tuning` | - | Tunes scheduler and batching for higher throughput. | `--max-num-batched-tokens` |
| `--attention-backend` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--attention-config` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--block-size` | `memory_tuning` | - | Bounds memory pressure and sequence length behavior. | `--gpu-memory-utilization`, `--max-model-len` |
| `--calculate-kv-scales` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--chat-template` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--backend`, `--dataset-name` |
| `--chat-template-content-format` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--dataset-name`, `--device` |
| `--code-revision` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `--collect-detailed-traces` | `logging_debug` | `profiling_observability` | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `--compilation-config` | `graph_mode` | - | Controls graph/eager execution and compile behavior. | `--enforce-eager`, `--max-num-batched-tokens` |
| `--config` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--config-format` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--convert` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `--cp-kv-cache-interleave-size` | `context_parallel` | `memory_tuning` | Splits long-context KV processing across ranks. | `--decode-context-parallel-size`, `--prefill-context-parallel-size` |
| `--cpu-offload-gb` | `memory_tuning` | - | Bounds memory pressure and sequence length behavior. | `--gpu-memory-utilization`, `--max-model-len`, `--block-size` |
| `--cudagraph-capture-sizes` | `graph_mode` | - | Controls graph/eager execution and compile behavior. | `--compilation-config`, `--enforce-eager`, `--max-num-batched-tokens` |
| `--cudagraph-metrics` | `graph_mode` | `profiling_observability` | Controls graph/eager execution and compile behavior. | `--compilation-config`, `--enforce-eager`, `--max-num-batched-tokens` |
| `--data-parallel-address` | `data_parallel` | - | Replicates workers for throughput and multi-node serving. | `--data-parallel-rpc-port`, `--data-parallel-size` |
| `--data-parallel-backend` | `data_parallel` | - | Replicates workers for throughput and multi-node serving. | `--data-parallel-size`, `--data-parallel-address`, `--data-parallel-rpc-port` |
| `--data-parallel-external-lb` | `data_parallel` | - | Replicates workers for throughput and multi-node serving. | `--data-parallel-size`, `--data-parallel-address`, `--data-parallel-rpc-port` |
| `--data-parallel-hybrid-lb` | `data_parallel` | - | Replicates workers for throughput and multi-node serving. | `--data-parallel-size`, `--data-parallel-address`, `--data-parallel-rpc-port` |
| `--data-parallel-rank` | `data_parallel` | - | Replicates workers for throughput and multi-node serving. | `--data-parallel-size` |
| `--data-parallel-rpc-port` | `data_parallel` | `network_serving` | Replicates workers for throughput and multi-node serving. | `--data-parallel-size` |
| `--data-parallel-size` | `data_parallel` | - | Replicates workers for throughput and multi-node serving. | `--data-parallel-address`, `--data-parallel-rpc-port` |
| `--data-parallel-size-local` | `data_parallel` | - | Replicates workers for throughput and multi-node serving. | `--data-parallel-size` |
| `--data-parallel-start-rank` | `data_parallel` | - | Replicates workers for throughput and multi-node serving. | `--data-parallel-address`, `--data-parallel-rpc-port`, `--data-parallel-size` |
| `--dbo-decode-token-threshold` | `throughput_tuning` | - | Tunes scheduler and batching for higher throughput. | `--async-scheduling`, `--max-num-batched-tokens`, `--max-num-seqs` |
| `--dbo-prefill-token-threshold` | `prefill_decode_disaggregation` | `throughput_tuning` | Separates prefill/decode services or connectors. | `--kv-transfer-config`, `--data-parallel-size`, `--data-parallel-address` |
| `--dcp-kv-cache-interleave-size` | `context_parallel` | `memory_tuning` | Splits long-context KV processing across ranks. | `--prefill-context-parallel-size`, `--decode-context-parallel-size`, `--max-model-len` |
| `--decode-context-parallel-size` | `context_parallel` | - | Splits long-context KV processing across ranks. | `--prefill-context-parallel-size` |
| `--default-chat-template-kwargs` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--default-mm-loras` | `lora` | `multimodal` | Enables adapter loading and runtime LoRA routing. | `--enable-lora`, `--lora-modules`, `--max-loras` |
| `--disable-access-log-for-endpoints` | `network_serving` | `logging_debug` | Controls API host/port/endpoints and serving interface. | `--host`, `--port`, `--served-model-name` |
| `--disable-cascade-attn` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--disable-chunked-mm-input` | `multimodal` | - | Controls multimodal I/O paths and media preprocessing. | `--limit-mm-per-prompt`, `--mm-processor-cache-gb`, `--allowed-local-media-path` |
| `--disable-custom-all-reduce` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--disable-fastapi-docs` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--disable-frontend-multiprocessing` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--disable-hybrid-kv-cache-manager` | `memory_tuning` | - | Bounds memory pressure and sequence length behavior. | `--gpu-memory-utilization`, `--max-model-len`, `--block-size` |
| `--disable-log-requests` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `--disable-log-stats` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--max-log-len`, `--log-config-file` |
| `--disable-nccl-for-dp-synchronization` | `data_parallel` | - | Replicates workers for throughput and multi-node serving. | `--data-parallel-size`, `--data-parallel-address`, `--data-parallel-rpc-port` |
| `--disable-sliding-window` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--disable-uvicorn-access-log` | `network_serving` | `logging_debug` | Controls API host/port/endpoints and serving interface. | `--host`, `--port`, `--served-model-name` |
| `--distributed-executor-backend` | `data_parallel` | - | Replicates workers for throughput and multi-node serving. | `--data-parallel-size`, `--data-parallel-address`, `--data-parallel-rpc-port` |
| `--download-dir` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `--dtype` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `--ec-transfer-config` | `prefill_decode_disaggregation` | - | Separates prefill/decode services or connectors. | `--decode-servers-urls` |
| `--enable-auto-tool-choice` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--enable-chunked-prefill` | `prefill_decode_disaggregation` | `throughput_tuning` | Separates prefill/decode services or connectors. | `--kv-transfer-config`, `--data-parallel-size`, `--data-parallel-address` |
| `--enable-dbo` | `throughput_tuning` | - | Tunes scheduler and batching for higher throughput. | `--async-scheduling`, `--max-num-batched-tokens`, `--max-num-seqs` |
| `--enable-eplb` | `expert_parallel` | - | Enables MoE expert routing parallelism; only valid on MoE models. | `--enable-expert-parallel`, `--tensor-parallel-size`, `--data-parallel-size` |
| `--enable-expert-parallel` | `expert_parallel` | - | Enables MoE expert routing parallelism; only valid on MoE models. | `--tensor-parallel-size`, `--data-parallel-size` |
| `--enable-flashinfer-autotune` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--enable-force-include-usage` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--enable-layerwise-nvtx-tracing` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `--enable-log-deltas` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `--enable-log-outputs` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `--enable-log-requests` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `--enable-logging-iteration-details` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `--enable-lora` | `lora` | - | Enables adapter loading and runtime LoRA routing. | `--lora-modules` |
| `--enable-mfu-metrics` | `profiling_observability` | - | Controls profiling, traces, and metrics visibility. | `--profiler-config`, `--collect-detailed-traces`, `--otlp-traces-endpoint` |
| `--enable-mm-embeds` | `multimodal` | - | Controls multimodal I/O paths and media preprocessing. | `--limit-mm-per-prompt`, `--mm-processor-cache-gb`, `--allowed-local-media-path` |
| `--enable-mm-processor-stats` | `multimodal` | - | Controls multimodal I/O paths and media preprocessing. | `--limit-mm-per-prompt`, `--mm-processor-cache-gb`, `--allowed-local-media-path` |
| `--enable-offline-docs` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--enable-prefix-caching` | `prefix_cache` | - | Reuses shared prompt prefixes to reduce prefill cost. | `--prefix-caching-hash-algo`, `--max-model-len` |
| `--enable-prompt-embeds` | `multimodal` | - | Controls multimodal I/O paths and media preprocessing. | `--limit-mm-per-prompt`, `--mm-processor-cache-gb`, `--allowed-local-media-path` |
| `--enable-prompt-tokens-details` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--enable-request-id-headers` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--backend`, `--dataset-name` |
| `--enable-return-routed-experts` | `expert_parallel` | - | Enables MoE expert routing parallelism; only valid on MoE models. | `--enable-expert-parallel`, `--tensor-parallel-size`, `--data-parallel-size` |
| `--enable-server-load-tracking` | `network_serving` | - | Controls API host/port/endpoints and serving interface. | `--host`, `--port`, `--served-model-name` |
| `--enable-sleep-mode` | `sleep_mode` | - | Enables idle-time memory/power saving mode. | `--gpu-memory-utilization`, `--max-model-len` |
| `--enable-ssl-refresh` | `security_auth` | - | Controls authentication, TLS, and request trust boundaries. | `--api-key`, `--ssl-certfile`, `--allowed-origins` |
| `--enable-tokenizer-info-endpoint` | `network_serving` | `model_selection` | Controls API host/port/endpoints and serving interface. | `--host`, `--port`, `--served-model-name` |
| `--enable-tower-connector-lora` | `prefill_decode_disaggregation` | `lora` | Separates prefill/decode services or connectors. | `--kv-transfer-config`, `--data-parallel-size`, `--data-parallel-address` |
| `--enforce-eager` | `graph_mode` | - | Controls graph/eager execution and compile behavior. | `--compilation-config`, `--max-num-batched-tokens` |
| `--eplb-config` | `expert_parallel` | - | Enables MoE expert routing parallelism; only valid on MoE models. | `--enable-expert-parallel`, `--tensor-parallel-size`, `--data-parallel-size` |
| `--exclude-tools-when-tool-choice-none` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--expert-placement-strategy` | `expert_parallel` | - | Enables MoE expert routing parallelism; only valid on MoE models. | `--enable-expert-parallel`, `--tensor-parallel-size`, `--data-parallel-size` |
| `--fully-sharded-loras` | `lora` | - | Enables adapter loading and runtime LoRA routing. | `--enable-lora`, `--lora-modules`, `--max-loras` |
| `--generation-config` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--dataset-args` |
| `--gpu-memory-utilization` | `memory_tuning` | - | Bounds memory pressure and sequence length behavior. | `--max-model-len` |
| `--h11-max-header-count` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--h11-max-incomplete-event-size` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--headless` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--hf-config-path` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `--hf-overrides` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `--hf-token` | `security_auth` | `model_selection` | Controls authentication, TLS, and request trust boundaries. | `--api-key`, `--ssl-certfile`, `--allowed-origins` |
| `--host` | `network_serving` | - | Controls API host/port/endpoints and serving interface. | `--port` |
| `--ignore-patterns` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--interleave-mm-strings` | `multimodal` | - | Controls multimodal I/O paths and media preprocessing. | `--limit-mm-per-prompt`, `--mm-processor-cache-gb`, `--allowed-local-media-path` |
| `--io-processor-plugin` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--kernel-config` | `graph_mode` | - | Controls graph/eager execution and compile behavior. | `--compilation-config`, `--enforce-eager`, `--max-num-batched-tokens` |
| `--kv-cache-dtype` | `memory_tuning` | `model_selection` | Bounds memory pressure and sequence length behavior. | `--gpu-memory-utilization`, `--max-model-len`, `--block-size` |
| `--kv-cache-memory-bytes` | `memory_tuning` | - | Bounds memory pressure and sequence length behavior. | `--gpu-memory-utilization`, `--max-model-len`, `--block-size` |
| `--kv-cache-metrics` | `memory_tuning` | `profiling_observability` | Bounds memory pressure and sequence length behavior. | `--gpu-memory-utilization`, `--max-model-len`, `--block-size` |
| `--kv-cache-metrics-sample` | `memory_tuning` | `profiling_observability` | Bounds memory pressure and sequence length behavior. | `--gpu-memory-utilization`, `--max-model-len`, `--block-size` |
| `--kv-events-config` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--kv-offloading-backend` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--kv-offloading-size` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--kv-sharing-fast-prefill` | `prefill_decode_disaggregation` | - | Separates prefill/decode services or connectors. | `--kv-transfer-config`, `--data-parallel-size`, `--data-parallel-address` |
| `--kv-transfer-config` | `prefill_decode_disaggregation` | - | Separates prefill/decode services or connectors. | `--data-parallel-size`, `--data-parallel-address` |
| `--limit-mm-per-prompt` | `multimodal` | - | Controls multimodal I/O paths and media preprocessing. | `--allowed-local-media-path` |
| `--load-format` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--log-config-file` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len` |
| `--log-error-stack` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `--logits-processor-pattern` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `--logits-processors` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `--logprobs-mode` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `--long-prefill-token-threshold` | `prefill_decode_disaggregation` | `memory_tuning` | Separates prefill/decode services or connectors. | `--kv-transfer-config`, `--data-parallel-size`, `--data-parallel-address` |
| `--lora-dtype` | `lora` | `model_selection` | Enables adapter loading and runtime LoRA routing. | `--enable-lora`, `--lora-modules`, `--max-loras` |
| `--lora-modules` | `lora` | - | Enables adapter loading and runtime LoRA routing. | `--enable-lora` |
| `--mamba-block-size` | `memory_tuning` | - | Bounds memory pressure and sequence length behavior. | `--gpu-memory-utilization`, `--max-model-len`, `--block-size` |
| `--mamba-cache-dtype` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `--mamba-cache-mode` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--mamba-ssm-cache-dtype` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `--master-addr` | `data_parallel` | - | Replicates workers for throughput and multi-node serving. | `--master-port` |
| `--master-port` | `data_parallel` | `network_serving` | Replicates workers for throughput and multi-node serving. | `--master-addr` |
| `--max-cpu-loras` | `lora` | - | Enables adapter loading and runtime LoRA routing. | `--enable-lora`, `--lora-modules`, `--max-loras` |
| `--max-cudagraph-capture-size` | `graph_mode` | - | Controls graph/eager execution and compile behavior. | `--compilation-config`, `--enforce-eager`, `--max-num-batched-tokens` |
| `--max-log-len` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--log-config-file` |
| `--max-logprobs` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `--max-long-partial-prefills` | `prefill_decode_disaggregation` | - | Separates prefill/decode services or connectors. | `--kv-transfer-config`, `--data-parallel-size`, `--data-parallel-address` |
| `--max-lora-rank` | `lora` | - | Enables adapter loading and runtime LoRA routing. | `--enable-lora`, `--lora-modules`, `--max-loras` |
| `--max-loras` | `lora` | - | Enables adapter loading and runtime LoRA routing. | `--enable-lora`, `--lora-modules` |
| `--max-model-len` | `memory_tuning` | `model_selection` | Bounds memory pressure and sequence length behavior. | `--gpu-memory-utilization`, `--block-size` |
| `--max-num-batched-tokens` | `throughput_tuning` | - | Tunes scheduler and batching for higher throughput. | `--async-scheduling`, `--max-num-seqs` |
| `--max-num-partial-prefills` | `prefill_decode_disaggregation` | - | Separates prefill/decode services or connectors. | `--kv-transfer-config`, `--data-parallel-size`, `--data-parallel-address` |
| `--max-num-seqs` | `throughput_tuning` | - | Tunes scheduler and batching for higher throughput. | `--async-scheduling`, `--max-num-batched-tokens` |
| `--max-parallel-loading-workers` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--media-io-kwargs` | `multimodal` | - | Controls multimodal I/O paths and media preprocessing. | `--limit-mm-per-prompt`, `--mm-processor-cache-gb`, `--allowed-local-media-path` |
| `--middleware` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--mm-encoder-attn-backend` | `multimodal` | - | Controls multimodal I/O paths and media preprocessing. | `--limit-mm-per-prompt`, `--mm-processor-cache-gb`, `--allowed-local-media-path` |
| `--mm-encoder-only` | `multimodal` | - | Controls multimodal I/O paths and media preprocessing. | `--limit-mm-per-prompt`, `--mm-processor-cache-gb`, `--allowed-local-media-path` |
| `--mm-encoder-tp-mode` | `tensor_parallel` | `multimodal` | Splits model tensors across NPUs/GPUs for scale-out inference. | `--tensor-parallel-size`, `--data-parallel-size`, `--distributed-executor-backend` |
| `--mm-processor-cache-gb` | `memory_tuning` | `multimodal` | Bounds memory pressure and sequence length behavior. | `--gpu-memory-utilization`, `--max-model-len`, `--block-size` |
| `--mm-processor-cache-type` | `memory_tuning` | `multimodal` | Bounds memory pressure and sequence length behavior. | `--gpu-memory-utilization`, `--max-model-len`, `--block-size` |
| `--mm-processor-kwargs` | `multimodal` | - | Controls multimodal I/O paths and media preprocessing. | `--limit-mm-per-prompt`, `--mm-processor-cache-gb`, `--allowed-local-media-path` |
| `--mm-shm-cache-max-object-size-mb` | `multimodal` | - | Controls multimodal I/O paths and media preprocessing. | `--limit-mm-per-prompt`, `--mm-processor-cache-gb`, `--allowed-local-media-path` |
| `--model` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--tokenizer`, `--revision`, `--trust-remote-code` |
| `--model-impl` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `--model-loader-extra-config` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `--model-weights` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `--nnodes` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--node-rank` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--num-gpu-blocks-override` | `memory_tuning` | - | Bounds memory pressure and sequence length behavior. | `--gpu-memory-utilization`, `--max-model-len`, `--block-size` |
| `--optimization-level` | `graph_mode` | - | Controls graph/eager execution and compile behavior. | `--compilation-config`, `--enforce-eager`, `--max-num-batched-tokens` |
| `--otlp-traces-endpoint` | `network_serving` | `logging_debug`, `profiling_observability` | Controls API host/port/endpoints and serving interface. | `--host`, `--port`, `--served-model-name` |
| `--override-attention-dtype` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `--override-generation-config` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--pipeline-parallel-size` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--address`, `--device` |
| `--pooler-config` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--port` | `network_serving` | - | Controls API host/port/endpoints and serving interface. | `--host`, `--served-model-name` |
| `--prefill-context-parallel-size` | `context_parallel` | `prefill_decode_disaggregation` | Splits long-context KV processing across ranks. | `--decode-context-parallel-size`, `--cp-kv-cache-interleave-size` |
| `--prefix-caching-hash-algo` | `prefix_cache` | - | Reuses shared prompt prefixes to reduce prefill cost. | `--enable-prefix-caching`, `--max-model-len` |
| `--profiler-config` | `profiling_observability` | - | Controls profiling, traces, and metrics visibility. | `--collect-detailed-traces`, `--otlp-traces-endpoint` |
| `--pt-load-map-location` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--quantization` | `quantization` | - | Controls model precision and quantized weight loading path. | `--model`, `--dtype`, `--tensor-parallel-size` |
| `--ray-workers-use-nsight` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--reasoning-parser` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--additional-config` |
| `--reasoning-parser-plugin` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--response-role` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--return-tokens-as-token-ids` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--revision` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--trust-remote-code` |
| `--root-path` | `network_serving` | - | Controls API host/port/endpoints and serving interface. | `--host`, `--port`, `--served-model-name` |
| `--runner` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `--safetensors-load-strategy` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--scheduler-cls` | `throughput_tuning` | - | Tunes scheduler and batching for higher throughput. | `--async-scheduling`, `--max-num-batched-tokens`, `--max-num-seqs` |
| `--scheduling-policy` | `throughput_tuning` | - | Tunes scheduler and batching for higher throughput. | `--async-scheduling`, `--max-num-batched-tokens`, `--max-num-seqs` |
| `--seed` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--served-model-name` | `network_serving` | `model_selection` | Controls API host/port/endpoints and serving interface. | `--port` |
| `--show-hidden-metrics-for-version` | `profiling_observability` | - | Controls profiling, traces, and metrics visibility. | `--profiler-config`, `--collect-detailed-traces`, `--otlp-traces-endpoint` |
| `--skip-mm-profiling` | `multimodal` | - | Controls multimodal I/O paths and media preprocessing. | `--limit-mm-per-prompt`, `--mm-processor-cache-gb`, `--allowed-local-media-path` |
| `--skip-tokenizer-init` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `--specialize-active-lora` | `lora` | - | Enables adapter loading and runtime LoRA routing. | `--enable-lora`, `--lora-modules`, `--max-loras` |
| `--speculative-config` | `speculative_decode` | - | Enables draft/speculative decoding acceleration path. | `--max-num-batched-tokens`, `--async-scheduling` |
| `--ssl-ca-certs` | `security_auth` | - | Controls authentication, TLS, and request trust boundaries. | `--api-key`, `--ssl-certfile`, `--allowed-origins` |
| `--ssl-cert-reqs` | `security_auth` | - | Controls authentication, TLS, and request trust boundaries. | `--api-key`, `--ssl-certfile`, `--allowed-origins` |
| `--ssl-certfile` | `security_auth` | - | Controls authentication, TLS, and request trust boundaries. | `--api-key`, `--allowed-origins` |
| `--ssl-ciphers` | `security_auth` | - | Controls authentication, TLS, and request trust boundaries. | `--api-key`, `--ssl-certfile`, `--allowed-origins` |
| `--ssl-keyfile` | `security_auth` | - | Controls authentication, TLS, and request trust boundaries. | `--api-key`, `--ssl-certfile`, `--allowed-origins` |
| `--stream-interval` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--structured-outputs-config` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--swap-space` | `memory_tuning` | - | Bounds memory pressure and sequence length behavior. | `--gpu-memory-utilization`, `--max-model-len`, `--block-size` |
| `--tensor-parallel-size` | `tensor_parallel` | - | Splits model tensors across NPUs/GPUs for scale-out inference. | `--data-parallel-size`, `--distributed-executor-backend` |
| `--tokenizer` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model` |
| `--tokenizer-mode` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `--tokenizer-revision` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `--tokens-only` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--tool-call-parser` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--tool-parser-plugin` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--tool-server` | `network_serving` | - | Controls API host/port/endpoints and serving interface. | `--host`, `--port`, `--served-model-name` |
| `--trust-remote-code` | `security_auth` | - | Controls authentication, TLS, and request trust boundaries. | `--api-key`, `--ssl-certfile`, `--allowed-origins` |
| `--trust-request-chat-template` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--ubatch-size` | `throughput_tuning` | - | Tunes scheduler and batching for higher throughput. | `--async-scheduling`, `--max-num-batched-tokens`, `--max-num-seqs` |
| `--uds` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--use-tqdm-on-load` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--uvicorn-log-level` | `network_serving` | `logging_debug` | Controls API host/port/endpoints and serving interface. | `--host`, `--port`, `--served-model-name` |
| `--video-pruning-rate` | `multimodal` | - | Controls multimodal I/O paths and media preprocessing. | `--limit-mm-per-prompt`, `--mm-processor-cache-gb`, `--allowed-local-media-path` |
| `--weight-transfer-config` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--worker-cls` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--worker-extension-cls` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |

## vLLM Env Vars -> Semantics

| Variable | Primary feature | Secondary features | Usage | Common combinations |
| --- | --- | --- | --- | --- |
| `CMAKE_BUILD_TYPE` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `CUDA_HOME` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `CUDA_VISIBLE_DEVICES` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `K_SCALE_CONSTANT` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `LD_LIBRARY_PATH` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `LOCAL_RANK` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `MAX_JOBS` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `NOTE` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `NO_COLOR` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `NVCC_THREADS` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `Q_SCALE_CONSTANT` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `S3_ACCESS_KEY_ID` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `S3_ENDPOINT_URL` | `network_serving` | - | Controls API host/port/endpoints and serving interface. | `--host`, `--port`, `--served-model-name` |
| `S3_SECRET_ACCESS_KEY` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VERBOSE` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE` | `memory_tuning` | - | Bounds memory pressure and sequence length behavior. | `--gpu-memory-utilization`, `--max-model-len`, `--block-size` |
| `VLLM_ALLOW_INSECURE_SERIALIZATION` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `memory_tuning` | `model_selection` | Bounds memory pressure and sequence length behavior. | `--gpu-memory-utilization`, `--max-model-len`, `--block-size` |
| `VLLM_ALLOW_RUNTIME_LORA_UPDATING` | `lora` | - | Enables adapter loading and runtime LoRA routing. | `--enable-lora`, `--lora-modules`, `--max-loras` |
| `VLLM_ALLREDUCE_USE_SYMM_MEM` | `multimodal` | - | Controls multimodal I/O paths and media preprocessing. | `--limit-mm-per-prompt`, `--mm-processor-cache-gb`, `--allowed-local-media-path` |
| `VLLM_API_KEY` | `security_auth` | - | Controls authentication, TLS, and request trust boundaries. | `--api-key`, `--ssl-certfile`, `--allowed-origins` |
| `VLLM_ASSETS_CACHE` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_ASSETS_CACHE_MODEL_CLEAN` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `VLLM_AUDIO_FETCH_TIMEOUT` | `multimodal` | - | Controls multimodal I/O paths and media preprocessing. | `--limit-mm-per-prompt`, `--mm-processor-cache-gb`, `--allowed-local-media-path` |
| `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER` | `quantization` | `multimodal` | Controls model precision and quantized weight loading path. | `--model`, `--dtype`, `--tensor-parallel-size` |
| `VLLM_CACHE_ROOT` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_CI_USE_S3` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_COMPILE_CACHE_SAVE_FORMAT` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_COMPUTE_NANS_IN_LOGITS` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `VLLM_CONFIGURE_LOGGING` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `VLLM_CONFIG_ROOT` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_CPU_KVCACHE_SPACE` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_CPU_NUM_OF_RESERVED_CPU` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_CPU_OMP_THREADS_BIND` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_CPU_SGL_KERNEL` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_CUDART_SO_PATH` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_CUSTOM_SCOPES_FOR_PROFILING` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_DBO_COMM_SMS` | `throughput_tuning` | `multimodal` | Tunes scheduler and batching for higher throughput. | `--async-scheduling`, `--max-num-batched-tokens`, `--max-num-seqs` |
| `VLLM_DEBUG_DUMP_PATH` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `VLLM_DEBUG_LOG_API_SERVER_RESPONSE` | `network_serving` | `logging_debug` | Controls API host/port/endpoints and serving interface. | `--host`, `--port`, `--served-model-name` |
| `VLLM_DEBUG_MFU_METRICS` | `logging_debug` | `profiling_observability` | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `VLLM_DEBUG_WORKSPACE` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `VLLM_DEEPEPLL_NVFP4_DISPATCH` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_DEEPEP_BUFFER_SIZE_MB` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_DEEP_GEMM_WARMUP` | `multimodal` | - | Controls multimodal I/O paths and media preprocessing. | `--limit-mm-per-prompt`, `--mm-processor-cache-gb`, `--allowed-local-media-path` |
| `VLLM_DISABLED_KERNELS` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_DISABLE_COMPILE_CACHE` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_DISABLE_LOG_LOGO` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `VLLM_DISABLE_PYNCCL` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_DISABLE_SHARED_EXPERTS_STREAM` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_DOCKER_BUILD_CONTEXT` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_DO_NOT_TRACK` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_DP_MASTER_IP` | `data_parallel` | - | Replicates workers for throughput and multi-node serving. | `--data-parallel-size`, `--data-parallel-address`, `--data-parallel-rpc-port` |
| `VLLM_DP_MASTER_PORT` | `data_parallel` | `network_serving` | Replicates workers for throughput and multi-node serving. | `--data-parallel-size`, `--data-parallel-address`, `--data-parallel-rpc-port` |
| `VLLM_DP_RANK` | `data_parallel` | - | Replicates workers for throughput and multi-node serving. | `--data-parallel-size`, `--data-parallel-address`, `--data-parallel-rpc-port` |
| `VLLM_DP_RANK_LOCAL` | `data_parallel` | - | Replicates workers for throughput and multi-node serving. | `--data-parallel-size`, `--data-parallel-address`, `--data-parallel-rpc-port` |
| `VLLM_DP_SIZE` | `data_parallel` | - | Replicates workers for throughput and multi-node serving. | `--data-parallel-size`, `--data-parallel-address`, `--data-parallel-rpc-port` |
| `VLLM_ENABLE_CUDAGRAPH_GC` | `graph_mode` | - | Controls graph/eager execution and compile behavior. | `--compilation-config`, `--enforce-eager`, `--max-num-batched-tokens` |
| `VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING` | `expert_parallel` | - | Enables MoE expert routing parallelism; only valid on MoE models. | `--enable-expert-parallel`, `--tensor-parallel-size`, `--data-parallel-size` |
| `VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_ENABLE_MOE_DP_CHUNK` | `data_parallel` | `expert_parallel` | Replicates workers for throughput and multi-node serving. | `--data-parallel-size`, `--data-parallel-address`, `--data-parallel-rpc-port` |
| `VLLM_ENABLE_RESPONSES_API_STORE` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_ENGINE_ITERATION_TIMEOUT_S` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_ENGINE_READY_TIMEOUT_S` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_FLASHINFER_MOE_BACKEND` | `expert_parallel` | - | Enables MoE expert routing parallelism; only valid on MoE models. | `--enable-expert-parallel`, `--tensor-parallel-size`, `--data-parallel-size` |
| `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_FLOAT32_MATMUL_PRECISION` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_FORCE_AOT_LOAD` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_FUSED_MOE_CHUNK_SIZE` | `expert_parallel` | - | Enables MoE expert routing parallelism; only valid on MoE models. | `--enable-expert-parallel`, `--tensor-parallel-size`, `--data-parallel-size` |
| `VLLM_GC_DEBUG` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_HAS_FLASHINFER_CUBIN` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_HOST_IP` | `network_serving` | - | Controls API host/port/endpoints and serving interface. | `--host`, `--port`, `--served-model-name` |
| `VLLM_HTTP_TIMEOUT_KEEP_ALIVE` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_IMAGE_FETCH_TIMEOUT` | `multimodal` | - | Controls multimodal I/O paths and media preprocessing. | `--limit-mm-per-prompt`, `--mm-processor-cache-gb`, `--allowed-local-media-path` |
| `VLLM_KEEP_ALIVE_ON_ENGINE_DEATH` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_KV_CACHE_LAYOUT` | `memory_tuning` | - | Bounds memory pressure and sequence length behavior. | `--gpu-memory-utilization`, `--max-model-len`, `--block-size` |
| `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_LOGGING_COLOR` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `VLLM_LOGGING_CONFIG_PATH` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `VLLM_LOGGING_LEVEL` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `VLLM_LOGGING_PREFIX` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `VLLM_LOGGING_STREAM` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `VLLM_LOG_BATCHSIZE_INTERVAL` | `throughput_tuning` | `logging_debug` | Tunes scheduler and batching for higher throughput. | `--async-scheduling`, `--max-num-batched-tokens`, `--max-num-seqs` |
| `VLLM_LOG_MODEL_INSPECTION` | `logging_debug` | `model_selection` | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `VLLM_LOG_STATS_INTERVAL` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `VLLM_LOOPBACK_IP` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_LORA_DISABLE_PDL` | `lora` | - | Enables adapter loading and runtime LoRA routing. | `--enable-lora`, `--lora-modules`, `--max-loras` |
| `VLLM_LORA_RESOLVER_CACHE_DIR` | `lora` | - | Enables adapter loading and runtime LoRA routing. | `--enable-lora`, `--lora-modules`, `--max-loras` |
| `VLLM_LORA_RESOLVER_HF_REPO_LIST` | `lora` | `model_selection` | Enables adapter loading and runtime LoRA routing. | `--enable-lora`, `--lora-modules`, `--max-loras` |
| `VLLM_MAIN_CUDA_VERSION` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_MARLIN_INPUT_DTYPE` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `VLLM_MARLIN_USE_ATOMIC_ADD` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` | `multimodal` | - | Controls multimodal I/O paths and media preprocessing. | `--limit-mm-per-prompt`, `--mm-processor-cache-gb`, `--allowed-local-media-path` |
| `VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE` | `expert_parallel` | - | Enables MoE expert routing parallelism; only valid on MoE models. | `--enable-expert-parallel`, `--tensor-parallel-size`, `--data-parallel-size` |
| `VLLM_MEDIA_CONNECTOR` | `prefill_decode_disaggregation` | `multimodal` | Separates prefill/decode services or connectors. | `--kv-transfer-config`, `--data-parallel-size`, `--data-parallel-address` |
| `VLLM_MEDIA_LOADING_THREAD_COUNT` | `multimodal` | - | Controls multimodal I/O paths and media preprocessing. | `--limit-mm-per-prompt`, `--mm-processor-cache-gb`, `--allowed-local-media-path` |
| `VLLM_MEDIA_URL_ALLOW_REDIRECTS` | `multimodal` | - | Controls multimodal I/O paths and media preprocessing. | `--limit-mm-per-prompt`, `--mm-processor-cache-gb`, `--allowed-local-media-path` |
| `VLLM_MLA_DISABLE` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_MM_HASHER_ALGORITHM` | `multimodal` | - | Controls multimodal I/O paths and media preprocessing. | `--limit-mm-per-prompt`, `--mm-processor-cache-gb`, `--allowed-local-media-path` |
| `VLLM_MODEL_REDIRECT_PATH` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `VLLM_MOE_DP_CHUNK_SIZE` | `data_parallel` | `expert_parallel` | Replicates workers for throughput and multi-node serving. | `--data-parallel-size`, `--data-parallel-address`, `--data-parallel-rpc-port` |
| `VLLM_MOE_ROUTING_SIMULATION_STRATEGY` | `expert_parallel` | - | Enables MoE expert routing parallelism; only valid on MoE models. | `--enable-expert-parallel`, `--tensor-parallel-size`, `--data-parallel-size` |
| `VLLM_MOE_USE_DEEP_GEMM` | `expert_parallel` | - | Enables MoE expert routing parallelism; only valid on MoE models. | `--enable-expert-parallel`, `--tensor-parallel-size`, `--data-parallel-size` |
| `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_MOONCAKE_BOOTSTRAP_PORT` | `network_serving` | - | Controls API host/port/endpoints and serving interface. | `--host`, `--port`, `--served-model-name` |
| `VLLM_MORIIO_CONNECTOR_READ_MODE` | `prefill_decode_disaggregation` | - | Separates prefill/decode services or connectors. | `--kv-transfer-config`, `--data-parallel-size`, `--data-parallel-address` |
| `VLLM_MORIIO_NUM_WORKERS` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_MORIIO_POST_BATCH_SIZE` | `throughput_tuning` | - | Tunes scheduler and batching for higher throughput. | `--async-scheduling`, `--max-num-batched-tokens`, `--max-num-seqs` |
| `VLLM_MORIIO_QP_PER_TRANSFER` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_MQ_MAX_CHUNK_BYTES_MB` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_MSGPACK_ZERO_COPY_THRESHOLD` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_MXFP4_USE_MARLIN` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_NCCL_INCLUDE_PATH` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_NCCL_SO_PATH` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_NIXL_ABORT_REQUEST_TIMEOUT` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_NIXL_SIDE_CHANNEL_HOST` | `network_serving` | - | Controls API host/port/endpoints and serving interface. | `--host`, `--port`, `--served-model-name` |
| `VLLM_NIXL_SIDE_CHANNEL_PORT` | `network_serving` | - | Controls API host/port/endpoints and serving interface. | `--host`, `--port`, `--served-model-name` |
| `VLLM_NO_USAGE_STATS` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_NVFP4_GEMM_BACKEND` | `multimodal` | - | Controls multimodal I/O paths and media preprocessing. | `--limit-mm-per-prompt`, `--mm-processor-cache-gb`, `--allowed-local-media-path` |
| `VLLM_NVTX_SCOPES_FOR_PROFILING` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_PATTERN_MATCH_DEBUG` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `VLLM_PLUGINS` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_PORT` | `network_serving` | - | Controls API host/port/endpoints and serving interface. | `--host`, `--port`, `--served-model-name` |
| `VLLM_PP_LAYER_PARTITION` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_PROCESS_NAME_PREFIX` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_RANDOMIZE_DP_DUMMY_INPUTS` | `data_parallel` | - | Replicates workers for throughput and multi-node serving. | `--data-parallel-size`, `--data-parallel-address`, `--data-parallel-rpc-port` |
| `VLLM_RAY_BUNDLE_INDICES` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_RAY_DP_PACK_STRATEGY` | `data_parallel` | - | Replicates workers for throughput and multi-node serving. | `--data-parallel-size`, `--data-parallel-address`, `--data-parallel-rpc-port` |
| `VLLM_RAY_PER_WORKER_GPUS` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_RINGBUFFER_WARNING_INTERVAL` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_ROCM_CUSTOM_PAGED_ATTN` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_ROCM_FP8_MFMA_PAGE_ATTN` | `quantization` | - | Controls model precision and quantized weight loading path. | `--model`, `--dtype`, `--tensor-parallel-size` |
| `VLLM_ROCM_FP8_PADDING` | `quantization` | - | Controls model precision and quantized weight loading path. | `--model`, `--dtype`, `--tensor-parallel-size` |
| `VLLM_ROCM_MOE_PADDING` | `expert_parallel` | - | Enables MoE expert routing parallelism; only valid on MoE models. | `--enable-expert-parallel`, `--tensor-parallel-size`, `--data-parallel-size` |
| `VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION` | `quantization` | - | Controls model precision and quantized weight loading path. | `--model`, `--dtype`, `--tensor-parallel-size` |
| `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT` | `memory_tuning` | - | Bounds memory pressure and sequence length behavior. | `--gpu-memory-utilization`, `--max-model-len`, `--block-size` |
| `VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_ROCM_USE_AITER` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_ROCM_USE_AITER_FP4BMM` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_ROCM_USE_AITER_FP8BMM` | `quantization` | - | Controls model precision and quantized weight loading path. | `--model`, `--dtype`, `--tensor-parallel-size` |
| `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_ROCM_USE_AITER_LINEAR` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_ROCM_USE_AITER_MHA` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_ROCM_USE_AITER_MLA` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_ROCM_USE_AITER_MOE` | `expert_parallel` | - | Enables MoE expert routing parallelism; only valid on MoE models. | `--enable-expert-parallel`, `--tensor-parallel-size`, `--data-parallel-size` |
| `VLLM_ROCM_USE_AITER_PAGED_ATTN` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_ROCM_USE_AITER_RMSNORM` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_ROCM_USE_AITER_TRITON_GEMM` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_ROCM_USE_AITER_TRITON_ROPE` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_ROCM_USE_SKINNY_GEMM` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_RPC_BASE_PATH` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_RPC_TIMEOUT` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_SERVER_DEV_MODE` | `network_serving` | - | Controls API host/port/endpoints and serving interface. | `--host`, `--port`, `--served-model-name` |
| `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_SKIP_P2P_CHECK` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_SLEEP_WHEN_IDLE` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_TARGET_DEVICE` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_TEST_FORCE_FP8_MARLIN` | `quantization` | - | Controls model precision and quantized weight loading path. | `--model`, `--dtype`, `--tensor-parallel-size` |
| `VLLM_TEST_FORCE_LOAD_FORMAT` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_TOOL_JSON_ERROR_AUTOMATIC_RETRY` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_TPU_BUCKET_PADDING_GAP` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_TPU_MOST_MODEL_LEN` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `VLLM_TPU_USING_PATHWAYS` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_TRACE_FUNCTION` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `VLLM_TUNED_CONFIG_FOLDER` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_USAGE_SOURCE` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_USAGE_STATS_SERVER` | `network_serving` | - | Controls API host/port/endpoints and serving interface. | `--host`, `--port`, `--served-model-name` |
| `VLLM_USE_AOT_COMPILE` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_USE_BYTECODE_HOOK` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_USE_DEEP_GEMM` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_USE_DEEP_GEMM_E8M0` | `multimodal` | - | Controls multimodal I/O paths and media preprocessing. | `--limit-mm-per-prompt`, `--mm-processor-cache-gb`, `--allowed-local-media-path` |
| `VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES` | `multimodal` | - | Controls multimodal I/O paths and media preprocessing. | `--limit-mm-per-prompt`, `--mm-processor-cache-gb`, `--allowed-local-media-path` |
| `VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_USE_FBGEMM` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_USE_FLASHINFER_MOE_FP16` | `expert_parallel` | - | Enables MoE expert routing parallelism; only valid on MoE models. | `--enable-expert-parallel`, `--tensor-parallel-size`, `--data-parallel-size` |
| `VLLM_USE_FLASHINFER_MOE_FP4` | `expert_parallel` | - | Enables MoE expert routing parallelism; only valid on MoE models. | `--enable-expert-parallel`, `--tensor-parallel-size`, `--data-parallel-size` |
| `VLLM_USE_FLASHINFER_MOE_FP8` | `quantization` | `expert_parallel` | Controls model precision and quantized weight loading path. | `--model`, `--dtype`, `--tensor-parallel-size` |
| `VLLM_USE_FLASHINFER_MOE_INT4` | `quantization` | `expert_parallel` | Controls model precision and quantized weight loading path. | `--model`, `--dtype`, `--tensor-parallel-size` |
| `VLLM_USE_FLASHINFER_MOE_MXFP4_BF16` | `expert_parallel` | - | Enables MoE expert routing parallelism; only valid on MoE models. | `--enable-expert-parallel`, `--tensor-parallel-size`, `--data-parallel-size` |
| `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8` | `quantization` | `expert_parallel` | Controls model precision and quantized weight loading path. | `--model`, `--dtype`, `--tensor-parallel-size` |
| `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS` | `quantization` | `expert_parallel` | Controls model precision and quantized weight loading path. | `--model`, `--dtype`, `--tensor-parallel-size` |
| `VLLM_USE_FLASHINFER_SAMPLER` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_USE_FUSED_MOE_GROUPED_TOPK` | `expert_parallel` | - | Enables MoE expert routing parallelism; only valid on MoE models. | `--enable-expert-parallel`, `--tensor-parallel-size`, `--data-parallel-size` |
| `VLLM_USE_MEGA_AOT_ARTIFACT` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_USE_MODELSCOPE` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `VLLM_USE_NCCL_SYMM_MEM` | `multimodal` | - | Controls multimodal I/O paths and media preprocessing. | `--limit-mm-per-prompt`, `--mm-processor-cache-gb`, `--allowed-local-media-path` |
| `VLLM_USE_NVFP4_CT_EMULATIONS` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_USE_PRECOMPILED` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_USE_RAY_WRAPPED_PP_COMM` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_USE_STANDALONE_COMPILE` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_USE_TRITON_AWQ` | `quantization` | - | Controls model precision and quantized weight loading path. | `--model`, `--dtype`, `--tensor-parallel-size` |
| `VLLM_USE_V2_MODEL_RUNNER` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_V1_USE_OUTLINES_CACHE` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_VIDEO_FETCH_TIMEOUT` | `multimodal` | - | Controls multimodal I/O paths and media preprocessing. | `--limit-mm-per-prompt`, `--mm-processor-cache-gb`, `--allowed-local-media-path` |
| `VLLM_VIDEO_LOADER_BACKEND` | `multimodal` | - | Controls multimodal I/O paths and media preprocessing. | `--limit-mm-per-prompt`, `--mm-processor-cache-gb`, `--allowed-local-media-path` |
| `VLLM_WORKER_MULTIPROC_METHOD` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_XGRAMMAR_CACHE_MB` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_XLA_CACHE_PATH` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_XLA_CHECK_RECOMPILATION` | `graph_mode` | - | Controls graph/eager execution and compile behavior. | `--compilation-config`, `--enforce-eager`, `--max-num-batched-tokens` |
| `VLLM_XLA_USE_SPMD` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `V_SCALE_CONSTANT` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |

## vLLM-Ascend Args -> Semantics

| Parameter | Primary feature | Secondary features | Usage | Common combinations |
| --- | --- | --- | --- | --- |
| `--20250429` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--additional-config` |
| `--additional-config` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--address` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--device` |
| `--allowed-local-media-path` | `network_serving` | `multimodal` | Controls API host/port/endpoints and serving interface. | `--host`, `--port`, `--served-model-name` |
| `--api-key` | `security_auth` | - | Controls authentication, TLS, and request trust boundaries. | `--ssl-certfile`, `--allowed-origins` |
| `--api-server-count` | `network_serving` | - | Controls API host/port/endpoints and serving interface. | `--data-parallel-rpc-port` |
| `--api-url` | `network_serving` | - | Controls API host/port/endpoints and serving interface. | `--host`, `--port`, `--served-model-name` |
| `--async-scheduling` | `throughput_tuning` | - | Tunes scheduler and batching for higher throughput. | `--max-num-batched-tokens` |
| `--audio-path1` | `multimodal` | - | Controls multimodal I/O paths and media preprocessing. | `--audio-path2` |
| `--audio-path2` | `multimodal` | - | Controls multimodal I/O paths and media preprocessing. | `--audio-path1` |
| `--backend` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--dataset-name` |
| `--block-size` | `memory_tuning` | - | Bounds memory pressure and sequence length behavior. | `--gpu-memory-utilization`, `--max-model-len` |
| `--bs` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--dcp`, `--pcp` |
| `--chat-template` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--backend`, `--dataset-name` |
| `--chat-template-content-format` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--dataset-name`, `--device` |
| `--compilation-config` | `graph_mode` | - | Controls graph/eager execution and compile behavior. | `--enforce-eager`, `--max-num-batched-tokens` |
| `--compress-process-num` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--enable-compress`, `--output` |
| `--cp-kv-cache-interleave-size` | `context_parallel` | `memory_tuning` | Splits long-context KV processing across ranks. | `--decode-context-parallel-size`, `--prefill-context-parallel-size` |
| `--data` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--engine` |
| `--data-parallel-address` | `data_parallel` | - | Replicates workers for throughput and multi-node serving. | `--data-parallel-rpc-port`, `--data-parallel-size` |
| `--data-parallel-rank` | `data_parallel` | - | Replicates workers for throughput and multi-node serving. | `--data-parallel-size` |
| `--data-parallel-rpc-port` | `data_parallel` | `network_serving` | Replicates workers for throughput and multi-node serving. | `--data-parallel-size` |
| `--data-parallel-size` | `data_parallel` | - | Replicates workers for throughput and multi-node serving. | `--data-parallel-address`, `--data-parallel-rpc-port` |
| `--data-parallel-size-local` | `data_parallel` | - | Replicates workers for throughput and multi-node serving. | `--data-parallel-size` |
| `--data-parallel-start-rank` | `data_parallel` | - | Replicates workers for throughput and multi-node serving. | `--data-parallel-address`, `--data-parallel-rpc-port`, `--data-parallel-size` |
| `--dataset-args` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--dataset-name` |
| `--dataset-name` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--random-input`, `--result-dir` |
| `--datasets` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--dcp` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--bs`, `--pcp` |
| `--debug` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `--decode-context-parallel-size` | `context_parallel` | - | Splits long-context KV processing across ranks. | `--prefill-context-parallel-size` |
| `--decode-servers-urls` | `prefill_decode_disaggregation` | `network_serving` | Separates prefill/decode services or connectors. | `--kv-transfer-config`, `--data-parallel-size`, `--data-parallel-address` |
| `--decoder-hosts` | `prefill_decode_disaggregation` | `network_serving` | Separates prefill/decode services or connectors. | `--kv-transfer-config`, `--data-parallel-size`, `--data-parallel-address` |
| `--decoder-hosts-num` | `prefill_decode_disaggregation` | `network_serving` | Separates prefill/decode services or connectors. | `--kv-transfer-config`, `--data-parallel-size`, `--data-parallel-address` |
| `--decoder-ports` | `network_serving` | - | Controls API host/port/endpoints and serving interface. | `--decoder-hosts`, `--host`, `--port` |
| `--decoder-ports-inc` | `network_serving` | - | Controls API host/port/endpoints and serving interface. | `--host`, `--port`, `--served-model-name` |
| `--depth` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--device` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--name`, `--rm`, `--shm-size` |
| `--disable-log-request` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `--disable-log-stats` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--max-log-len`, `--log-config-file` |
| `--distributed-executor-backend` | `data_parallel` | - | Replicates workers for throughput and multi-node serving. | `--data-parallel-size`, `--data-parallel-address`, `--data-parallel-rpc-port` |
| `--dp-address` | `data_parallel` | - | Replicates workers for throughput and multi-node serving. | `--dp-rank-start`, `--dp-rpc-port`, `--dp-size` |
| `--dp-hosts` | `data_parallel` | `network_serving` | Replicates workers for throughput and multi-node serving. | `--data-parallel-rank`, `--dp-ports` |
| `--dp-ports` | `data_parallel` | `network_serving` | Replicates workers for throughput and multi-node serving. | `--data-parallel-rank`, `--dp-hosts` |
| `--dp-rank-start` | `data_parallel` | - | Replicates workers for throughput and multi-node serving. | `--dp-address`, `--dp-rpc-port`, `--dp-size` |
| `--dp-rpc-port` | `data_parallel` | `network_serving` | Replicates workers for throughput and multi-node serving. | `--dp-address`, `--dp-rank-start`, `--dp-size` |
| `--dp-size` | `data_parallel` | - | Replicates workers for throughput and multi-node serving. | `--dp-address`, `--dp-rank-start` |
| `--dp-size-local` | `data_parallel` | - | Replicates workers for throughput and multi-node serving. | `--dp-address`, `--dp-rank-start`, `--dp-rpc-port` |
| `--dtype` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `--ec-transfer-config` | `prefill_decode_disaggregation` | - | Separates prefill/decode services or connectors. | `--decode-servers-urls` |
| `--enable-chunked-prefill` | `prefill_decode_disaggregation` | `throughput_tuning` | Separates prefill/decode services or connectors. | `--kv-transfer-config`, `--data-parallel-size`, `--data-parallel-address` |
| `--enable-compress` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--compress-process-num`, `--output` |
| `--enable-expert-parallel` | `expert_parallel` | - | Enables MoE expert routing parallelism; only valid on MoE models. | `--tensor-parallel-size`, `--data-parallel-size` |
| `--enable-lora` | `lora` | - | Enables adapter loading and runtime LoRA routing. | `--lora-modules` |
| `--enable-prefix-caching` | `prefix_cache` | - | Reuses shared prompt prefixes to reduce prefill cost. | `--prefix-caching-hash-algo`, `--max-model-len` |
| `--enable-request-id-headers` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--backend`, `--dataset-name` |
| `--enable-sleep-mode` | `sleep_mode` | - | Enables idle-time memory/power saving mode. | `--gpu-memory-utilization`, `--max-model-len` |
| `--encode-servers-urls` | `network_serving` | - | Controls API host/port/endpoints and serving interface. | `--decode-servers-urls`, `--host`, `--port` |
| `--encoder-dispatch-mode` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--endpoint` | `network_serving` | - | Controls API host/port/endpoints and serving interface. | `--host`, `--port`, `--served-model-name` |
| `--enforce-eager` | `graph_mode` | - | Controls graph/eager execution and compile behavior. | `--compilation-config`, `--max-num-batched-tokens` |
| `--engine` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--data` |
| `--engine-base-url` | `network_serving` | - | Controls API host/port/endpoints and serving interface. | `--host`, `--port`, `--served-model-name` |
| `--eval-batch-size` | `throughput_tuning` | - | Tunes scheduler and batching for higher throughput. | `--async-scheduling`, `--max-num-batched-tokens`, `--max-num-seqs` |
| `--eval-type` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--dataset-args` |
| `--extra-index-url` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--chat-template-content-format`, `--dataset-name` |
| `--generation-config` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--dataset-args` |
| `--gpu-memory-utilization` | `memory_tuning` | - | Bounds memory pressure and sequence length behavior. | `--max-model-len` |
| `--head` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--address`, `--device` |
| `--header` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--data`, `--engine` |
| `--headless` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--hf-overrides` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `--host` | `network_serving` | - | Controls API host/port/endpoints and serving interface. | `--port` |
| `--ignore-eos` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--dataset-name` |
| `--init` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--device` |
| `--kv-transfer-config` | `prefill_decode_disaggregation` | - | Separates prefill/decode services or connectors. | `--data-parallel-size`, `--data-parallel-address` |
| `--limit` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--dataset-args` |
| `--limit-mm-per-prompt` | `multimodal` | - | Controls multimodal I/O paths and media preprocessing. | `--allowed-local-media-path` |
| `--load-format` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--location` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--data`, `--engine` |
| `--lora-modules` | `lora` | - | Enables adapter loading and runtime LoRA routing. | `--enable-lora` |
| `--master-addr` | `data_parallel` | - | Replicates workers for throughput and multi-node serving. | `--master-port` |
| `--master-port` | `data_parallel` | `network_serving` | Replicates workers for throughput and multi-node serving. | `--master-addr` |
| `--max` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--max-concurrency` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--max-model-len` | `memory_tuning` | `model_selection` | Bounds memory pressure and sequence length behavior. | `--gpu-memory-utilization`, `--block-size` |
| `--max-num-batched-tokens` | `throughput_tuning` | - | Tunes scheduler and batching for higher throughput. | `--async-scheduling`, `--max-num-seqs` |
| `--max-num-seqs` | `throughput_tuning` | - | Tunes scheduler and batching for higher throughput. | `--async-scheduling`, `--max-num-batched-tokens` |
| `--max-retries` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--retry-delay` |
| `--max-waiting-retries` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--metric-percentiles` | `profiling_observability` | - | Controls profiling, traces, and metrics visibility. | `--profiler-config`, `--collect-detailed-traces`, `--otlp-traces-endpoint` |
| `--mm-processor-cache-gb` | `memory_tuning` | `multimodal` | Bounds memory pressure and sequence length behavior. | `--gpu-memory-utilization`, `--max-model-len`, `--block-size` |
| `--mode` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--datasets` |
| `--model` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--tokenizer`, `--revision`, `--trust-remote-code` |
| `--model-loader-extra-config` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `--model-weight-gib` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `--models` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `--name` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--device`, `--rm`, `--shm-size` |
| `--net` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--device`, `--name`, `--rm` |
| `--network` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--device` |
| `--nnodes` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--no-enable-chunked-prefill` | `prefill_decode_disaggregation` | `throughput_tuning` | Separates prefill/decode services or connectors. | `--kv-transfer-config`, `--data-parallel-size`, `--data-parallel-address` |
| `--no-enable-prefix-caching` | `prefix_cache` | - | Reuses shared prompt prefixes to reduce prefill cost. | `--enable-prefix-caching`, `--prefix-caching-hash-algo`, `--max-model-len` |
| `--node-ip-address` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--address`, `--device` |
| `--node-rank` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--node-size` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--num-prompts` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--dataset-name` |
| `--output` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--compress-process-num` |
| `--output-dir` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--data`, `--engine` |
| `--pcp` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--bs`, `--dcp` |
| `--percentile-metrics` | `profiling_observability` | - | Controls profiling, traces, and metrics visibility. | `--profiler-config`, `--collect-detailed-traces`, `--otlp-traces-endpoint` |
| `--pipeline-parallel-size` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--address`, `--device` |
| `--pod` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--data`, `--engine` |
| `--port` | `network_serving` | - | Controls API host/port/endpoints and serving interface. | `--host`, `--served-model-name` |
| `--prefill-context-parallel-size` | `context_parallel` | `prefill_decode_disaggregation` | Splits long-context KV processing across ranks. | `--decode-context-parallel-size`, `--cp-kv-cache-interleave-size` |
| `--prefill-servers-urls` | `prefill_decode_disaggregation` | `network_serving` | Separates prefill/decode services or connectors. | `--decode-servers-urls` |
| `--prefiller-hosts` | `prefill_decode_disaggregation` | `network_serving` | Separates prefill/decode services or connectors. | `--decoder-hosts` |
| `--prefiller-hosts-num` | `prefill_decode_disaggregation` | `network_serving` | Separates prefill/decode services or connectors. | `--kv-transfer-config`, `--data-parallel-size`, `--data-parallel-address` |
| `--prefiller-port` | `prefill_decode_disaggregation` | `network_serving` | Separates prefill/decode services or connectors. | `--decoder-hosts` |
| `--prefiller-ports` | `prefill_decode_disaggregation` | `network_serving` | Separates prefill/decode services or connectors. | `--decoder-hosts` |
| `--prefiller-ports-inc` | `prefill_decode_disaggregation` | `network_serving` | Separates prefill/decode services or connectors. | `--kv-transfer-config`, `--data-parallel-size`, `--data-parallel-address` |
| `--prefix-repetition-num-prefixes` | `prefix_cache` | - | Reuses shared prompt prefixes to reduce prefill cost. | `--enable-prefix-caching`, `--prefix-caching-hash-algo`, `--max-model-len` |
| `--prefix-repetition-output-len` | `prefix_cache` | - | Reuses shared prompt prefixes to reduce prefill cost. | `--enable-prefix-caching`, `--prefix-caching-hash-algo`, `--max-model-len` |
| `--prefix-repetition-prefix-len` | `prefix_cache` | - | Reuses shared prompt prefixes to reduce prefill cost. | `--enable-prefix-caching`, `--prefix-caching-hash-algo`, `--max-model-len` |
| `--prefix-repetition-suffix-len` | `prefix_cache` | - | Reuses shared prompt prefixes to reduce prefill cost. | `--enable-prefix-caching`, `--prefix-caching-hash-algo`, `--max-model-len` |
| `--privileged` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--device` |
| `--proc-per-node` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--profiler-config` | `profiling_observability` | - | Controls profiling, traces, and metrics visibility. | `--collect-detailed-traces`, `--otlp-traces-endpoint` |
| `--quantization` | `quantization` | - | Controls model precision and quantized weight loading path. | `--model`, `--dtype`, `--tensor-parallel-size` |
| `--random-input` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--dataset-name`, `--result-dir` |
| `--random-input-len` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--dataset-name` |
| `--random-output-len` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--dataset-name` |
| `--reasoning-parser` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--additional-config` |
| `--recursive` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--device` |
| `--request-rate` | `throughput_tuning` | - | Tunes scheduler and batching for higher throughput. | `--async-scheduling`, `--max-num-batched-tokens`, `--max-num-seqs` |
| `--result-dir` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--dataset-name`, `--random-input` |
| `--retry-delay` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--max-retries` |
| `--rm` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--device`, `--name`, `--shm-size` |
| `--runner` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `--save-result` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--dataset-name`, `--random-input` |
| `--seed` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--served-model-name` | `network_serving` | `model_selection` | Controls API host/port/endpoints and serving interface. | `--port` |
| `--shm-size` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--device`, `--name`, `--rm` |
| `--sleep-mode-level` | `sleep_mode` | - | Enables idle-time memory/power saving mode. | `--enable-sleep-mode` |
| `--source` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--data`, `--engine` |
| `--speculative-config` | `speculative_decode` | - | Enables draft/speculative decoding acceleration path. | `--max-num-batched-tokens`, `--async-scheduling` |
| `--summarizer` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--datasets`, `--device` |
| `--swap-space` | `memory_tuning` | - | Bounds memory pressure and sequence length behavior. | `--gpu-memory-utilization`, `--max-model-len`, `--block-size` |
| `--task` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `--tasks` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `--temperature` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `--tensor-parallel-size` | `tensor_parallel` | - | Splits model tensors across NPUs/GPUs for scale-out inference. | `--data-parallel-size`, `--distributed-executor-backend` |
| `--tokenizer` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model` |
| `--tokenizer-mode` | `model_selection` | - | Selects model/tokenizer/artifact and runner mode. | `--model`, `--tokenizer`, `--revision` |
| `--tp` | `tensor_parallel` | - | Splits model tensors across NPUs/GPUs for scale-out inference. | `--tensor-parallel-size`, `--data-parallel-size`, `--distributed-executor-backend` |
| `--tp-size` | `tensor_parallel` | - | Splits model tensors across NPUs/GPUs for scale-out inference. | `--tensor-parallel-size`, `--data-parallel-size`, `--distributed-executor-backend` |
| `--trust-remote-code` | `security_auth` | - | Controls authentication, TLS, and request trust boundaries. | `--api-key`, `--ssl-certfile`, `--allowed-origins` |
| `--vllm-start-port` | `network_serving` | - | Controls API host/port/endpoints and serving interface. | `--dp-rpc-port` |
| `--waiting-retry-interval` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |

## vLLM-Ascend Env Vars -> Semantics

| Variable | Primary feature | Secondary features | Usage | Common combinations |
| --- | --- | --- | --- | --- |
| `ASCEND_HOME_PATH` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `CMAKE_BUILD_TYPE` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `COMPILE_CUSTOM_KERNELS` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `CXX_COMPILER` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `C_COMPILER` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `DYNAMIC_EPLB` | `expert_parallel` | - | Enables MoE expert routing parallelism; only valid on MoE models. | `--enable-expert-parallel`, `--tensor-parallel-size`, `--data-parallel-size` |
| `HCCL_SO_PATH` | `data_parallel` | - | Replicates workers for throughput and multi-node serving. | `--data-parallel-size`, `--data-parallel-address`, `--data-parallel-rpc-port` |
| `MAX_JOBS` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `MSMONITOR_USE_DAEMON` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `SOC_VERSION` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VERBOSE` | `logging_debug` | - | Controls logs, debug verbosity, and troubleshooting signal. | `--disable-log-stats`, `--max-log-len`, `--log-config-file` |
| `VLLM_ASCEND_BALANCE_SCHEDULING` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL` | `context_parallel` | - | Splits long-context KV processing across ranks. | `--prefill-context-parallel-size`, `--decode-context-parallel-size`, `--max-model-len` |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | `throughput_tuning` | - | Tunes scheduler and batching for higher throughput. | `--async-scheduling`, `--max-num-batched-tokens`, `--max-num-seqs` |
| `VLLM_ASCEND_ENABLE_FUSED_MC2` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE` | `tensor_parallel` | - | Splits model tensors across NPUs/GPUs for scale-out inference. | `--tensor-parallel-size`, `--data-parallel-size`, `--distributed-executor-backend` |
| `VLLM_ASCEND_ENABLE_MLAPO` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_ASCEND_ENABLE_NZ` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |
| `VLLM_ASCEND_ENABLE_PREFETCH_MLP` | `weight_prefetch` | - | Warms model weight blocks before decode to reduce stalls. | `--additional-config`, `--max-num-batched-tokens`, `--gpu-memory-utilization` |
| `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE` | `throughput_tuning` | - | Tunes scheduler and batching for higher throughput. | `--async-scheduling`, `--max-num-batched-tokens`, `--max-num-seqs` |
| `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK` | `memory_tuning` | - | Bounds memory pressure and sequence length behavior. | `--gpu-memory-utilization`, `--max-model-len`, `--block-size` |
| `VLLM_ASCEND_MLP_DOWN_PREFETCH_SIZE` | `weight_prefetch` | - | Warms model weight blocks before decode to reduce stalls. | `--additional-config`, `--max-num-batched-tokens`, `--gpu-memory-utilization` |
| `VLLM_ASCEND_MLP_GATE_UP_PREFETCH_SIZE` | `weight_prefetch` | - | Warms model weight blocks before decode to reduce stalls. | `--additional-config`, `--max-num-batched-tokens`, `--gpu-memory-utilization` |
| `VLLM_VERSION` | `general_runtime` | - | General runtime behavior; review source and profile defaults. | `--model`, `--device`, `--dtype` |

Back to [INDEX](../../INDEX.md).
