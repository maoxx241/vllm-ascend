---
knowledge_id: vllm-ascend-core.inputs-and-envs-global
domain: vllm-ascend-core
knowledge_type: reference
summary: Global inventory of vLLM-Ascend environment variables and observed deployment arguments.
applicable_vllm_versions: [">=0.15.0", "<0.17.0"]
applicable_cann_versions: [">=8.0.0"]
last_verified: "2026-03-05"
watch_files:
  - "vllm_ascend/envs.py"
  - "docs/source/user_guide/feature_guide/index.md"
  - "docs/source/tutorials/models/index.md"
  - "examples/run_dp_server.sh"
depends_on:
  - "../../INDEX.md"
  - "references/repo-full-knowledge-map.md"
source_commit: "workspace-head"
freshness: "fresh"
---

# vLLM-Ascend Global Inputs and Envs

Generated at: `2026-03-05`

- vLLM-Ascend env vars discovered: **24**
- vLLM-Ascend args observed across docs/examples/tests: **159**

## vLLM-Ascend Environment Variables (inventory)

| Variable | Kind | Source preview |
| --- | --- | --- |
| `ASCEND_HOME_PATH` | vllm_ascend_env | vllm_ascend/envs.py:env_variables |
| `CMAKE_BUILD_TYPE` | vllm_ascend_env | vllm_ascend/envs.py:env_variables |
| `COMPILE_CUSTOM_KERNELS` | vllm_ascend_env | vllm_ascend/envs.py:env_variables |
| `CXX_COMPILER` | vllm_ascend_env | vllm_ascend/envs.py:env_variables |
| `C_COMPILER` | vllm_ascend_env | vllm_ascend/envs.py:env_variables |
| `DYNAMIC_EPLB` | vllm_ascend_env | vllm_ascend/envs.py:env_variables |
| `HCCL_SO_PATH` | vllm_ascend_env | vllm_ascend/envs.py:env_variables |
| `MAX_JOBS` | vllm_ascend_env | vllm_ascend/envs.py:env_variables |
| `MSMONITOR_USE_DAEMON` | vllm_ascend_env | vllm_ascend/envs.py:env_variables |
| `SOC_VERSION` | vllm_ascend_env | vllm_ascend/envs.py:env_variables |
| `VERBOSE` | vllm_ascend_env | vllm_ascend/envs.py:env_variables |
| `VLLM_ASCEND_BALANCE_SCHEDULING` | vllm_ascend_env | vllm_ascend/envs.py:env_variables |
| `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL` | vllm_ascend_env | vllm_ascend/envs.py:env_variables |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | vllm_ascend_env | vllm_ascend/envs.py:env_variables |
| `VLLM_ASCEND_ENABLE_FUSED_MC2` | vllm_ascend_env | vllm_ascend/envs.py:env_variables |
| `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE` | vllm_ascend_env | vllm_ascend/envs.py:env_variables |
| `VLLM_ASCEND_ENABLE_MLAPO` | vllm_ascend_env | vllm_ascend/envs.py:env_variables |
| `VLLM_ASCEND_ENABLE_NZ` | vllm_ascend_env | vllm_ascend/envs.py:env_variables |
| `VLLM_ASCEND_ENABLE_PREFETCH_MLP` | vllm_ascend_env | vllm_ascend/envs.py:env_variables |
| `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE` | vllm_ascend_env | vllm_ascend/envs.py:env_variables |
| `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK` | vllm_ascend_env | vllm_ascend/envs.py:env_variables |
| `VLLM_ASCEND_MLP_DOWN_PREFETCH_SIZE` | vllm_ascend_env | vllm_ascend/envs.py:env_variables |
| `VLLM_ASCEND_MLP_GATE_UP_PREFETCH_SIZE` | vllm_ascend_env | vllm_ascend/envs.py:env_variables |
| `VLLM_VERSION` | vllm_ascend_env | vllm_ascend/envs.py:env_variables |

## vLLM-Ascend Arguments (observed inventory)

| Argument | Kind | Source preview |
| --- | --- | --- |
| `--20250429` | vllm_ascend_arg | docs/source/tutorials/models/Qwen3-Dense.md |
| `--additional-config` | vllm_ascend_arg | docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md, docs/source/tutorials/features/suffix_speculative_decoding.md, docs/source/tutorials/models/DeepSeek-V3.1.md (+32 more) |
| `--address` | vllm_ascend_arg | docs/source/tutorials/features/ray.md |
| `--allowed-local-media-path` | vllm_ascend_arg | docs/source/tutorials/models/Qwen2.5-Omni.md, docs/source/tutorials/models/Qwen3-VL-30B-A3B-Instruct.md |
| `--api-key` | vllm_ascend_arg | docs/source/tutorials/models/Qwen3-Omni-30B-A3B-Thinking.md |
| `--api-server-count` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/models/Qwen3-235B-A22B.md, docs/source/tutorials/models/Qwen3-VL-235B-A22B-Instruct.md (+1 more) |
| `--api-url` | vllm_ascend_arg | docs/source/tutorials/models/Qwen3-Omni-30B-A3B-Thinking.md |
| `--async-scheduling` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/tutorials/models/DeepSeek-R1.md, docs/source/tutorials/models/DeepSeek-V3.1.md (+20 more) |
| `--audio-path1` | vllm_ascend_arg | examples/offline_inference_audio_language.py |
| `--audio-path2` | vllm_ascend_arg | examples/offline_inference_audio_language.py |
| `--backend` | vllm_ascend_arg | docs/source/tutorials/models/GLM4.x.md, docs/source/tutorials/models/Qwen3-VL-Embedding.md, docs/source/tutorials/models/Qwen3-VL-Reranker.md (+4 more) |
| `--block-size` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/models/Qwen3-Dense.md, docs/source/user_guide/feature_guide/dynamic_batch.md (+7 more) |
| `--bs` | vllm_ascend_arg | examples/offline_inference_npu_long_seq.py |
| `--chat-template` | vllm_ascend_arg | docs/source/tutorials/models/Qwen3-VL-Reranker.md |
| `--chat-template-content-format` | vllm_ascend_arg | docs/source/tutorials/models/Qwen-VL-Dense.md |
| `--compilation-config` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md (+32 more) |
| `--compress-process-num` | vllm_ascend_arg | examples/save_sharded_state_310.py |
| `--cp-kv-cache-interleave-size` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/user_guide/feature_guide/context_parallel.md, tests/e2e/nightly/multi_node/config/DeepSeek-R1-W8A8-longseq.yaml (+1 more) |
| `--data` | vllm_ascend_arg | docs/source/user_guide/deployment_guide/using_volcano_kthena.md |
| `--data-parallel-address` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md, docs/source/tutorials/models/DeepSeek-R1.md (+20 more) |
| `--data-parallel-rank` | vllm_ascend_arg | docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md, docs/source/tutorials/models/DeepSeek-V3.1.md, docs/source/tutorials/models/DeepSeek-V3.2.md (+3 more) |
| `--data-parallel-rpc-port` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md, docs/source/tutorials/models/DeepSeek-R1.md (+21 more) |
| `--data-parallel-size` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md (+41 more) |
| `--data-parallel-size-local` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/models/DeepSeek-R1.md, docs/source/tutorials/models/DeepSeek-V3.1.md (+23 more) |
| `--data-parallel-start-rank` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/models/DeepSeek-R1.md, docs/source/tutorials/models/DeepSeek-V3.1.md (+16 more) |
| `--dataset-args` | vllm_ascend_arg | docs/source/tutorials/models/Qwen3-Omni-30B-A3B-Thinking.md |
| `--dataset-name` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/tutorials/models/DeepSeek-R1.md (+17 more) |
| `--datasets` | vllm_ascend_arg | docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md, docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md, docs/source/tutorials/features/suffix_speculative_decoding.md (+2 more) |
| `--dcp` | vllm_ascend_arg | examples/offline_inference_npu_long_seq.py |
| `--debug` | vllm_ascend_arg | docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md, docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md, docs/source/tutorials/features/suffix_speculative_decoding.md (+1 more) |
| `--decode-context-parallel-size` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/user_guide/feature_guide/context_parallel.md (+3 more) |
| `--decode-servers-urls` | vllm_ascend_arg | examples/disaggregated_encoder/disagg_1e1pd_example.sh, examples/disaggregated_encoder/disagg_epd_proxy.py, tests/e2e/nightly/single_node/models/configs/Qwen2.5-VL-7B-Instruct-EPD.yaml |
| `--decoder-hosts` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md, docs/source/tutorials/features/pd_disaggregation_mooncake_single_node.md (+7 more) |
| `--decoder-hosts-num` | vllm_ascend_arg | docs/source/user_guide/feature_guide/large_scale_ep.md |
| `--decoder-ports` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md, docs/source/tutorials/features/pd_disaggregation_mooncake_single_node.md (+7 more) |
| `--decoder-ports-inc` | vllm_ascend_arg | docs/source/user_guide/feature_guide/large_scale_ep.md |
| `--depth` | vllm_ascend_arg | docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md, docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md, docs/source/tutorials/features/pd_disaggregation_mooncake_single_node.md (+1 more) |
| `--device` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md (+25 more) |
| `--disable-log-request` | vllm_ascend_arg | docs/source/tutorials/models/Qwen3-Omni-30B-A3B-Thinking.md |
| `--disable-log-stats` | vllm_ascend_arg | docs/source/tutorials/models/Qwen3-Omni-30B-A3B-Thinking.md |
| `--distributed-executor-backend` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/ray.md, docs/source/tutorials/features/suffix_speculative_decoding.md (+1 more) |
| `--dp-address` | vllm_ascend_arg | docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md, docs/source/tutorials/models/DeepSeek-V3.1.md, docs/source/tutorials/models/DeepSeek-V3.2.md (+2 more) |
| `--dp-hosts` | vllm_ascend_arg | docs/source/user_guide/feature_guide/external_dp.md, examples/external_online_dp/dp_load_balance_proxy_server.py |
| `--dp-ports` | vllm_ascend_arg | docs/source/user_guide/feature_guide/external_dp.md, examples/external_online_dp/dp_load_balance_proxy_server.py |
| `--dp-rank-start` | vllm_ascend_arg | docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md, docs/source/tutorials/models/DeepSeek-V3.1.md, docs/source/tutorials/models/DeepSeek-V3.2.md (+2 more) |
| `--dp-rpc-port` | vllm_ascend_arg | docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md, docs/source/tutorials/models/DeepSeek-V3.1.md, docs/source/tutorials/models/DeepSeek-V3.2.md (+2 more) |
| `--dp-size` | vllm_ascend_arg | docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md, docs/source/tutorials/models/DeepSeek-V3.1.md, docs/source/tutorials/models/DeepSeek-V3.2.md (+3 more) |
| `--dp-size-local` | vllm_ascend_arg | docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md, docs/source/tutorials/models/DeepSeek-V3.1.md, docs/source/tutorials/models/DeepSeek-V3.2.md (+2 more) |
| `--dtype` | vllm_ascend_arg | docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md, docs/source/tutorials/hardwares/310p.md, docs/source/tutorials/models/PaddleOCR-VL.md (+1 more) |
| `--ec-transfer-config` | vllm_ascend_arg | examples/disaggregated_encoder/disagg_1e1pd_example.sh, tests/e2e/nightly/single_node/models/configs/Qwen2.5-VL-7B-Instruct-EPD.yaml |
| `--enable-chunked-prefill` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/models/GLM5.md, tests/e2e/nightly/multi_node/config/DeepSeek-R1-W8A8-longseq.yaml |
| `--enable-compress` | vllm_ascend_arg | examples/save_sharded_state_310.py |
| `--enable-expert-parallel` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md (+49 more) |
| `--enable-lora` | vllm_ascend_arg | docs/source/user_guide/feature_guide/lora.md |
| `--enable-prefix-caching` | vllm_ascend_arg | docs/source/tutorials/models/GLM5.md |
| `--enable-request-id-headers` | vllm_ascend_arg | examples/disaggregated_encoder/disagg_1e1pd_example.sh |
| `--enable-sleep-mode` | vllm_ascend_arg | docs/source/user_guide/feature_guide/sleep_mode.md, examples/offline_external_launcher.py, examples/offline_weight_load.py |
| `--encode-servers-urls` | vllm_ascend_arg | examples/disaggregated_encoder/disagg_1e1pd_example.sh, examples/disaggregated_encoder/disagg_epd_proxy.py, tests/e2e/nightly/single_node/models/configs/Qwen2.5-VL-7B-Instruct-EPD.yaml |
| `--encoder-dispatch-mode` | vllm_ascend_arg | examples/disaggregated_encoder/disagg_epd_proxy.py |
| `--endpoint` | vllm_ascend_arg | docs/source/tutorials/models/GLM4.x.md, docs/source/tutorials/models/Qwen3-VL-Embedding.md, docs/source/tutorials/models/Qwen3-VL-Reranker.md (+3 more) |
| `--enforce-eager` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md, docs/source/tutorials/hardwares/310p.md (+29 more) |
| `--engine` | vllm_ascend_arg | docs/source/user_guide/deployment_guide/using_volcano_kthena.md |
| `--engine-base-url` | vllm_ascend_arg | docs/source/user_guide/deployment_guide/using_volcano_kthena.md |
| `--eval-batch-size` | vllm_ascend_arg | docs/source/tutorials/models/Qwen3-Omni-30B-A3B-Thinking.md |
| `--eval-type` | vllm_ascend_arg | docs/source/tutorials/models/Qwen3-Omni-30B-A3B-Thinking.md |
| `--extra-index-url` | vllm_ascend_arg | docs/source/tutorials/models/Qwen-VL-Dense.md |
| `--generation-config` | vllm_ascend_arg | docs/source/tutorials/models/Qwen3-Omni-30B-A3B-Thinking.md |
| `--gpu-memory-utilization` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md (+63 more) |
| `--head` | vllm_ascend_arg | docs/source/tutorials/features/ray.md |
| `--header` | vllm_ascend_arg | docs/source/user_guide/deployment_guide/using_volcano_kthena.md |
| `--headless` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/models/DeepSeek-R1.md, docs/source/tutorials/models/DeepSeek-V3.1.md (+15 more) |
| `--hf-overrides` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/tutorials/models/Qwen3-235B-A22B.md |
| `--host` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md (+45 more) |
| `--ignore-eos` | vllm_ascend_arg | docs/source/tutorials/models/GLM4.x.md, docs/source/tutorials/models/Qwen3-235B-A22B.md, docs/source/user_guide/feature_guide/ucm_deployment.md |
| `--init` | vllm_ascend_arg | docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md, docs/source/tutorials/models/GLM5.md |
| `--kv-transfer-config` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md, docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md (+20 more) |
| `--limit` | vllm_ascend_arg | docs/source/tutorials/models/Qwen3-Omni-30B-A3B-Thinking.md |
| `--limit-mm-per-prompt` | vllm_ascend_arg | docs/source/tutorials/models/Qwen3-VL-30B-A3B-Instruct.md |
| `--load-format` | vllm_ascend_arg | docs/source/tutorials/models/Qwen3-Omni-30B-A3B-Thinking.md, docs/source/user_guide/feature_guide/netloader.md |
| `--location` | vllm_ascend_arg | docs/source/user_guide/deployment_guide/using_volcano_kthena.md |
| `--lora-modules` | vllm_ascend_arg | docs/source/user_guide/feature_guide/lora.md |
| `--master-addr` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, examples/offline_data_parallel.py, examples/offline_external_launcher.py (+1 more) |
| `--master-port` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, examples/offline_data_parallel.py, examples/offline_external_launcher.py (+1 more) |
| `--max` | vllm_ascend_arg | docs/source/user_guide/feature_guide/kv_pool.md |
| `--max-concurrency` | vllm_ascend_arg | docs/source/tutorials/models/GLM4.x.md, docs/source/tutorials/models/Qwen3-235B-A22B.md |
| `--max-model-len` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md (+67 more) |
| `--max-num-batched-tokens` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md (+60 more) |
| `--max-num-seqs` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md (+50 more) |
| `--max-retries` | vllm_ascend_arg | examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py, examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py, examples/external_online_dp/dp_load_balance_proxy_server.py |
| `--max-waiting-retries` | vllm_ascend_arg | examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py |
| `--metric-percentiles` | vllm_ascend_arg | docs/source/user_guide/feature_guide/ucm_deployment.md |
| `--mm-processor-cache-gb` | vllm_ascend_arg | docs/source/tutorials/models/PaddleOCR-VL.md, tests/e2e/nightly/single_node/models/configs/Qwen2.5-VL-32B-Instruct.yaml, tests/e2e/nightly/single_node/models/configs/Qwen2.5-VL-7B-Instruct.yaml |
| `--mode` | vllm_ascend_arg | docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md, docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md, docs/source/tutorials/features/suffix_speculative_decoding.md (+1 more) |
| `--model` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/tutorials/models/DeepSeek-R1.md (+28 more) |
| `--model-loader-extra-config` | vllm_ascend_arg | docs/source/user_guide/feature_guide/netloader.md |
| `--model-weight-gib` | vllm_ascend_arg | examples/offline_external_launcher.py, examples/offline_weight_load.py |
| `--models` | vllm_ascend_arg | docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md, docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md, docs/source/tutorials/features/suffix_speculative_decoding.md (+1 more) |
| `--name` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md (+25 more) |
| `--net` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md (+19 more) |
| `--network` | vllm_ascend_arg | docs/source/tutorials/models/PaddleOCR-VL.md |
| `--nnodes` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md |
| `--no-enable-chunked-prefill` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md |
| `--no-enable-prefix-caching` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md (+45 more) |
| `--node-ip-address` | vllm_ascend_arg | docs/source/tutorials/features/ray.md |
| `--node-rank` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, examples/offline_data_parallel.py, examples/offline_external_launcher.py (+1 more) |
| `--node-size` | vllm_ascend_arg | examples/offline_data_parallel.py, examples/offline_external_launcher.py, examples/offline_weight_load.py |
| `--num-prompts` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/tutorials/features/suffix_speculative_decoding.md (+14 more) |
| `--output` | vllm_ascend_arg | docs/source/user_guide/deployment_guide/using_volcano_kthena.md, examples/save_sharded_state_310.py |
| `--output-dir` | vllm_ascend_arg | docs/source/user_guide/deployment_guide/using_volcano_kthena.md |
| `--pcp` | vllm_ascend_arg | examples/offline_inference_npu_long_seq.py |
| `--percentile-metrics` | vllm_ascend_arg | docs/source/user_guide/feature_guide/ucm_deployment.md |
| `--pipeline-parallel-size` | vllm_ascend_arg | docs/source/tutorials/features/ray.md |
| `--pod` | vllm_ascend_arg | docs/source/user_guide/deployment_guide/using_volcano_kthena.md |
| `--port` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md (+72 more) |
| `--prefill-context-parallel-size` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/user_guide/feature_guide/context_parallel.md (+2 more) |
| `--prefill-servers-urls` | vllm_ascend_arg | examples/disaggregated_encoder/disagg_1e1pd_example.sh, examples/disaggregated_encoder/disagg_epd_proxy.py, tests/e2e/nightly/single_node/models/configs/Qwen2.5-VL-7B-Instruct-EPD.yaml |
| `--prefiller-hosts` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md, docs/source/tutorials/features/pd_disaggregation_mooncake_single_node.md (+7 more) |
| `--prefiller-hosts-num` | vllm_ascend_arg | docs/source/user_guide/feature_guide/large_scale_ep.md |
| `--prefiller-port` | vllm_ascend_arg | docs/source/tutorials/features/pd_disaggregation_mooncake_single_node.md, docs/source/tutorials/models/Qwen3-235B-A22B.md |
| `--prefiller-ports` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md, docs/source/tutorials/models/DeepSeek-V3.1.md (+5 more) |
| `--prefiller-ports-inc` | vllm_ascend_arg | docs/source/user_guide/feature_guide/large_scale_ep.md |
| `--prefix-repetition-num-prefixes` | vllm_ascend_arg | docs/source/tutorials/models/GLM4.x.md |
| `--prefix-repetition-output-len` | vllm_ascend_arg | docs/source/tutorials/models/GLM4.x.md |
| `--prefix-repetition-prefix-len` | vllm_ascend_arg | docs/source/tutorials/models/GLM4.x.md |
| `--prefix-repetition-suffix-len` | vllm_ascend_arg | docs/source/tutorials/models/GLM4.x.md |
| `--privileged` | vllm_ascend_arg | docs/source/tutorials/models/PaddleOCR-VL.md, docs/source/tutorials/models/Qwen3-Dense.md, docs/source/tutorials/models/Qwen3-VL-235B-A22B-Instruct.md |
| `--proc-per-node` | vllm_ascend_arg | examples/offline_external_launcher.py, examples/offline_weight_load.py |
| `--profiler-config` | vllm_ascend_arg | docs/source/tutorials/models/DeepSeek-V3.2.md, docs/source/user_guide/release_notes.md |
| `--quantization` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md (+39 more) |
| `--random-input` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/tutorials/models/DeepSeek-R1.md (+14 more) |
| `--random-input-len` | vllm_ascend_arg | docs/source/tutorials/models/Qwen3-235B-A22B.md, docs/source/user_guide/feature_guide/ucm_deployment.md |
| `--random-output-len` | vllm_ascend_arg | docs/source/tutorials/models/Qwen3-235B-A22B.md, docs/source/user_guide/feature_guide/ucm_deployment.md |
| `--reasoning-parser` | vllm_ascend_arg | tests/e2e/nightly/multi_node/config/DeepSeek-V3_2-W8A8-A3-dual-nodes.yaml, tests/e2e/nightly/multi_node/config/DeepSeek-V3_2-W8A8-EP.yaml, tests/e2e/nightly/multi_node/config/DeepSeek-V3_2-W8A8-cp.yaml (+7 more) |
| `--recursive` | vllm_ascend_arg | docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md, docs/source/tutorials/models/GLM5.md |
| `--request-rate` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/tutorials/models/DeepSeek-R1.md (+12 more) |
| `--result-dir` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/tutorials/models/DeepSeek-R1.md (+14 more) |
| `--retry-delay` | vllm_ascend_arg | examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py, examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py, examples/external_online_dp/dp_load_balance_proxy_server.py |
| `--rm` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md (+25 more) |
| `--runner` | vllm_ascend_arg | docs/source/tutorials/models/Qwen3-VL-Embedding.md, docs/source/tutorials/models/Qwen3-VL-Reranker.md, docs/source/tutorials/models/Qwen3_embedding.md |
| `--save-result` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/tutorials/models/DeepSeek-R1.md (+14 more) |
| `--seed` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md (+35 more) |
| `--served-model-name` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md (+27 more) |
| `--shm-size` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md (+25 more) |
| `--sleep-mode-level` | vllm_ascend_arg | examples/offline_external_launcher.py |
| `--source` | vllm_ascend_arg | docs/source/user_guide/deployment_guide/using_volcano_kthena.md |
| `--speculative-config` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md, docs/source/tutorials/features/suffix_speculative_decoding.md (+21 more) |
| `--summarizer` | vllm_ascend_arg | docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md, docs/source/tutorials/features/suffix_speculative_decoding.md |
| `--swap-space` | vllm_ascend_arg | docs/source/tutorials/models/Qwen3-Omni-30B-A3B-Thinking.md |
| `--task` | vllm_ascend_arg | docs/source/user_guide/release_notes.md |
| `--tasks` | vllm_ascend_arg | docs/source/tutorials/models/DeepSeek-R1.md, docs/source/tutorials/models/DeepSeek-V3.2.md, docs/source/tutorials/models/Qwen-VL-Dense.md |
| `--temperature` | vllm_ascend_arg | examples/offline_external_launcher.py, examples/offline_weight_load.py |
| `--tensor-parallel-size` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md (+74 more) |
| `--tokenizer` | vllm_ascend_arg | docs/source/tutorials/models/GLM4.x.md, docs/source/tutorials/models/Qwen3-235B-A22B.md, docs/source/tutorials/models/Qwen3_embedding.md (+1 more) |
| `--tokenizer-mode` | vllm_ascend_arg | tests/e2e/nightly/multi_node/config/DeepSeek-V3_2-W8A8-A3-dual-nodes.yaml, tests/e2e/nightly/multi_node/config/DeepSeek-V3_2-W8A8-EP.yaml, tests/e2e/nightly/multi_node/config/DeepSeek-V3_2-W8A8-cp.yaml |
| `--tp` | vllm_ascend_arg | examples/offline_inference_npu_long_seq.py |
| `--tp-size` | vllm_ascend_arg | docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md, docs/source/tutorials/models/DeepSeek-V3.1.md, docs/source/tutorials/models/DeepSeek-V3.2.md (+5 more) |
| `--trust-remote-code` | vllm_ascend_arg | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md, docs/source/tutorials/features/long_sequence_context_parallel_single_node.md, docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md (+66 more) |
| `--vllm-start-port` | vllm_ascend_arg | docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md, docs/source/tutorials/models/DeepSeek-V3.1.md, docs/source/tutorials/models/DeepSeek-V3.2.md (+1 more) |
| `--waiting-retry-interval` | vllm_ascend_arg | examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py |

Detailed semantics and combinations:
- `../../deployment-config/references/global-parameter-feature-map.md`
- `../../deployment-config/references/global-parameter-combination-guide.md`

Machine-readable artifacts:
- `generated/vllm_ascend_args_inventory.json`
- `generated/vllm_ascend_env_inventory.json`

Back to [INDEX](../../INDEX.md).
