---
knowledge_id: vllm-ascend-core.inputs-and-envs-global
domain: vllm-ascend-core
knowledge_type: reference
summary: Code-truth inventory of vLLM-Ascend deployment arguments and environment variables with evidence refs.
last_verified: "2026-03-06"
source_commit: "workspace-head"
freshness: "fresh"
---

# vLLM-Ascend Global Inputs and Envs (Code Truth)

- vLLM-Ascend deployment args: **37**
- vLLM-Ascend env vars: **71**

## vLLM-Ascend Deployment Args

| Name | Type | Definition ref |
| --- | --- | --- |
| `--decode-servers-urls` | string | examples/disaggregated_encoder/disagg_epd_proxy.py:711 |
| `--decoder-hosts` | list | examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py:264, examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:512 |
| `--decoder-ports` | list | examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py:265, examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:513 |
| `--dp-address` | string | examples/external_online_dp/launch_online_dp.py:14 |
| `--dp-hosts` | list | examples/external_online_dp/dp_load_balance_proxy_server.py:185 |
| `--dp-ports` | list | examples/external_online_dp/dp_load_balance_proxy_server.py:186 |
| `--dp-rank-start` | int | examples/external_online_dp/launch_online_dp.py:13 |
| `--dp-rpc-port` | int | examples/external_online_dp/launch_online_dp.py:15 |
| `--dp-size` | int | examples/external_online_dp/launch_online_dp.py:10, examples/offline_data_parallel.py:81 |
| `--dp-size-local` | int | examples/external_online_dp/launch_online_dp.py:12 |
| `--enable-expert-parallel` | bool | examples/offline_data_parallel.py:89, examples/offline_external_launcher.py:122 |
| `--enable-sleep-mode` | bool | examples/offline_external_launcher.py:125, examples/offline_weight_load.py:134 |
| `--encode-servers-urls` | string | examples/disaggregated_encoder/disagg_epd_proxy.py:700 |
| `--encoder-dispatch-mode` | string | examples/disaggregated_encoder/disagg_epd_proxy.py:717 |
| `--enforce-eager` | bool | examples/offline_data_parallel.py:87, examples/offline_external_launcher.py:120 |
| `--host` | string | examples/disaggregated_encoder/disagg_epd_proxy.py:698, examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py:261 |
| `--master-addr` | string | examples/offline_data_parallel.py:85, examples/offline_external_launcher.py:118 |
| `--master-port` | int | examples/offline_data_parallel.py:86, examples/offline_external_launcher.py:119 |
| `--max-retries` | int | examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py:266, examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:514 |
| `--max-waiting-retries` | int | examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:518 |
| `--model` | string | examples/offline_data_parallel.py:75, examples/offline_external_launcher.py:108 |
| `--model-weight-gib` | float | examples/offline_external_launcher.py:129, examples/offline_weight_load.py:138 |
| `--node-rank` | int | examples/offline_data_parallel.py:84, examples/offline_external_launcher.py:116 |
| `--node-size` | int | examples/offline_data_parallel.py:83, examples/offline_external_launcher.py:115 |
| `--port` | int | examples/disaggregated_encoder/disagg_epd_proxy.py:699, examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py:260 |
| `--prefill-servers-urls` | string | examples/disaggregated_encoder/disagg_epd_proxy.py:705 |
| `--prefiller-hosts` | list | examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py:262, examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:510 |
| `--prefiller-ports` | list | examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py:263, examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:511 |
| `--proc-per-node` | int | examples/offline_external_launcher.py:117, examples/offline_weight_load.py:126 |
| `--quantization` | string | examples/offline_data_parallel.py:92 |
| `--retry-delay` | float | examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py:267, examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:515 |
| `--sleep-mode-level` | int | examples/offline_external_launcher.py:135 |
| `--temperature` | float | examples/offline_external_launcher.py:126, examples/offline_weight_load.py:135 |
| `--tp-size` | int | examples/external_online_dp/launch_online_dp.py:11, examples/offline_data_parallel.py:82 |
| `--trust-remote-code` | bool | examples/offline_data_parallel.py:88, examples/offline_external_launcher.py:121 |
| `--vllm-start-port` | int | examples/external_online_dp/launch_online_dp.py:16 |
| `--waiting-retry-interval` | int | examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:521 |

## vLLM-Ascend Env Vars

| Name | Type | Definition ref |
| --- | --- | --- |
| `ACL_OP_INIT_MODE` | string | docs/source/tutorials/models/DeepSeek-V3.2.md:527, docs/source/tutorials/models/DeepSeek-V3.2.md:602 |
| `ASCEND_A3_ENABLE` | string | docs/source/tutorials/models/DeepSeek-V3.2.md:528, docs/source/tutorials/models/DeepSeek-V3.2.md:603 |
| `ASCEND_AGGREGATE_ENABLE` | string | docs/source/tutorials/models/DeepSeek-V3.2.md:525, docs/source/tutorials/models/DeepSeek-V3.2.md:600 |
| `ASCEND_BUFFER_POOL` | string | docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md:221, docs/source/tutorials/models/DeepSeek-V3.1.md:299 |
| `ASCEND_CONNECT_TIMEOUT` | string | docs/source/user_guide/feature_guide/kv_pool.md:157, docs/source/user_guide/feature_guide/kv_pool.md:224 |
| `ASCEND_CUSTOM_OPP_PATH` | string | vllm_ascend/platform.py:471, vllm_ascend/platform.py:473 |
| `ASCEND_ENABLE_USE_FABRIC_MEM` | string | docs/source/user_guide/feature_guide/kv_pool.md:90, vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py:39 |
| `ASCEND_HOME_PATH` | string | vllm_ascend/envs.py:58, vllm_ascend/worker/worker.py:166 |
| `ASCEND_RT_VISIBLE_DEVICES` | string | docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:272, docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:331 |
| `ASCEND_TRANSFER_TIMEOUT` | string | docs/source/user_guide/feature_guide/kv_pool.md:160, docs/source/user_guide/feature_guide/kv_pool.md:225 |
| `ASCEND_TRANSPORT_PRINT` | string | docs/source/tutorials/models/DeepSeek-V3.2.md:526, docs/source/tutorials/models/DeepSeek-V3.2.md:601 |
| `CMAKE_BUILD_TYPE` | string | vllm_ascend/envs.py:37 |
| `COMPILE_CUSTOM_KERNELS` | int | vllm_ascend/envs.py:43 |
| `CXX_COMPILER` | string | vllm_ascend/envs.py:46 |
| `C_COMPILER` | string | vllm_ascend/envs.py:49 |
| `DYNAMIC_EPLB` | string | vllm_ascend/ascend_config.py:429, vllm_ascend/envs.py:105 |
| `EXPERT_MAP_RECORD` | string | vllm_ascend/ascend_config.py:430, vllm_ascend/patch/platform/__init__.py:23 |
| `HCCL_BUFFSIZE` | string | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:153, docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:220 |
| `HCCL_CONNECT_TIMEOUT` | string | docs/source/tutorials/models/DeepSeek-V3.1.md:289, docs/source/tutorials/models/DeepSeek-V3.1.md:366 |
| `HCCL_DETERMINISTIC` | string | docs/source/faqs.md:199, vllm_ascend/batch_invariant.py:85 |
| `HCCL_EXEC_TIMEOUT` | string | docs/source/tutorials/models/DeepSeek-V3.1.md:288, docs/source/tutorials/models/DeepSeek-V3.1.md:365 |
| `HCCL_IF_IP` | string | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:149, docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:216 |
| `HCCL_INTRA_PCIE_ENABLE` | string | docs/source/tutorials/models/DeepSeek-R1.md:153, docs/source/tutorials/models/DeepSeek-R1.md:199 |
| `HCCL_INTRA_ROCE_ENABLE` | string | docs/source/tutorials/models/DeepSeek-R1.md:154, docs/source/tutorials/models/DeepSeek-R1.md:200 |
| `HCCL_OP_EXPANSION_MODE` | string | docs/source/developer_guide/performance_and_debug/optimization_and_tuning.md:176, docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:158 |
| `HCCL_RDMA_RETRY_CNT` | string | vllm_ascend/distributed/kv_transfer/utils/utils.py:56 |
| `HCCL_RDMA_TIMEOUT` | string | vllm_ascend/distributed/kv_transfer/utils/utils.py:55 |
| `HCCL_SOCKET_IFNAME` | string | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:152, docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:219 |
| `HCCL_SO_PATH` | string | vllm_ascend/envs.py:61 |
| `LCCL_DETERMINISTIC` | string | vllm_ascend/batch_invariant.py:86 |
| `LOCAL_RANK` | string | examples/offline_external_launcher.py:176, examples/offline_weight_load.py:177 |
| `MASTER_ADDR` | string | examples/offline_external_launcher.py:173, examples/offline_weight_load.py:174 |
| `MASTER_PORT` | string | examples/offline_external_launcher.py:174, examples/offline_weight_load.py:175 |
| `MAX_JOBS` | string | vllm_ascend/envs.py:34 |
| `MOONCAKE_CONFIG_PATH` | string | vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py:125 |
| `MSMONITOR_USE_DAEMON` | int | vllm_ascend/envs.py:91 |
| `OMP_PROC_BIND` | string | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:154, docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:221 |
| `OPENAI_API_KEY` | string | examples/disaggregated_encoder/disagg_epd_proxy.py:650 |
| `PAGED_ATTENTION_MASK_LEN` | string | tests/e2e/nightly/single_node/models/configs/Qwen3-32B.yaml:12 |
| `PYTORCH_NPU_ALLOC_CONF` | string | docs/source/developer_guide/performance_and_debug/optimization_and_tuning.md:149, docs/source/developer_guide/performance_and_debug/optimization_and_tuning.md:152 |
| `RANK` | string | examples/offline_external_launcher.py:175, examples/offline_weight_load.py:176 |
| `SERVER_PORT` | string | tests/e2e/nightly/single_node/models/configs/Prefix-Cache-Qwen3-32B-Int8.yaml:11, tests/e2e/nightly/single_node/models/configs/QwQ-32B.yaml:10 |
| `SOC_VERSION` | string | vllm_ascend/envs.py:53 |
| `TASK_QUEUE_ENABLE` | string | docs/source/developer_guide/performance_and_debug/optimization_and_tuning.md:160, docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:160 |
| `TRITON_ALL_BLOCKS_PARALLEL` | string | vllm_ascend/ops/rotary_embedding.py:497 |
| `VERBOSE` | int | vllm_ascend/envs.py:55 |
| `VLLM_ASCEND_BALANCE_SCHEDULING` | int | docs/source/tutorials/models/DeepSeek-R1.md:152, docs/source/tutorials/models/DeepSeek-R1.md:198 |
| `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL` | int | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:161, docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:227 |
| `VLLM_ASCEND_ENABLE_FLASHCOMM` | string | vllm_ascend/utils.py:765 |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | int | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:155, docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:320 |
| `VLLM_ASCEND_ENABLE_FUSED_MC2` | int | docs/source/tutorials/models/Qwen3-235B-A22B.md:346, docs/source/tutorials/models/Qwen3-235B-A22B.md:392 |
| `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE` | int | vllm_ascend/batch_invariant.py:82, vllm_ascend/envs.py:71 |
| `VLLM_ASCEND_ENABLE_MLAPO` | int | docs/source/tutorials/models/DeepSeek-V3.2.md:141, docs/source/tutorials/models/DeepSeek-V3.2.md:200 |
| `VLLM_ASCEND_ENABLE_NZ` | int | docs/source/user_guide/feature_guide/sleep_mode.md:81, tests/e2e/nightly/single_node/models/configs/Qwen2.5-VL-32B-Instruct.yaml:10 |
| `VLLM_ASCEND_ENABLE_PREFETCH_MLP` | int | docs/source/tutorials/features/suffix_speculative_decoding.md:84, vllm_ascend/ascend_config.py:150 |
| `VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED` | string | docs/source/user_guide/feature_guide/large_scale_ep.md:141, docs/source/user_guide/feature_guide/large_scale_ep.md:208 |
| `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE` | int | docs/source/user_guide/feature_guide/layer_sharding.md:53, vllm_ascend/envs.py:79 |
| `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK` | int | vllm_ascend/envs.py:117, vllm_ascend/envs.py:118 |
| `VLLM_ASCEND_MLP_DOWN_PREFETCH_SIZE` | int | vllm_ascend/ascend_config.py:153, vllm_ascend/envs.py:87 |
| `VLLM_ASCEND_MLP_GATE_UP_PREFETCH_SIZE` | int | vllm_ascend/ascend_config.py:152, vllm_ascend/envs.py:83 |
| `VLLM_DISABLE_SHARED_EXPERTS_STREAM` | string | vllm_ascend/platform.py:31 |
| `VLLM_DP_MASTER_IP` | string | examples/offline_data_parallel.py:123 |
| `VLLM_DP_MASTER_PORT` | string | examples/offline_data_parallel.py:124 |
| `VLLM_DP_RANK` | string | examples/offline_data_parallel.py:120 |
| `VLLM_DP_RANK_LOCAL` | string | examples/offline_data_parallel.py:121 |
| `VLLM_DP_SIZE` | string | examples/offline_data_parallel.py:122 |
| `VLLM_USE_MODELSCOPE` | string | examples/offline_data_parallel.py:67, examples/offline_external_launcher.py:76 |
| `VLLM_USE_V1` | string | docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:159, docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:225 |
| `VLLM_VERSION` | string | vllm_ascend/envs.py:68 |
| `VLLM_WORKER_MULTIPROC_METHOD` | string | examples/offline_data_parallel.py:68, examples/offline_external_launcher.py:77 |
| `WORLD_SIZE` | string | examples/offline_external_launcher.py:177, examples/offline_weight_load.py:178 |

Back to [INDEX](../../INDEX.md).
