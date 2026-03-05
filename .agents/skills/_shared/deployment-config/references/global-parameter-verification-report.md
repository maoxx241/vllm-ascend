---
knowledge_id: deployment-config.global-parameter-verification-report
domain: deployment-config
knowledge_type: verification
summary: Dual-baseline verification report (local code truth + upstream web checks).
last_verified: "2026-03-05"
source_commit: "workspace-head"
freshness: "fresh"
---

# Global Parameter Verification Report

- Coverage ratio: **1.0**
- Evidence completeness ratio: **1.0**
- Conflict count: **52**
- High-risk validated count: **52**
- Official refs: **1195**
- External refs: **337**
- Entries with external refs: **282**

## Unresolved items (first 50)

- `vllm.arg.disable_log_requests:upstream_delta`
- `vllm.arg.logits_processor_pattern:upstream_delta`
- `vllm.env.vllm_sleep_when_idle:upstream_delta`
- `vllm_ascend.arg.decode_servers_urls:needs_manual_review`
- `vllm_ascend.arg.decoder_hosts:needs_manual_review`
- `vllm_ascend.arg.decoder_ports:needs_manual_review`
- `vllm_ascend.arg.dp_address:needs_manual_review`
- `vllm_ascend.arg.dp_hosts:needs_manual_review`
- `vllm_ascend.arg.dp_ports:needs_manual_review`
- `vllm_ascend.arg.dp_rank_start:needs_manual_review`
- `vllm_ascend.arg.dp_rpc_port:needs_manual_review`
- `vllm_ascend.arg.dp_size:needs_manual_review`
- `vllm_ascend.arg.dp_size_local:needs_manual_review`
- `vllm_ascend.arg.encode_servers_urls:needs_manual_review`
- `vllm_ascend.arg.encoder_dispatch_mode:needs_manual_review`
- `vllm_ascend.arg.host:needs_manual_review`
- `vllm_ascend.arg.max_retries:needs_manual_review`
- `vllm_ascend.arg.max_waiting_retries:needs_manual_review`
- `vllm_ascend.arg.model_weight_gib:needs_manual_review`
- `vllm_ascend.arg.node_size:needs_manual_review`
- `vllm_ascend.arg.port:needs_manual_review`
- `vllm_ascend.arg.prefill_servers_urls:needs_manual_review`
- `vllm_ascend.arg.prefiller_hosts:needs_manual_review`
- `vllm_ascend.arg.prefiller_ports:needs_manual_review`
- `vllm_ascend.arg.proc_per_node:needs_manual_review`
- `vllm_ascend.arg.retry_delay:needs_manual_review`
- `vllm_ascend.arg.sleep_mode_level:needs_manual_review`
- `vllm_ascend.arg.temperature:needs_manual_review`
- `vllm_ascend.arg.tp_size:needs_manual_review`
- `vllm_ascend.arg.vllm_start_port:needs_manual_review`
- `vllm_ascend.arg.waiting_retry_interval:needs_manual_review`
- `vllm_ascend.env.ascend_custom_opp_path:upstream_delta`
- `vllm_ascend.env.ascend_enable_use_fabric_mem:upstream_delta`
- `vllm_ascend.env.ascend_rt_visible_devices:upstream_delta`
- `vllm_ascend.env.ascend_transfer_timeout:upstream_delta`
- `vllm_ascend.env.expert_map_record:upstream_delta`
- `vllm_ascend.env.hccl_deterministic:upstream_delta`
- `vllm_ascend.env.hccl_intra_pcie_enable:upstream_delta`
- `vllm_ascend.env.hccl_intra_roce_enable:upstream_delta`
- `vllm_ascend.env.hccl_op_expansion_mode:upstream_delta`
- `vllm_ascend.env.hccl_rdma_retry_cnt:upstream_delta`
- `vllm_ascend.env.hccl_rdma_timeout:upstream_delta`
- `vllm_ascend.env.lccl_deterministic:upstream_delta`
- `vllm_ascend.env.master_addr:upstream_delta`
- `vllm_ascend.env.master_port:upstream_delta`
- `vllm_ascend.env.mooncake_config_path:upstream_delta`
- `vllm_ascend.env.openai_api_key:upstream_delta`
- `vllm_ascend.env.pytorch_npu_alloc_conf:upstream_delta`
- `vllm_ascend.env.rank:upstream_delta`
- `vllm_ascend.env.triton_all_blocks_parallel:upstream_delta`

Back to [INDEX](../../INDEX.md).
