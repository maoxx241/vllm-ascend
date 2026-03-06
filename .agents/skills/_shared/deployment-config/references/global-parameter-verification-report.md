---
knowledge_id: deployment-config.global-parameter-verification-report
domain: deployment-config
knowledge_type: verification
summary: Dual-baseline verification report (local code truth + upstream web checks).
last_verified: "2026-03-06"
source_commit: "workspace-head"
freshness: "fresh"
---

# Global Parameter Verification Report

- Coverage ratio: **1.0**
- Evidence completeness ratio: **0.9891**
- Conflict count: **72**
- High-risk validated count: **53**
- Official refs: **1272**
- External refs: **338**
- Entries with external refs: **283**
- Value semantics done: **491**
- Value semantics ratio: **1.0**

## Unresolved items (first 50)

- `vllm.arg.api_server_count:upstream_delta`
- `vllm.arg.config:upstream_delta`
- `vllm.arg.disable_log_requests:upstream_delta`
- `vllm.arg.headless:upstream_delta`
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
- `vllm_ascend.env.acl_op_init_mode:missing_behavior_ref`
- `vllm_ascend.env.acl_op_init_mode:upstream_delta`
- `vllm_ascend.env.ascend_a3_enable:missing_behavior_ref`
- `vllm_ascend.env.ascend_a3_enable:upstream_delta`
- `vllm_ascend.env.ascend_aggregate_enable:missing_behavior_ref`
- `vllm_ascend.env.ascend_aggregate_enable:upstream_delta`
- `vllm_ascend.env.ascend_buffer_pool:missing_behavior_ref`
- `vllm_ascend.env.ascend_buffer_pool:upstream_delta`
- `vllm_ascend.env.ascend_connect_timeout:missing_behavior_ref`
- `vllm_ascend.env.ascend_connect_timeout:upstream_delta`
- `vllm_ascend.env.ascend_custom_opp_path:upstream_delta`
- `vllm_ascend.env.ascend_enable_use_fabric_mem:upstream_delta`
- `vllm_ascend.env.ascend_rt_visible_devices:upstream_delta`
- `vllm_ascend.env.ascend_transfer_timeout:upstream_delta`
- `vllm_ascend.env.ascend_transport_print:missing_behavior_ref`
- `vllm_ascend.env.ascend_transport_print:upstream_delta`

Back to [INDEX](../../INDEX.md).
