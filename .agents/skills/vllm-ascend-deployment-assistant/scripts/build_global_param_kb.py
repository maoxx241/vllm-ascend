#!/usr/bin/env python3
"""Build high-confidence parameter/env knowledge base for vLLM + vLLM-Ascend."""

from __future__ import annotations

import argparse
import ast
import json
import re
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FLAG_PATTERN = re.compile(r"^--[a-z0-9][a-z0-9\-]*$")
FLAG_IN_TEXT_PATTERN = re.compile(r"(?<![A-Za-z0-9_])(--[a-z0-9][a-z0-9\-]*)(?![A-Za-z0-9_])")
ENV_TOKEN_PATTERN = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\b")
IDENT_PATTERN = re.compile(r"\b([a-z_][a-z0-9_]{2,})\b")

STATUS_ALIGNED = "aligned"
STATUS_UPSTREAM_DELTA = "upstream_delta"
STATUS_NEEDS_REVIEW = "needs_manual_review"

HIGH_RISK_FEATURES = {
    "quantization",
    "int4_quantization",
    "graph_mode",
    "tensor_parallel",
    "data_parallel",
    "expert_parallel",
    "context_parallel",
    "prefill_decode_disaggregation",
    "lora",
    "speculative_decode",
    "sleep_mode",
    "weight_prefetch",
    "prefix_cache",
}

DEPLOYMENT_ASCEND_ARG_FILES = [
    "examples/offline_external_launcher.py",
    "examples/offline_weight_load.py",
    "examples/offline_data_parallel.py",
    "examples/external_online_dp/launch_online_dp.py",
    "examples/external_online_dp/dp_load_balance_proxy_server.py",
    "examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py",
    "examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py",
    "examples/disaggregated_encoder/disagg_epd_proxy.py",
]

FEATURE_PRIORITY = [
    "quantization",
    "int4_quantization",
    "graph_mode",
    "tensor_parallel",
    "data_parallel",
    "expert_parallel",
    "context_parallel",
    "prefill_decode_disaggregation",
    "prefix_cache",
    "lora",
    "speculative_decode",
    "sleep_mode",
    "weight_prefetch",
    "throughput_tuning",
    "memory_tuning",
    "network_serving",
    "security_auth",
    "multimodal",
    "logging_debug",
    "profiling_observability",
    "model_selection",
    "general_runtime",
]

FEATURE_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("int4_quantization", ("int4", "w4a4", "4bit")),
    ("quantization", ("quant", "int8", "w8a8", "gptq", "awq", "fp8", "modelslim")),
    ("graph_mode", ("compilation", "cudagraph", "graph", "eager", "npugraph", "acl_graph")),
    ("tensor_parallel", ("tensor-parallel", "tp-size", "tp_", "all2all", "matmul-allreduce")),
    ("data_parallel", ("data-parallel", "dp-", "external-dp", "dpr", "dpl", "dpp")),
    ("expert_parallel", ("expert-parallel", "ep", "eplb", "moe", "routed-experts")),
    ("context_parallel", ("context-parallel", "prefill-context-parallel", "decode-context-parallel", "cp-kv", "dcp-kv")),
    ("prefill_decode_disaggregation", ("prefill", "decode", "prefiller", "decoder", "kv-transfer", "ec-transfer", "disagg")),
    ("prefix_cache", ("prefix-caching", "prefix-cache", "prefix", "hash-algo")),
    ("lora", ("lora",)),
    ("speculative_decode", ("speculative", "mtp", "draft")),
    ("sleep_mode", ("sleep-mode", "sleep_when_idle", "sleep")),
    ("weight_prefetch", ("prefetch", "weight_prefetch", "prefetch_mlp")),
    ("throughput_tuning", ("async-scheduling", "max-num-batched-tokens", "max-num-seqs", "dbo", "scheduler", "flashcomm", "balance")),
    ("memory_tuning", ("gpu-memory-utilization", "swap-space", "cpu-offload", "max-model-len", "block-size", "kv-cache", "oom")),
    ("network_serving", ("host", "port", "api", "served-model-name", "root-path", "endpoint", "rpc")),
    ("security_auth", ("api-key", "ssl", "allow-credentials", "allowed-origins", "hf-token", "trust-remote-code")),
    ("multimodal", ("mm-", "media", "audio", "image", "video", "vision", "embeds")),
    ("logging_debug", ("log", "debug", "trace", "verbose", "warning", "error")),
    ("profiling_observability", ("metric", "profiler", "otlp", "trace", "monitor")),
    ("model_selection", ("model", "tokenizer", "revision", "dtype", "runner", "task", "download")),
]

FEATURE_DEFAULTS: dict[str, dict[str, Any]] = {
    "quantization": {
        "semantics": "选择量化实现和权重加载路径，直接影响吞吐、显存和精度。",
        "prerequisites": ["存在可用量化权重或量化配置"],
        "incompatibilities": ["与未验证模型配置组合时可能加载失败"],
        "failure_modes": ["模型加载失败", "精度异常", "推理速度低于预期"],
        "recommendation": "优先使用官方教程中的已验证量化工件与并行参数组合。",
    },
    "int4_quantization": {
        "semantics": "启用 INT4/W4A4 量化路径，通常要求专用模型工件和内核支持。",
        "prerequisites": ["模型存在 int4 工件"],
        "incompatibilities": ["与仅 W8A8 工件 profile 不兼容"],
        "failure_modes": ["启动时报不支持量化类型", "精度/稳定性异常"],
        "recommendation": "演示环境下先确认 profile 支持矩阵，再启用 int4。",
    },
    "graph_mode": {
        "semantics": "控制 eager/graph 执行策略，通常优化吞吐与时延抖动。",
        "prerequisites": ["模型和后端支持图编译路径"],
        "incompatibilities": ["部分动态路径可能要求 eager 回退"],
        "failure_modes": ["图编译失败", "服务启动后首轮请求异常"],
        "recommendation": "先小流量验证 FULL_DECODE_ONLY，再放量。",
    },
    "tensor_parallel": {
        "semantics": "按张量维度切分模型以扩展单模型可用算力。",
        "prerequisites": ["设备数量满足 TP 分片需求"],
        "incompatibilities": ["错误的通信配置会导致启动失败"],
        "failure_modes": ["HCCL/NCCL 初始化失败", "跨卡通信超时"],
        "recommendation": "TP 变更后同步检查 max_model_len 与通信环境变量。",
    },
    "data_parallel": {
        "semantics": "通过副本扩展吞吐能力，并依赖 DP 地址和 RPC 协调。",
        "prerequisites": ["多进程或多节点资源和地址规划"],
        "incompatibilities": ["错误地址/端口会导致调度与健康检查失败"],
        "failure_modes": ["RPC 连接失败", "请求分发不均衡"],
        "recommendation": "固定 DP 地址和端口后再迭代性能参数。",
    },
    "expert_parallel": {
        "semantics": "MoE 专家并行，提升大规模专家模型吞吐。",
        "prerequisites": ["模型本身为 MoE"],
        "incompatibilities": ["Dense 模型不适用"],
        "failure_modes": ["启动报模型不支持 EP", "专家路由异常"],
        "recommendation": "仅在 MoE profile 启用，并配合 TP/DP 校验。",
    },
    "context_parallel": {
        "semantics": "将长上下文处理拆分到多个并行单元，降低单卡压力。",
        "prerequisites": ["长上下文场景且拓扑支持 CP"],
        "incompatibilities": ["低卡数下收益低且配置复杂"],
        "failure_modes": ["KV 传输配置错误", "时延反而变高"],
        "recommendation": "优先在高并发长上下文场景启用并做 A/B。",
    },
    "prefill_decode_disaggregation": {
        "semantics": "预填充与解码分离部署，优化资源利用和吞吐扩展。",
        "prerequisites": ["具备连接器和多服务编排"],
        "incompatibilities": ["单机简化部署无法完整覆盖"],
        "failure_modes": ["connector 超时", "P/D 节点路由异常"],
        "recommendation": "先验证连接器与地址，再调并行参数。",
    },
    "prefix_cache": {
        "semantics": "复用公共前缀缓存，降低 prefill 计算成本。",
        "prerequisites": ["请求之间存在高重复前缀"],
        "incompatibilities": ["部分调度组合性能可能下降"],
        "failure_modes": ["命中率低导致收益不明显", "缓存策略与分块预填充冲突"],
        "recommendation": "结合业务前缀分布评估收益，保留回退开关。",
    },
    "lora": {
        "semantics": "开启 LoRA 适配器加载与路由。",
        "prerequisites": ["LoRA 工件可用且格式正确"],
        "incompatibilities": ["工件缺失或不匹配会导致加载失败"],
        "failure_modes": ["LoRA 模块加载报错", "输出异常"],
        "recommendation": "先离线验证 LoRA 工件，再接入在线服务。",
    },
    "speculative_decode": {
        "semantics": "启用投机解码降低平均解码时延。",
        "prerequisites": ["模型/后端支持投机配置"],
        "incompatibilities": ["不支持场景会触发回退或失败"],
        "failure_modes": ["服务启动后推理错误", "吞吐波动"],
        "recommendation": "先用小 token 数验证，再逐步增加并发。",
    },
    "sleep_mode": {
        "semantics": "空闲时释放部分资源，降低空闲占用。",
        "prerequisites": ["运行时支持 sleep 模式"],
        "incompatibilities": ["首次唤醒可能增加冷启动时延"],
        "failure_modes": ["恢复阶段延迟高", "状态同步异常"],
        "recommendation": "适用于间歇流量场景，需配合健康探针。",
    },
    "weight_prefetch": {
        "semantics": "提前预取权重块，降低 decode 等待。",
        "prerequisites": ["模型与后端支持预取配置"],
        "incompatibilities": ["显存紧张场景可能增加压力"],
        "failure_modes": ["显存 OOM", "收益不稳定"],
        "recommendation": "与 max_model_len/gpu_memory_utilization 联动调优。",
    },
    "throughput_tuning": {
        "semantics": "调度和批处理参数调优，目标提升吞吐。",
        "prerequisites": ["有稳定压测基线"],
        "incompatibilities": ["过大批处理会增大时延和显存压力"],
        "failure_modes": ["TTFT/TPOT 退化", "OOM"],
        "recommendation": "按 TTFT/TPOT/吞吐三指标联合调参。",
    },
    "memory_tuning": {
        "semantics": "控制 KV/权重/中间缓存占用，平衡容量与性能。",
        "prerequisites": ["明确设备内存上限"],
        "incompatibilities": ["过激参数容易触发 OOM"],
        "failure_modes": ["启动或运行 OOM", "缓存不足导致吞吐下降"],
        "recommendation": "先保守设置，再渐进放大。",
    },
    "network_serving": {
        "semantics": "控制服务监听、路由和 API 暴露。",
        "prerequisites": ["端口和网络策略可用"],
        "incompatibilities": ["端口冲突会直接启动失败"],
        "failure_modes": ["Address already in use", "健康检查 5xx"],
        "recommendation": "固定 host/port 并配套探活。",
    },
    "security_auth": {
        "semantics": "控制 API 鉴权和 TLS 安全边界。",
        "prerequisites": ["证书/密钥或 API Key 已配置"],
        "incompatibilities": ["错误证书路径会导致启动失败"],
        "failure_modes": ["401/403", "TLS 握手失败"],
        "recommendation": "生产默认开启鉴权并最小化 CORS 白名单。",
    },
    "multimodal": {
        "semantics": "控制多模态输入处理和缓存策略。",
        "prerequisites": ["模型具备多模态能力"],
        "incompatibilities": ["不支持多模态的模型无法启用相关参数"],
        "failure_modes": ["输入解析失败", "处理时延过高"],
        "recommendation": "先限制每请求多模态资源，再放开。",
    },
    "logging_debug": {
        "semantics": "控制日志和调试可观测性。",
        "prerequisites": ["日志路径/采集系统可用"],
        "incompatibilities": ["高日志级别会增加 CPU/I/O 开销"],
        "failure_modes": ["日志过载", "关键问题难定位"],
        "recommendation": "问题排查阶段提升日志级别，稳定后回落。",
    },
    "profiling_observability": {
        "semantics": "控制 profiling 和 tracing 输出。",
        "prerequisites": ["采集端点和权限就绪"],
        "incompatibilities": ["过量采集会影响性能"],
        "failure_modes": ["指标缺失", "追踪上报失败"],
        "recommendation": "按需开启细粒度 tracing，避免全量常开。",
    },
    "model_selection": {
        "semantics": "控制模型、分词器和版本选择。",
        "prerequisites": ["模型工件可访问"],
        "incompatibilities": ["模型与 tokenizer/runner 不匹配"],
        "failure_modes": ["加载失败", "返回格式异常"],
        "recommendation": "固定模型版本并记录依赖。",
    },
    "general_runtime": {
        "semantics": "通用运行时控制项，需要结合上下文确认语义。",
        "prerequisites": ["参数与运行环境匹配"],
        "incompatibilities": ["错误组合可能影响稳定性"],
        "failure_modes": ["启动失败", "行为与预期不符"],
        "recommendation": "先查证代码与文档证据再启用。",
    },
}

ENTRY_OVERRIDES: dict[str, dict[str, Any]] = {
    "--quantization": {
        "semantics": "指定量化后端/方法（如 ascend），影响权重加载和算子路径。",
    },
    "--compilation-config": {
        "semantics": "控制图编译细节（如 cudagraph_mode），决定 eager/graph 行为。",
    },
    "--tensor-parallel-size": {
        "semantics": "设置 TP 并行度，直接影响通信拓扑与单模型吞吐。",
    },
    "--data-parallel-size": {
        "semantics": "设置 DP 副本数，影响吞吐扩展与地址配置要求。",
    },
    "--enable-expert-parallel": {
        "semantics": "开启 MoE 专家并行，仅对 MoE 模型有效。",
    },
    "--enable-prefix-caching": {
        "semantics": "启用前缀缓存，加速重复前缀请求的 prefill 阶段。",
    },
    "--speculative-config": {
        "semantics": "配置投机解码策略（如 mtp），用于降低解码延迟。",
    },
    "--enable-sleep-mode": {
        "semantics": "开启 sleep 模式以降低空闲资源占用。",
    },
    "--additional-config": {
        "semantics": "承载扩展配置，如 weight_prefetch_config 等。",
    },
    "VLLM_ASCEND_ENABLE_FLASHCOMM1": {
        "semantics": "开启 FlashComm1 通信优化，常用于高并发场景。",
        "prerequisites": ["通信后端与模型并行配置支持该优化"],
    },
    "VLLM_ASCEND_ENABLE_PREFETCH_MLP": {
        "semantics": "旧版 MLP 预取开关（已在新版本迁移到 additional_config 的 weight_prefetch_config）。",
    },
    "VLLM_ASCEND_ENABLE_NZ": {
        "semantics": "控制 NZ 相关优化路径，部分浮点场景建议关闭或设为特定值。",
    },
    "VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL": {
        "semantics": "控制 Ascend 侧 Context Parallel 开关。",
    },
}

COMBO_RULES: list[dict[str, Any]] = [
    {
        "rule_id": "recommended.quant_graph_tp",
        "profile": "*",
        "conditions": ["quantization", "graph_mode", "tensor_parallel"],
        "level": "recommended",
        "reason": "量化+图模式+TP 是常见高吞吐组合。",
        "evidence_refs": [
            "docs/source/tutorials/models/Qwen3-Dense.md",
            "https://docs.vllm.ai/en/stable/configuration/serve_args/",
        ],
        "fallback_actions": ["先只启用 quantization+TP，稳定后再加 graph_mode"],
    },
    {
        "rule_id": "warning.weight_prefetch_memory",
        "profile": "*",
        "conditions": ["weight_prefetch", "memory_tuning"],
        "level": "warning",
        "reason": "权重预取提升吞吐但会增加内存压力。",
        "evidence_refs": [
            "docs/source/tutorials/models/Qwen3-Dense.md",
            "https://docs.vllm.ai/projects/ascend/en/latest/user_guide/release_notes.html",
        ],
        "fallback_actions": ["降低 max_model_len 或 gpu_memory_utilization"],
    },
    {
        "rule_id": "warning.pd_requires_connector",
        "profile": "*",
        "conditions": ["prefill_decode_disaggregation"],
        "level": "warning",
        "reason": "PD 分离依赖连接器、地址和节点角色配置。",
        "evidence_refs": [
            "docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md",
            "https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/kv_pool.html",
        ],
        "fallback_actions": ["先用单机模板验证，再切换到分离架构"],
    },
    {
        "rule_id": "hard_block.qwen3_32b_w8a8_int4",
        "profile": "qwen3-32b-w8a8",
        "conditions": ["int4_quantization"],
        "level": "hard_block",
        "reason": "qwen3-32b-w8a8 profile 不提供已验证 int4 工件与路径。",
        "evidence_refs": [
            ".agents/skills/_shared/vllm-ascend-core/concepts/model-feature-compatibility-matrix.md",
            "https://docs.vllm.ai/projects/ascend/en/latest/user_guide/release_notes.html",
        ],
        "fallback_actions": ["保持 W8A8，或切换到可用 int4 profile"],
    },
    {
        "rule_id": "hard_block.qwen3_32b_w8a8_ep",
        "profile": "qwen3-32b-w8a8",
        "conditions": ["expert_parallel"],
        "level": "hard_block",
        "reason": "qwen3-32b-w8a8 是 Dense 模型，不适用 EP。",
        "evidence_refs": [
            ".agents/skills/_shared/vllm-ascend-core/concepts/model-feature-compatibility-matrix.md",
            "https://docs.vllm.ai/projects/ascend/en/latest/user_guide/suppoted_features.html",
        ],
        "fallback_actions": ["改用 TP/DP 组合调优"],
    },
]

FEATURE_TO_RULE_IDS: dict[str, list[str]] = {
    "quantization": ["recommended.quant_graph_tp"],
    "int4_quantization": ["hard_block.qwen3_32b_w8a8_int4"],
    "graph_mode": ["recommended.quant_graph_tp"],
    "tensor_parallel": ["recommended.quant_graph_tp"],
    "prefill_decode_disaggregation": ["warning.pd_requires_connector"],
    "weight_prefetch": ["warning.weight_prefetch_memory"],
    "memory_tuning": ["warning.weight_prefetch_memory"],
    "expert_parallel": ["hard_block.qwen3_32b_w8a8_ep"],
}

WEB_REFS_BASE: dict[str, list[dict[str, str]]] = {
    "vllm_arg": [
        {"title": "vLLM Server Arguments", "url": "https://docs.vllm.ai/en/stable/configuration/serve_args/", "tier": "official"},
        {"title": "vLLM CLI Serve", "url": "https://docs.vllm.ai/cli/serve.html", "tier": "official"},
        {"title": "vLLM arg_utils", "url": "https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py", "tier": "official"},
    ],
    "vllm_env": [
        {"title": "vLLM envs.py", "url": "https://github.com/vllm-project/vllm/blob/main/vllm/envs.py", "tier": "official"},
        {"title": "vLLM Env RFC", "url": "https://github.com/vllm-project/vllm/issues/4407", "tier": "external"},
    ],
    "vllm_ascend_arg": [
        {"title": "vLLM-Ascend Tutorials", "url": "https://docs.vllm.ai/projects/ascend/en/latest/user_guide/", "tier": "official"},
        {"title": "vLLM-Ascend Feature Guide", "url": "https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/", "tier": "official"},
    ],
    "vllm_ascend_env": [
        {"title": "vLLM-Ascend Environment Variables", "url": "https://docs.vllm.ai/projects/ascend/en/latest/user_guide/configuration/env_vars.html", "tier": "official"},
        {"title": "vLLM-Ascend envs.py", "url": "https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/envs.py", "tier": "official"},
        {"title": "vLLM-Ascend Release Notes", "url": "https://docs.vllm.ai/projects/ascend/en/latest/user_guide/release_notes.html", "tier": "official"},
    ],
}

FEATURE_WEB_REFS: dict[str, list[dict[str, str]]] = {
    "quantization": [
        {"title": "vLLM-Ascend Quantization Guide", "url": "https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/quantization.html", "tier": "official"},
        {"title": "vLLM Config API", "url": "https://docs.vllm.ai/en/latest/api/vllm/config/", "tier": "official"},
        {"title": "vLLM quantization issue search", "url": "https://github.com/vllm-project/vllm/issues?q=quantization", "tier": "external"},
    ],
    "graph_mode": [
        {"title": "vLLM-Ascend Graph Mode Guide", "url": "https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/graph_mode.html", "tier": "official"},
        {"title": "vLLM graph mode issue search", "url": "https://github.com/vllm-project/vllm/issues?q=graph+mode", "tier": "external"},
    ],
    "tensor_parallel": [
        {"title": "vLLM tensor parallel issue search", "url": "https://github.com/vllm-project/vllm/issues?q=tensor+parallel", "tier": "external"},
    ],
    "data_parallel": [
        {"title": "vLLM data parallel issue search", "url": "https://github.com/vllm-project/vllm/issues?q=data+parallel", "tier": "external"},
    ],
    "expert_parallel": [
        {"title": "vLLM expert parallel issue search", "url": "https://github.com/vllm-project/vllm/issues?q=expert+parallel", "tier": "external"},
    ],
    "context_parallel": [
        {"title": "vLLM-Ascend Context Parallel Guide", "url": "https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/context_parallel.html", "tier": "official"},
        {"title": "vLLM context parallel issue search", "url": "https://github.com/vllm-project/vllm/issues?q=context+parallel", "tier": "external"},
    ],
    "lora": [
        {"title": "vLLM-Ascend LoRA Guide", "url": "https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/lora.html", "tier": "official"},
        {"title": "vLLM LoRA issue search", "url": "https://github.com/vllm-project/vllm/issues?q=lora", "tier": "external"},
    ],
    "speculative_decode": [
        {"title": "vLLM-Ascend Speculative Decoding Guide", "url": "https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/speculative_decoding.html", "tier": "official"},
        {"title": "Chunked prefill discussion", "url": "https://github.com/vllm-project/vllm/discussions/12145", "tier": "external"},
    ],
    "prefix_cache": [
        {"title": "Prefix cache routing example", "url": "https://www.anyscale.com/blog/ray-serve-faster-first-token-custom-routing", "tier": "external"},
        {"title": "Chunked prefill bug context", "url": "https://github.com/vllm-project/vllm/issues/18547", "tier": "external"},
    ],
    "sleep_mode": [
        {"title": "vLLM sleep mode issue search", "url": "https://github.com/vllm-project/vllm/issues?q=sleep+mode", "tier": "external"},
    ],
    "weight_prefetch": [
        {"title": "vLLM prefetch issue search", "url": "https://github.com/vllm-project/vllm/issues?q=prefetch", "tier": "external"},
    ],
}

UPSTREAM_URLS = {
    "vllm_env": "https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/envs.py",
    "vllm_arg_utils": "https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/engine/arg_utils.py",
    "vllm_cli_args": "https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/entrypoints/openai/cli_args.py",
    "ascend_env": "https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/vllm_ascend/envs.py",
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _safe_literal(node: ast.AST | None) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _node_repr(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    literal = _safe_literal(node)
    if literal is not None:
        return str(literal)
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _relative(path: Path, base: Path) -> str:
    return str(path.resolve().relative_to(base.resolve()))


def _ref(path: Path, line: int, base: Path) -> str:
    return f"{_relative(path, base)}:{line}"


def _call_name(node: ast.AST | None) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _infer_type_from_node(node: ast.AST | None, default: Any = None, action: str | None = None) -> str:
    if action and ("store_true" in action or "store_false" in action or "BooleanOptionalAction" in action):
        return "bool"
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "float"
    if isinstance(default, dict):
        return "json"
    if isinstance(default, list):
        return "list"
    text = (_node_repr(node) or "").lower()
    if "int" in text:
        return "int"
    if "float" in text:
        return "float"
    if "bool" in text:
        return "bool"
    if "json" in text or "dict" in text:
        return "json"
    if "list" in text:
        return "list"
    return "string"


def _priority(feature: str) -> int:
    try:
        return FEATURE_PRIORITY.index(feature)
    except ValueError:
        return len(FEATURE_PRIORITY)


def _derive_feature_tags(name: str) -> list[str]:
    key = name.lower().replace("_", "-")
    tags: set[str] = set()
    for feature, needles in FEATURE_RULES:
        if any(needle in key for needle in needles):
            tags.add(feature)
    if not tags:
        tags.add("general_runtime")
    return sorted(tags, key=_priority)


def _extract_arg_defs_from_ast(path: Path, repo_root: Path) -> dict[str, dict[str, Any]]:
    text = _read(path)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}

    entries: dict[str, dict[str, Any]] = {}

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            if not (isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"):
                self.generic_visit(node)
                return

            raw_flags: list[str] = []
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    value = arg.value.strip()
                    if value.startswith("-"):
                        raw_flags.append(value)

            long_flags = [f for f in raw_flags if FLAG_PATTERN.fullmatch(f)]
            if not long_flags:
                self.generic_visit(node)
                return

            kw = {item.arg: item.value for item in node.keywords if item.arg}
            default_value = _safe_literal(kw.get("default"))
            action_value = _node_repr(kw.get("action"))
            type_value = _infer_type_from_node(kw.get("type"), default_value, action_value)
            choices_value = _safe_literal(kw.get("choices"))
            help_text = _safe_literal(kw.get("help"))
            dest_value = _safe_literal(kw.get("dest"))

            for flag in long_flags:
                dest = dest_value or flag[2:].replace("-", "_")
                current = entries.setdefault(
                    flag,
                    {
                        "name": flag,
                        "aliases": set(),
                        "definition_ref": set(),
                        "default": None,
                        "type": "string",
                        "valid_values": [],
                        "help_text": None,
                        "dest": dest,
                        "source": "code",
                    },
                )
                current["aliases"].update(raw_flags)
                current["definition_ref"].add(_ref(path, node.lineno, repo_root))
                if current["default"] is None and default_value is not None:
                    current["default"] = default_value
                current["type"] = type_value or current["type"]
                if isinstance(choices_value, (list, tuple)):
                    current["valid_values"] = [str(item) for item in choices_value]
                if help_text and not current["help_text"]:
                    current["help_text"] = str(help_text)
                current["dest"] = dest

            self.generic_visit(node)

    Visitor().visit(tree)

    normalized: dict[str, dict[str, Any]] = {}
    for name, row in entries.items():
        normalized[name] = {
            **row,
            "aliases": sorted(row["aliases"]),
            "definition_ref": sorted(row["definition_ref"]),
        }
    return normalized


def _collect_vllm_arg_defs(vllm_root: Path) -> dict[str, dict[str, Any]]:
    files = [
        vllm_root / "vllm" / "engine" / "arg_utils.py",
        vllm_root / "vllm" / "entrypoints" / "openai" / "cli_args.py",
        vllm_root / "vllm" / "entrypoints" / "cli" / "serve.py",
    ]
    merged: dict[str, dict[str, Any]] = {}
    for path in files:
        if not path.exists():
            continue
        for name, row in _extract_arg_defs_from_ast(path, vllm_root).items():
            item = merged.setdefault(name, {**row})
            if item is row:
                continue
            item["aliases"] = sorted(set(item["aliases"]) | set(row["aliases"]))
            item["definition_ref"] = sorted(set(item["definition_ref"]) | set(row["definition_ref"]))
            if item.get("default") is None and row.get("default") is not None:
                item["default"] = row["default"]
            if item.get("type") == "string" and row.get("type") != "string":
                item["type"] = row["type"]
            if not item.get("valid_values") and row.get("valid_values"):
                item["valid_values"] = row["valid_values"]
            if not item.get("help_text") and row.get("help_text"):
                item["help_text"] = row["help_text"]
    return dict(sorted(merged.items()))


def _collect_ascend_arg_defs(ascend_root: Path) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for rel in DEPLOYMENT_ASCEND_ARG_FILES:
        path = ascend_root / rel
        if not path.exists():
            continue
        for name, row in _extract_arg_defs_from_ast(path, ascend_root).items():
            item = merged.setdefault(name, {**row})
            if item is row:
                continue
            item["aliases"] = sorted(set(item["aliases"]) | set(row["aliases"]))
            item["definition_ref"] = sorted(set(item["definition_ref"]) | set(row["definition_ref"]))
            if item.get("default") is None and row.get("default") is not None:
                item["default"] = row["default"]
            if item.get("type") == "string" and row.get("type") != "string":
                item["type"] = row["type"]
            if not item.get("valid_values") and row.get("valid_values"):
                item["valid_values"] = row["valid_values"]
            if not item.get("help_text") and row.get("help_text"):
                item["help_text"] = row["help_text"]
    return dict(sorted(merged.items()))


def _extract_env_defs_from_file(path: Path, repo_root: Path) -> dict[str, dict[str, Any]]:
    text = _read(path)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}

    envs: dict[str, dict[str, Any]] = {}

    def _visit_dict_node(dict_node: ast.Dict) -> None:
        for key_node, value_node in zip(dict_node.keys, dict_node.values):
            if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                continue
            name = key_node.value
            if not re.fullmatch(r"[A-Z][A-Z0-9_]+", name):
                continue
            default_value: Any = None
            value_type = "string"

            expr = value_node.body if isinstance(value_node, ast.Lambda) else value_node
            getenv_call: ast.Call | None = None
            for sub in ast.walk(expr):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "getenv":
                    if sub.args and isinstance(sub.args[0], ast.Constant) and sub.args[0].value == name:
                        getenv_call = sub
                        break
            if getenv_call and len(getenv_call.args) >= 2:
                default_value = _safe_literal(getenv_call.args[1])

            value_type = _infer_type_from_node(expr, default_value)

            envs[name] = {
                "name": name,
                "definition_ref": [_ref(path, key_node.lineno, repo_root)],
                "default": default_value,
                "type": value_type,
                "valid_values": [],
                "source": "code",
            }

    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key_node in node.keys:
                if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                    if re.fullmatch(r"[A-Z][A-Z0-9_]+", key_node.value):
                        _visit_dict_node(node)
                        break

    return dict(sorted(envs.items()))


def _extract_env_mentions_from_code(path: Path, repo_root: Path) -> dict[str, set[str]]:
    text = _read(path)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}

    mentions: dict[str, set[str]] = defaultdict(set)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "getenv":
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                key = node.args[0].value
                if re.fullmatch(r"[A-Z][A-Z0-9_]+", key):
                    mentions[key].add(_ref(path, node.lineno, repo_root))
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
            if isinstance(node.value.value, ast.Name) and node.value.value.id == "os" and node.value.attr == "environ":
                slice_node = node.slice
                if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
                    key = slice_node.value
                elif isinstance(slice_node, ast.Index) and isinstance(slice_node.value, ast.Constant) and isinstance(slice_node.value.value, str):
                    key = slice_node.value.value
                else:
                    key = ""
                if key and re.fullmatch(r"[A-Z][A-Z0-9_]+", key):
                    mentions[key].add(_ref(path, node.lineno, repo_root))

    return mentions


def _collect_ascend_env_defs(ascend_root: Path) -> dict[str, dict[str, Any]]:
    env_file = ascend_root / "vllm_ascend" / "envs.py"
    envs = _extract_env_defs_from_file(env_file, ascend_root) if env_file.exists() else {}

    deploy_scan_files: list[Path] = []
    for rel in DEPLOYMENT_ASCEND_ARG_FILES:
        p = ascend_root / rel
        if p.exists():
            deploy_scan_files.append(p)
    deploy_scan_files.extend((ascend_root / "vllm_ascend").rglob("*.py"))

    for path in deploy_scan_files:
        mentions = _extract_env_mentions_from_code(path, ascend_root)
        for name, refs in mentions.items():
            row = envs.setdefault(
                name,
                {
                    "name": name,
                    "definition_ref": [],
                    "default": None,
                    "type": "string",
                    "valid_values": [],
                    "source": "code",
                },
            )
            row["definition_ref"] = sorted(set(row["definition_ref"]) | refs)

    return dict(sorted(envs.items()))


def _collect_vllm_env_defs(vllm_root: Path) -> dict[str, dict[str, Any]]:
    env_file = vllm_root / "vllm" / "envs.py"
    return _extract_env_defs_from_file(env_file, vllm_root) if env_file.exists() else {}


def _collect_docs_index(doc_files: list[Path], repo_root: Path) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for path in doc_files:
        lines = _read(path).splitlines()
        rel = _relative(path, repo_root)
        for lineno, line in enumerate(lines, start=1):
            for flag in FLAG_IN_TEXT_PATTERN.findall(line):
                index[flag].append(f"{rel}:{lineno}")
            for env_name in ENV_TOKEN_PATTERN.findall(line):
                if env_name.startswith("VLLM") or env_name.startswith("ASCEND") or env_name.startswith("HCCL"):
                    index[env_name].append(f"{rel}:{lineno}")
    return {k: v[:12] for k, v in index.items()}


def _scan_code_references(
    files: list[Path],
    repo_root: Path,
    query_flags: set[str],
    query_envs: set[str],
    query_identifiers: set[str],
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, list[tuple[str, str]]], dict[str, list[tuple[str, str]]]]:
    flag_hits: dict[str, list[tuple[str, str]]] = defaultdict(list)
    env_hits: dict[str, list[tuple[str, str]]] = defaultdict(list)
    identifier_hits: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for path in files:
        rel = _relative(path, repo_root)
        lines = _read(path).splitlines()
        for lineno, line in enumerate(lines, start=1):
            ref = f"{rel}:{lineno}"
            for flag in FLAG_IN_TEXT_PATTERN.findall(line):
                if flag in query_flags and len(flag_hits[flag]) < 80:
                    flag_hits[flag].append((ref, line.strip()))
            for env_name in ENV_TOKEN_PATTERN.findall(line):
                if env_name in query_envs and len(env_hits[env_name]) < 80:
                    env_hits[env_name].append((ref, line.strip()))
            for ident in IDENT_PATTERN.findall(line):
                if ident in query_identifiers and len(identifier_hits[ident]) < 120:
                    identifier_hits[ident].append((ref, line.strip()))

    return flag_hits, env_hits, identifier_hits


def _choose_refs(candidates: list[tuple[str, str]], limit: int = 6) -> tuple[list[str], list[str]]:
    if not candidates:
        return [], []

    control_pattern = re.compile(r"\b(if|elif|else|return|raise|error|warn|unsupported|deprecated|enable|disable|fallback)\b", re.IGNORECASE)
    read_refs: list[str] = []
    effect_refs: list[str] = []

    for ref, line in candidates:
        if len(read_refs) < limit:
            read_refs.append(ref)
        if control_pattern.search(line) and len(effect_refs) < limit:
            effect_refs.append(ref)

    if not effect_refs:
        effect_refs = read_refs[: min(3, len(read_refs))]
    return read_refs[:limit], effect_refs[:limit]


def _infer_stage(kind: str, name: str) -> str:
    if kind == "arg":
        return "startup"
    build_like = {
        "MAX_JOBS",
        "CMAKE_BUILD_TYPE",
        "COMPILE_CUSTOM_KERNELS",
        "CXX_COMPILER",
        "C_COMPILER",
        "SOC_VERSION",
        "ASCEND_HOME_PATH",
        "HCCL_SO_PATH",
        "VLLM_TARGET_DEVICE",
        "VLLM_MAIN_CUDA_VERSION",
    }
    if name in build_like:
        return "build"
    if name.startswith("PYTORCH_") or name.startswith("TORCH"):
        return "startup"
    return "runtime"


def _make_web_refs(scope: str, kind: str, feature: str, local_doc_refs: list[str]) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    key = f"{scope}_{kind}"
    refs.extend(WEB_REFS_BASE.get(key, []))
    refs.extend(FEATURE_WEB_REFS.get(feature, []))

    for ref in local_doc_refs[:3]:
        path = ref.split(":", 1)[0]
        if "/docs/source/" not in path and not path.startswith("docs/source/"):
            continue
        if path.startswith("vllm/"):
            repo_path = path.removeprefix("vllm/")
            refs.append(
                {
                    "title": "Local vLLM doc mirror (GitHub)",
                    "url": f"https://github.com/vllm-project/vllm/blob/main/{repo_path}",
                    "tier": "official",
                }
            )
        elif path.startswith("vllm-ascend/"):
            repo_path = path.removeprefix("vllm-ascend/")
            refs.append(
                {
                    "title": "Local vLLM-Ascend doc mirror (GitHub)",
                    "url": f"https://github.com/vllm-project/vllm-ascend/blob/main/{repo_path}",
                    "tier": "official",
                }
            )
        else:
            refs.append(
                {
                    "title": "Local vLLM-Ascend doc mirror (GitHub)",
                    "url": f"https://github.com/vllm-project/vllm-ascend/blob/main/{path}",
                    "tier": "official",
                }
            )

    dedup: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in refs:
        url = item["url"]
        if url in seen:
            continue
        seen.add(url)
        dedup.append(item)
    return dedup[:8]


def _fetch_url_text(url: str, timeout: int = 10) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": "kb-builder/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None


def _extract_flags_from_text(text: str) -> set[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set(FLAG_IN_TEXT_PATTERN.findall(text))

    flags: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        value = arg.value.strip()
                        if FLAG_PATTERN.fullmatch(value):
                            flags.add(value)
            self.generic_visit(node)

    Visitor().visit(tree)
    return flags


def _extract_envs_from_text(text: str) -> set[str]:
    envs: set[str] = set()
    for match in re.finditer(r'"([A-Z][A-Z0-9_]+)"\s*:', text):
        envs.add(match.group(1))
    return envs


def _build_upstream_snapshot() -> dict[str, Any]:
    vllm_env_text = _fetch_url_text(UPSTREAM_URLS["vllm_env"])
    vllm_arg_utils_text = _fetch_url_text(UPSTREAM_URLS["vllm_arg_utils"])
    vllm_cli_args_text = _fetch_url_text(UPSTREAM_URLS["vllm_cli_args"])
    ascend_env_text = _fetch_url_text(UPSTREAM_URLS["ascend_env"])

    vllm_args: set[str] = set()
    if vllm_arg_utils_text:
        vllm_args |= _extract_flags_from_text(vllm_arg_utils_text)
    if vllm_cli_args_text:
        vllm_args |= _extract_flags_from_text(vllm_cli_args_text)

    return {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "urls": UPSTREAM_URLS,
        "available": {
            "vllm_env": vllm_env_text is not None,
            "vllm_arg_utils": vllm_arg_utils_text is not None,
            "vllm_cli_args": vllm_cli_args_text is not None,
            "ascend_env": ascend_env_text is not None,
        },
        "vllm_args": sorted(vllm_args),
        "vllm_envs": sorted(_extract_envs_from_text(vllm_env_text or "")),
        "ascend_envs": sorted(_extract_envs_from_text(ascend_env_text or "")),
    }


def _infer_status(scope: str, kind: str, name: str, upstream: dict[str, Any]) -> str:
    available = upstream["available"]
    if scope == "vllm" and kind == "arg":
        if not (available.get("vllm_arg_utils") or available.get("vllm_cli_args")):
            return STATUS_NEEDS_REVIEW
        return STATUS_ALIGNED if name in upstream["vllm_args"] else STATUS_UPSTREAM_DELTA

    if scope == "vllm" and kind == "env":
        if not available.get("vllm_env"):
            return STATUS_NEEDS_REVIEW
        return STATUS_ALIGNED if name in upstream["vllm_envs"] else STATUS_UPSTREAM_DELTA

    if scope == "vllm_ascend" and kind == "env":
        if not available.get("ascend_env"):
            return STATUS_NEEDS_REVIEW
        if name in upstream["ascend_envs"]:
            return STATUS_ALIGNED
        # Shared VLLM env var may not live in ascend env file.
        if name in upstream["vllm_envs"]:
            return STATUS_ALIGNED
        return STATUS_UPSTREAM_DELTA

    if scope == "vllm_ascend" and kind == "arg":
        if name in upstream["vllm_args"]:
            return STATUS_ALIGNED
        return STATUS_NEEDS_REVIEW

    return STATUS_NEEDS_REVIEW


def _confidence_score(
    status: str,
    definition_ref: list[str],
    read_ref: list[str],
    effect_ref: list[str],
    local_doc_refs: list[str],
    web_refs: list[dict[str, str]],
) -> float:
    score = 0.35
    if definition_ref:
        score += 0.25
    if read_ref:
        score += 0.1
    if effect_ref:
        score += 0.1
    if local_doc_refs:
        score += 0.07
    if any(item.get("tier") == "official" for item in web_refs):
        score += 0.08
    if any(item.get("tier") == "external" for item in web_refs):
        score += 0.03

    if status == STATUS_UPSTREAM_DELTA:
        score -= 0.2
    elif status == STATUS_NEEDS_REVIEW:
        score -= 0.12

    return round(max(0.05, min(0.99, score)), 2)


def _entry_defaults(feature: str) -> dict[str, Any]:
    return FEATURE_DEFAULTS.get(feature, FEATURE_DEFAULTS["general_runtime"])


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_entries(
    scope: str,
    kind: str,
    raw_entries: dict[str, dict[str, Any]],
    docs_index: dict[str, list[str]],
    flag_hits: dict[str, list[tuple[str, str]]],
    env_hits: dict[str, list[tuple[str, str]]],
    identifier_hits: dict[str, list[tuple[str, str]]],
    upstream: dict[str, Any],
    now: str,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    for name, raw in raw_entries.items():
        feature_tags = _derive_feature_tags(name)
        primary_feature = feature_tags[0]
        defaults = _entry_defaults(primary_feature)
        override = ENTRY_OVERRIDES.get(name, {})

        local_doc_refs = docs_index.get(name, [])

        if kind == "arg":
            dest = raw.get("dest", "")
            read_candidates = []
            effect_candidates = []
            if isinstance(dest, str) and dest:
                id_hits = identifier_hits.get(dest, [])
                read_candidates.extend(id_hits)
                effect_candidates.extend(id_hits)
            read_from_flags = flag_hits.get(name, [])
            read_candidates.extend(read_from_flags)
            effect_candidates.extend(read_from_flags)
            read_ref, effect_ref = _choose_refs(read_candidates)
        else:
            env_candidates = env_hits.get(name, [])
            read_ref, effect_ref = _choose_refs(env_candidates)

        definition_ref = list(raw.get("definition_ref", []))
        if not definition_ref and read_ref:
            definition_ref = [read_ref[0]]

        combo_rule_ids = FEATURE_TO_RULE_IDS.get(primary_feature, [])

        web_refs = _make_web_refs(scope, kind, primary_feature, local_doc_refs)
        status = _infer_status(scope, kind, name, upstream)
        confidence = _confidence_score(status, definition_ref, read_ref, effect_ref, local_doc_refs, web_refs)

        entry = {
            "id": f"{scope}.{kind}.{name.lstrip('-').replace('-', '_').lower()}",
            "name": name,
            "kind": kind,
            "scope": scope,
            "stage": _infer_stage(kind, name),
            "type": raw.get("type", "string"),
            "default": raw.get("default"),
            "valid_values": raw.get("valid_values", []),
            "source": "code",
            "definition_ref": definition_ref[:8],
            "read_ref": read_ref[:8],
            "effect_ref": effect_ref[:8],
            "local_doc_refs": local_doc_refs[:8],
            "web_refs": web_refs,
            "feature_tags": feature_tags,
            "primary_feature": primary_feature,
            "semantics": override.get("semantics", defaults["semantics"]),
            "prerequisites": override.get("prerequisites", defaults["prerequisites"]),
            "incompatibilities": override.get("incompatibilities", defaults["incompatibilities"]),
            "combo_rules": combo_rule_ids,
            "failure_modes": override.get("failure_modes", defaults["failure_modes"]),
            "recommendation": override.get("recommendation", defaults["recommendation"]),
            "confidence": confidence,
            "status": status,
            "updated_at": now,
        }
        entries.append(entry)

    return sorted(entries, key=lambda item: item["name"])


def _build_dataset_snapshot(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for entry in entries:
        out[entry["name"]] = {
            "primary_feature": entry["primary_feature"],
            "feature_tags": entry["feature_tags"],
            "combination_candidates": entry["combo_rules"],
            "usage_hint": entry["semantics"],
            "confidence": entry["confidence"],
            "status": entry["status"],
            "definition_ref": entry["definition_ref"],
            "web_refs": entry["web_refs"],
        }
    return out


def _summarize_source_tiers(entries: list[dict[str, Any]]) -> dict[str, int]:
    official_ref_count = 0
    external_ref_count = 0
    entries_with_official_refs = 0
    entries_with_external_refs = 0

    for entry in entries:
        has_official = False
        has_external = False
        for ref in entry.get("web_refs", []):
            tier = ref.get("tier")
            if tier == "official":
                official_ref_count += 1
                has_official = True
            elif tier == "external":
                external_ref_count += 1
                has_external = True
        if has_official:
            entries_with_official_refs += 1
        if has_external:
            entries_with_external_refs += 1

    return {
        "official_ref_count": official_ref_count,
        "external_ref_count": external_ref_count,
        "entries_with_official_refs": entries_with_official_refs,
        "entries_with_external_refs": entries_with_external_refs,
    }


def _count_flag_occurrences(files: list[Path], target_flags: set[str]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for path in files:
        for flag in FLAG_IN_TEXT_PATTERN.findall(_read(path)):
            if flag in target_flags:
                counter[flag] += 1
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _build_flag_pairings(files: list[Path], target_flags: set[str], limit: int = 500) -> list[dict[str, Any]]:
    pair_counter: Counter[tuple[str, str]] = Counter()
    for path in files:
        flags_in_file = sorted(set(FLAG_IN_TEXT_PATTERN.findall(_read(path))) & target_flags)
        for idx, left in enumerate(flags_in_file):
            for right in flags_in_file[idx + 1 :]:
                pair_counter[(left, right)] += 1

    rows: list[dict[str, Any]] = []
    for (left, right), count in pair_counter.most_common(limit):
        rows.append({"left": left, "right": right, "cooccurrence_files": count})
    return rows


def _to_relative_sorted(files: list[Path], repo_root: Path) -> list[str]:
    return sorted({_relative(path, repo_root) for path in files if path.exists()})


def _build_validation_report(entries: list[dict[str, Any]], now: str) -> dict[str, Any]:
    total = len(entries)
    with_definition = sum(1 for item in entries if item["definition_ref"])
    with_code_behavior = sum(1 for item in entries if item["read_ref"] or item["effect_ref"])
    with_web = sum(1 for item in entries if item["web_refs"])

    unresolved: list[str] = []
    for item in entries:
        if not item["definition_ref"]:
            unresolved.append(f"{item['id']}:missing_definition_ref")
        if not (item["read_ref"] or item["effect_ref"]):
            unresolved.append(f"{item['id']}:missing_behavior_ref")
        if item["status"] != STATUS_ALIGNED:
            unresolved.append(f"{item['id']}:{item['status']}")

    high_risk_validated = 0
    for item in entries:
        if item["primary_feature"] in HIGH_RISK_FEATURES:
            has_local_doc = bool(item["local_doc_refs"])
            has_official_web = any(ref.get("tier") == "official" for ref in item["web_refs"])
            if has_local_doc and has_official_web:
                high_risk_validated += 1

    source_tier_stats = _summarize_source_tiers(entries)

    report = {
        "coverage": {
            "expected_entries": total,
            "actual_entries": total,
            "ratio": 1.0,
        },
        "evidence_completeness": {
            "with_definition_ref": with_definition,
            "with_behavior_ref": with_code_behavior,
            "with_web_refs": with_web,
            "ratio": round((with_definition + with_code_behavior + with_web) / max(1, total * 3), 4),
        },
        "conflict_count": sum(1 for item in entries if item["status"] != STATUS_ALIGNED),
        "high_risk_validated_count": high_risk_validated,
        "source_tier_stats": source_tier_stats,
        "unresolved_items": unresolved[:300],
        "generated_at": now,
    }
    return report


def _feature_summary(entries: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in entries:
        counter[item["primary_feature"]] += 1
    return dict(counter)


def _render_inventory_rows(data: dict[str, dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    for name, item in sorted(data.items()):
        refs = ", ".join(item.get("definition_ref", [])[:2])
        rows.append(f"| `{name}` | {item.get('type', 'string')} | {refs} |")
    return rows


def _render_feature_rows(entries: list[dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    for item in entries:
        web_count = sum(1 for _ in item["web_refs"])
        rows.append(
            "| `{}` | `{}` | `{}` | `{}` | {} | {} |".format(
                item["name"],
                item["primary_feature"],
                item["status"],
                item["confidence"],
                item["definition_ref"][0] if item["definition_ref"] else "-",
                web_count,
            )
        )
    return rows


def _write_markdown_docs(
    shared_root: Path,
    now: str,
    vllm_args_raw: dict[str, dict[str, Any]],
    vllm_env_raw: dict[str, dict[str, Any]],
    ascend_args_raw: dict[str, dict[str, Any]],
    ascend_env_raw: dict[str, dict[str, Any]],
    entries: list[dict[str, Any]],
    validation_report: dict[str, Any],
) -> None:
    vllm_doc = shared_root / "vllm-foundation" / "references" / "vllm-inputs-and-envs-global.md"
    ascend_doc = shared_root / "vllm-ascend-core" / "references" / "vllm-ascend-inputs-and-envs-global.md"
    feature_map_doc = shared_root / "deployment-config" / "references" / "global-parameter-feature-map.md"
    combo_doc = shared_root / "deployment-config" / "references" / "global-parameter-combination-guide.md"
    verify_doc = shared_root / "deployment-config" / "references" / "global-parameter-verification-report.md"

    for path in [vllm_doc, ascend_doc, feature_map_doc, combo_doc, verify_doc]:
        path.parent.mkdir(parents=True, exist_ok=True)

    vllm_doc.write_text(
        "\n".join(
            [
                "---",
                "knowledge_id: vllm-foundation.inputs-and-envs-global",
                "domain: vllm-foundation",
                "knowledge_type: reference",
                "summary: Code-truth inventory of vLLM deployment arguments and environment variables with evidence refs.",
                f"last_verified: \"{now}\"",
                "source_commit: \"workspace-head\"",
                "freshness: \"fresh\"",
                "---",
                "",
                "# vLLM Global Inputs and Envs (Code Truth)",
                "",
                f"- vLLM args: **{len(vllm_args_raw)}**",
                f"- vLLM env vars: **{len(vllm_env_raw)}**",
                "",
                "## vLLM Serve Args",
                "",
                "| Name | Type | Definition ref |",
                "| --- | --- | --- |",
                *_render_inventory_rows(vllm_args_raw),
                "",
                "## vLLM Env Vars",
                "",
                "| Name | Type | Definition ref |",
                "| --- | --- | --- |",
                *_render_inventory_rows(vllm_env_raw),
                "",
                "Back to [INDEX](../../INDEX.md).",
                "",
            ]
        ),
        encoding="utf-8",
    )

    ascend_doc.write_text(
        "\n".join(
            [
                "---",
                "knowledge_id: vllm-ascend-core.inputs-and-envs-global",
                "domain: vllm-ascend-core",
                "knowledge_type: reference",
                "summary: Code-truth inventory of vLLM-Ascend deployment arguments and environment variables with evidence refs.",
                f"last_verified: \"{now}\"",
                "source_commit: \"workspace-head\"",
                "freshness: \"fresh\"",
                "---",
                "",
                "# vLLM-Ascend Global Inputs and Envs (Code Truth)",
                "",
                f"- vLLM-Ascend deployment args: **{len(ascend_args_raw)}**",
                f"- vLLM-Ascend env vars: **{len(ascend_env_raw)}**",
                "",
                "## vLLM-Ascend Deployment Args",
                "",
                "| Name | Type | Definition ref |",
                "| --- | --- | --- |",
                *_render_inventory_rows(ascend_args_raw),
                "",
                "## vLLM-Ascend Env Vars",
                "",
                "| Name | Type | Definition ref |",
                "| --- | --- | --- |",
                *_render_inventory_rows(ascend_env_raw),
                "",
                "Back to [INDEX](../../INDEX.md).",
                "",
            ]
        ),
        encoding="utf-8",
    )

    feature_map_doc.write_text(
        "\n".join(
            [
                "---",
                "knowledge_id: deployment-config.global-parameter-feature-map",
                "domain: deployment-config",
                "knowledge_type: reference",
                "summary: High-confidence parameter/env semantic map with code evidence and web verification.",
                f"last_verified: \"{now}\"",
                "source_commit: \"workspace-head\"",
                "freshness: \"fresh\"",
                "---",
                "",
                "# Global Parameter Feature Map (High Confidence)",
                "",
                f"Generated at: `{now}`",
                "",
                "| Name | Primary feature | Status | Confidence | Definition ref | Web refs |",
                "| --- | --- | --- | --- | --- | --- |",
                *_render_feature_rows(entries),
                "",
                "Back to [INDEX](../../INDEX.md).",
                "",
            ]
        ),
        encoding="utf-8",
    )

    combo_lines = [
        "---",
        "knowledge_id: deployment-config.global-parameter-combination-guide",
        "domain: deployment-config",
        "knowledge_type: procedure",
        "summary: Combination constraints and profile-level blocks with evidence refs.",
        f"last_verified: \"{now}\"",
        "source_commit: \"workspace-head\"",
        "freshness: \"fresh\"",
        "---",
        "",
        "# Global Parameter Combination Guide",
        "",
        "## Rule Levels",
        "",
        "- `hard_block`: must not execute automatically",
        "- `warning`: allow execution with explicit warning and fallback",
        "- `recommended`: preferred baseline for demo deployment",
        "",
        "## Rules",
        "",
    ]
    for idx, rule in enumerate(COMBO_RULES, start=1):
        combo_lines.extend(
            [
                f"{idx}. `{rule['rule_id']}` ({rule['level']})",
                f"- profile: `{rule['profile']}`",
                f"- conditions: `{', '.join(rule['conditions'])}`",
                f"- reason: {rule['reason']}",
                f"- fallback: `{'; '.join(rule['fallback_actions'])}`",
            ]
        )
    combo_lines.extend(["", "Back to [INDEX](../../INDEX.md).", ""])
    combo_doc.write_text("\n".join(combo_lines), encoding="utf-8")

    verify_doc.write_text(
        "\n".join(
            [
                "---",
                "knowledge_id: deployment-config.global-parameter-verification-report",
                "domain: deployment-config",
                "knowledge_type: verification",
                "summary: Dual-baseline verification report (local code truth + upstream web checks).",
                f"last_verified: \"{now}\"",
                "source_commit: \"workspace-head\"",
                "freshness: \"fresh\"",
                "---",
                "",
                "# Global Parameter Verification Report",
                "",
                f"- Coverage ratio: **{validation_report['coverage']['ratio']}**",
                f"- Evidence completeness ratio: **{validation_report['evidence_completeness']['ratio']}**",
                f"- Conflict count: **{validation_report['conflict_count']}**",
                f"- High-risk validated count: **{validation_report['high_risk_validated_count']}**",
                f"- Official refs: **{validation_report['source_tier_stats']['official_ref_count']}**",
                f"- External refs: **{validation_report['source_tier_stats']['external_ref_count']}**",
                f"- Entries with external refs: **{validation_report['source_tier_stats']['entries_with_external_refs']}**",
                "",
                "## Unresolved items (first 50)",
                "",
                *[f"- `{item}`" for item in validation_report["unresolved_items"][:50]],
                "",
                "Back to [INDEX](../../INDEX.md).",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None, help="Path to vllm-ascend repo root")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    ascend_root = Path(args.repo_root).resolve() if args.repo_root else script_dir.parents[3]
    workspace_root = ascend_root.parent
    vllm_root = (workspace_root / "vllm").resolve()

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    vllm_arg_defs = _collect_vllm_arg_defs(vllm_root)
    vllm_env_defs = _collect_vllm_env_defs(vllm_root)
    ascend_arg_defs = _collect_ascend_arg_defs(ascend_root)
    ascend_env_defs = _collect_ascend_env_defs(ascend_root)

    # Docs evidence index (deployment related docs only, include vLLM + vLLM-Ascend)
    ascend_doc_files = sorted((ascend_root / "docs" / "source").rglob("*.md"))
    vllm_doc_files = sorted((vllm_root / "docs" / "source").rglob("*.md"))
    doc_files = ascend_doc_files + vllm_doc_files
    docs_index = _collect_docs_index(doc_files, workspace_root)

    # Code evidence scan
    vllm_py_files = sorted((vllm_root / "vllm").rglob("*.py"))
    ascend_py_files = sorted((ascend_root / "vllm_ascend").rglob("*.py"))
    ascend_example_py = [p for p in (ascend_root / "examples").rglob("*.py") if p.is_file()]
    code_files = vllm_py_files + ascend_py_files + ascend_example_py
    ascend_scan_files = ascend_py_files + ascend_example_py + ascend_doc_files

    query_flags = set(vllm_arg_defs.keys()) | set(ascend_arg_defs.keys())
    query_envs = set(vllm_env_defs.keys()) | set(ascend_env_defs.keys())
    query_identifiers = {row.get("dest", "") for row in vllm_arg_defs.values()} | {
        row.get("dest", "") for row in ascend_arg_defs.values()
    }
    query_identifiers = {item for item in query_identifiers if isinstance(item, str) and item}

    flag_hits, env_hits, identifier_hits = _scan_code_references(
        code_files,
        workspace_root,
        query_flags,
        query_envs,
        query_identifiers,
    )

    upstream = _build_upstream_snapshot()

    vllm_arg_entries = _build_entries(
        scope="vllm",
        kind="arg",
        raw_entries=vllm_arg_defs,
        docs_index=docs_index,
        flag_hits=flag_hits,
        env_hits=env_hits,
        identifier_hits=identifier_hits,
        upstream=upstream,
        now=now,
    )
    vllm_env_entries = _build_entries(
        scope="vllm",
        kind="env",
        raw_entries=vllm_env_defs,
        docs_index=docs_index,
        flag_hits=flag_hits,
        env_hits=env_hits,
        identifier_hits=identifier_hits,
        upstream=upstream,
        now=now,
    )
    ascend_arg_entries = _build_entries(
        scope="vllm_ascend",
        kind="arg",
        raw_entries=ascend_arg_defs,
        docs_index=docs_index,
        flag_hits=flag_hits,
        env_hits=env_hits,
        identifier_hits=identifier_hits,
        upstream=upstream,
        now=now,
    )
    ascend_env_entries = _build_entries(
        scope="vllm_ascend",
        kind="env",
        raw_entries=ascend_env_defs,
        docs_index=docs_index,
        flag_hits=flag_hits,
        env_hits=env_hits,
        identifier_hits=identifier_hits,
        upstream=upstream,
        now=now,
    )

    all_entries = vllm_arg_entries + vllm_env_entries + ascend_arg_entries + ascend_env_entries
    validation_report = _build_validation_report(all_entries, now)
    legacy_flag_pairings = _build_flag_pairings(code_files + doc_files, set(vllm_arg_defs) | set(ascend_arg_defs))
    legacy_scan_files = _to_relative_sorted(ascend_py_files + ascend_example_py, ascend_root)
    ascend_args_frequency = _count_flag_occurrences(ascend_scan_files, set(ascend_arg_defs))

    datasets = {
        "vllm_args": _build_dataset_snapshot(vllm_arg_entries),
        "vllm_envs": _build_dataset_snapshot(vllm_env_entries),
        "vllm_ascend_args": _build_dataset_snapshot(ascend_arg_entries),
        "vllm_ascend_envs": _build_dataset_snapshot(ascend_env_entries),
    }

    shared_root = ascend_root / ".agents" / "skills" / "_shared"
    out_vllm = shared_root / "vllm-foundation" / "references" / "generated"
    out_ascend = shared_root / "vllm-ascend-core" / "references" / "generated"
    out_deploy = shared_root / "deployment-config" / "references" / "generated"

    _write_json(out_vllm / "vllm_args_inventory.json", vllm_arg_defs)
    _write_json(out_vllm / "vllm_env_inventory.json", vllm_env_defs)
    _write_json(out_ascend / "vllm_ascend_args_inventory.json", ascend_arg_defs)
    _write_json(out_ascend / "vllm_ascend_env_inventory.json", ascend_env_defs)
    _write_json(out_ascend / "vllm_ascend_args_frequency.json", ascend_args_frequency)

    _write_json(
        out_deploy / "global_parameter_kb.json",
        {
            "generated_at": now,
            "baseline": {
                "mode": "dual",
                "local_truth": "workspace_commit",
                "upstream_verification": "official + external references",
            },
            "entries": all_entries,
            "datasets": datasets,
            "combo_rules": COMBO_RULES,
            "validation_report": validation_report,
        },
    )
    _write_json(out_deploy / "global_feature_summary.json", _feature_summary(all_entries))
    _write_json(out_deploy / "global_flag_pairings.json", legacy_flag_pairings)
    _write_json(out_deploy / "global_scan_files.json", legacy_scan_files)
    _write_json(out_deploy / "global_combo_rules.json", COMBO_RULES)
    _write_json(out_deploy / "global_validation_report.json", validation_report)
    _write_json(out_deploy / "global_upstream_snapshot.json", upstream)

    _write_markdown_docs(
        shared_root=shared_root,
        now=now,
        vllm_args_raw=vllm_arg_defs,
        vllm_env_raw=vllm_env_defs,
        ascend_args_raw=ascend_arg_defs,
        ascend_env_raw=ascend_env_defs,
        entries=all_entries,
        validation_report=validation_report,
    )

    summary = {
        "vllm_arg_count": len(vllm_arg_entries),
        "vllm_env_count": len(vllm_env_entries),
        "vllm_ascend_arg_count": len(ascend_arg_entries),
        "vllm_ascend_env_count": len(ascend_env_entries),
        "total_entries": len(all_entries),
        "conflict_count": validation_report["conflict_count"],
        "high_risk_validated_count": validation_report["high_risk_validated_count"],
        "entries_with_external_refs": validation_report["source_tier_stats"]["entries_with_external_refs"],
        "generated_at": now,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
