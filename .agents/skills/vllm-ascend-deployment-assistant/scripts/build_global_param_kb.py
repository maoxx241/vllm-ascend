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

VALUE_SEMANTICS_OVERRIDES: dict[str, dict[str, Any]] = {
    "--quantization": {
        "value_shape": "enum",
        "accepted_values": [
            "awq",
            "fp8",
            "ptpc_fp8",
            "fbgemm_fp8",
            "fp_quant",
            "modelopt",
            "modelopt_fp4",
            "modelopt_mxfp8",
            "gguf",
            "gptq_marlin",
            "awq_marlin",
            "gptq",
            "compressed-tensors",
            "bitsandbytes",
            "experts_int8",
            "quark",
            "moe_wna16",
            "torchao",
            "inc",
            "mxfp4",
            "petit_nvfp4",
            "cpu_awq",
            "custom_registered_method",
        ],
        "default_behavior": "未显式设置时优先读取模型 quantization_config；若模型未声明则按非量化路径并由 --dtype 决定精度。",
        "value_effects": [
            "设置为具体后端: 选择对应量化权重加载/算子实现路径。",
            "未设置: 自动从模型配置推断或回落到非量化。",
        ],
        "constraints": [
            "若显式 --quantization 与模型 quantization_config 推断不一致，会直接报错。",
            "未知量化方法会报错（必须在 QUANTIZATION_METHODS 内或已注册自定义方法）。",
            "若方法处于 deprecated 列表且未开启 --allow-deprecated-quantization，会报错。",
        ],
        "combo_effects": [
            "与 --dtype 联动：量化关闭时主要由 dtype 决定权重/激活精度。",
            "与模型 profile 绑定：如 qwen3-32b-w8a8 不应叠加 int4 路径（由组合规则 hard block）。",
        ],
        "performance_tradeoffs": [
            "通常可降低显存并提升吞吐，但可能引入精度退化或后端兼容问题。",
        ],
        "failure_signals": [
            "ValueError: Unknown quantization method",
            "ValueError: Quantization method specified in model config does not match argument",
            "ValueError: quantization method is deprecated ... set --allow-deprecated-quantization",
        ],
        "evidence_refs": [
            "vllm/config/model.py:861",
            "vllm/config/model.py:925",
            "vllm/config/model.py:933",
            "vllm/config/model.py:942",
            "vllm/model_executor/layers/quantization/__init__.py:12",
            "vllm/model_executor/layers/quantization/__init__.py:38",
        ],
        "completion_status": "done",
    },
    "--allow-deprecated-quantization": {
        "value_shape": "binary_toggle",
        "accepted_values": ["enabled", "disabled"],
        "default_behavior": "默认 disabled（False），遇到废弃量化方法时直接阻断。",
        "value_effects": [
            "enabled: 允许使用废弃量化方法并给出 warning。",
            "disabled: 使用废弃量化方法时抛出 ValueError 阻断启动。",
        ],
        "constraints": ["仅在量化方法命中 deprecated 列表时生效。"],
        "combo_effects": ["与 --quantization 联动决定 deprecated 后端是告警继续还是硬失败。"],
        "performance_tradeoffs": ["开启兼容模式可提高可用性，但会增加未来升级风险。"],
        "failure_signals": [
            "ValueError: quantization method is deprecated ... set --allow-deprecated-quantization",
        ],
        "evidence_refs": [
            "vllm/config/model.py:192",
            "vllm/config/model.py:942",
            "vllm/config/model.py:950",
            "vllm/model_executor/layers/quantization/__init__.py:38",
        ],
        "completion_status": "done",
    },
    "--dtype": {
        "value_shape": "enum",
        "accepted_values": ["auto", "half", "float16", "bfloat16", "float", "float32"],
        "default_behavior": "默认 auto：根据模型 config dtype 与平台支持自动决策。",
        "value_effects": [
            "auto: 自动在可支持精度中选择（常见为 fp16/bf16）。",
            "float32/float: 提高数值稳定性但通常降低吞吐并增加显存。",
            "half/float16/bfloat16: 以吞吐和显存效率为主。",
        ],
        "constraints": [
            "不支持的 dtype 字符串会报错。",
            "部分模型类型禁用 float16（如 gemma2/gemma3/plamo2 等），会报错要求改为 bf16/float32。",
        ],
        "combo_effects": [
            "与 --quantization 联动：量化未启用时，dtype 直接控制权重/激活主精度。",
            "与 --max-model-len/批处理参数联动影响显存峰值与可并发。",
        ],
        "performance_tradeoffs": [
            "低精度通常吞吐更高、显存更省；高精度通常更稳但更慢。",
        ],
        "failure_signals": [
            "ValueError: Unknown dtype",
            "ValueError: For Gemma 2 and 3, float16 is not supported",
        ],
        "evidence_refs": [
            "vllm/config/model.py:133",
            "vllm/config/model.py:1745",
            "vllm/config/model.py:1831",
            "vllm/config/model.py:1855",
            "vllm/config/model.py:1760",
        ],
        "completion_status": "done",
    },
    "--kv-cache-dtype": {
        "value_shape": "enum",
        "accepted_values": ["auto", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc", "fp8_ds_mla"],
        "default_behavior": "默认 auto；在构建 CacheConfig 时会解析为具体 cache dtype。",
        "value_effects": [
            "fp8 系列: 降低 KV cache 占用并可能提升性能，但精度风险更高。",
            "bfloat16: 精度更稳但显存占用更高。",
            "auto: 跟随模型/平台能力自动选择。",
        ],
        "constraints": [
            "部分 dtype 受平台后端限制（如 CUDA/ROCm/Gaudi 差异）。",
            "与 --calculate-kv-scales 联动决定 fp8 KV scale 来源（动态计算或读取权重）。",
        ],
        "combo_effects": [
            "与 max_model_len、gpu_memory_utilization、max_num_batched_tokens 联动决定可用上下文容量。",
        ],
        "performance_tradeoffs": [
            "fp8 可显著减小缓存占用，但需接受潜在精度下降风险。",
        ],
        "failure_signals": ["非法枚举值会在配置解析阶段报错。"],
        "evidence_refs": [
            "vllm/config/cache.py:23",
            "vllm/config/cache.py:59",
            "vllm/config/cache.py:105",
            "vllm/config/cache.py:210",
            "vllm/engine/arg_utils.py:1428",
        ],
        "completion_status": "done",
    },
    "--swap-space": {
        "value_shape": "numeric",
        "accepted_values": ["float >= 0 (GiB per GPU)"],
        "default_behavior": "默认 4 GiB / GPU。",
        "value_effects": [
            "值增大: 提供更多 CPU 侧交换空间，缓解显存压力。",
            "值减小: 降低 CPU 内存占用，但更易触发缓存不足。",
        ],
        "constraints": [
            "总 CPU swap 预算按 tensor_parallel_size 放大。",
            "若 swap 占用 > 70% 总内存将报错，> 40% 给 warning。",
        ],
        "combo_effects": ["与 tensor_parallel_size 联动计算总 CPU 内存占用。"],
        "performance_tradeoffs": [
            "更大 swap 提升可容纳能力，但可能因 CPU<->GPU 交换导致时延增加。",
        ],
        "failure_signals": ["ValueError: Too large swap space."],
        "evidence_refs": [
            "vllm/config/cache.py:57",
            "vllm/config/cache.py:226",
            "vllm/config/cache.py:238",
        ],
        "completion_status": "done",
    },
    "--cpu-offload-gb": {
        "value_shape": "numeric",
        "accepted_values": ["float >= 0 (GiB per GPU)"],
        "default_behavior": "默认 0（不启用 CPU offload）。",
        "value_effects": [
            "值增大: 允许更多参数驻留 CPU，降低显存需求。",
            "值为 0: 不做 offload，路径更简单。",
        ],
        "constraints": ["需要较高带宽 CPU-GPU 互联，否则可能引入明显额外时延。"],
        "combo_effects": [
            "与 gpu_memory_utilization、max_model_len 联动影响是否能加载更大模型。",
        ],
        "performance_tradeoffs": [
            "提高可部署容量，但 token 级前向可能受制于数据搬运带宽。",
        ],
        "failure_signals": ["配置为负值会在参数校验时报错。"],
        "evidence_refs": [
            "vllm/config/cache.py:95",
            "vllm/config/cache.py:96",
            "vllm/engine/arg_utils.py:946",
        ],
        "completion_status": "done",
    },
    "--async-scheduling": {
        "value_shape": "binary_toggle",
        "accepted_values": ["enabled", "disabled"],
        "default_behavior": "未显式设置时，系统会在兼容场景自动开启，不兼容时自动关闭。",
        "value_effects": [
            "enabled: 调度与执行重叠，通常降低空转并提升吞吐。",
            "disabled: 保守同步调度，兼容性更高但并行重叠收益较低。",
        ],
        "constraints": [
            "仅支持 distributed_executor_backend in {mp, uni, external_launcher}",
            "与 disable_padded_drafter_batch=True 不兼容",
            "Mamba prefix cache 模式（mamba_cache_mode != none）下不兼容",
        ],
        "combo_effects": [
            "与部分 speculative decoding 组合会被强制关闭或报错",
            "启用后会影响 DP 同步策略（默认倾向 disable NCCL for DP sync）",
        ],
        "performance_tradeoffs": [
            "通常改善吞吐与时延抖动",
            "不兼容配置会触发硬失败，建议先灰度验证",
        ],
        "failure_signals": [
            "ValueError: async scheduling only supports mp/uni/external_launcher",
            "ValueError: not compatible with disable_padded_drafter_batch=True",
        ],
        "evidence_refs": [
            "vllm/config/vllm.py:618",
            "vllm/config/vllm.py:637",
            "vllm/config/vllm.py:647",
            "vllm/config/scheduler.py:131",
        ],
        "completion_status": "done",
    },
    "--scheduling-policy": {
        "value_shape": "enum",
        "accepted_values": ["fcfs", "priority"],
        "default_behavior": "默认 fcfs。",
        "value_effects": [
            "fcfs: 按到达顺序调度，行为稳定且可预期。",
            "priority: 按优先级调度，同优先级按到达顺序打破平局。",
        ],
        "constraints": ["仅在 v1 scheduler 语义下生效"],
        "combo_effects": ["与请求优先级字段联合生效，不设置优先级时接近 FCFS 行为"],
        "performance_tradeoffs": [
            "priority 可提升高优请求响应，但可能增加低优请求排队",
        ],
        "failure_signals": ["非法值会在参数校验时报错"],
        "evidence_refs": [
            "vllm/config/scheduler.py:22",
            "vllm/config/scheduler.py:101",
            "vllm/docs/usage/v1_guide.md:79",
        ],
        "completion_status": "done",
    },
    "--max-num-batched-tokens": {
        "value_shape": "numeric",
        "accepted_values": ["int >= 1"],
        "default_behavior": "默认 2048（部署侧常按场景覆盖）。",
        "value_effects": [
            "值增大: 单步 token budget 变大，通常吞吐更高、TTFT 可能下降，但激活/显存压力上升。",
            "值减小: 显存压力更低，但长请求可能被切分更多轮次，吞吐下降。",
        ],
        "constraints": [
            "必须 >= max_num_seqs",
            "当关闭 chunked prefill 时，必须 >= max_model_len",
        ],
        "combo_effects": [
            "与 max_num_seqs、max_model_len 共同决定调度上限与排队行为",
            "在 FlashComm1 + PCP 场景可能被对齐为 tp_size*pcp_size 的倍数",
        ],
        "performance_tradeoffs": [
            "增大可提吞吐，但更易触发 OOM 或碎片风险",
            "减小可控内存，但会放大排队/分片开销",
        ],
        "failure_signals": [
            "ValueError: max_num_batched_tokens must be >= max_num_seqs",
            "ValueError: smaller than max_model_len when chunked prefill disabled",
        ],
        "evidence_refs": [
            "vllm/config/scheduler.py:48",
            "vllm/config/scheduler.py:258",
            "vllm/config/scheduler.py:246",
            "vllm-ascend/vllm_ascend/ascend_config.py:76",
            "vllm-ascend/docs/source/tutorials/models/Qwen3-235B-A22B.md:138",
        ],
        "completion_status": "done",
    },
    "--max-num-seqs": {
        "value_shape": "numeric",
        "accepted_values": ["int >= 1"],
        "default_behavior": "默认 128（部署侧常按并发目标覆盖）。",
        "value_effects": [
            "值增大: 可容纳并发请求数上限提升，但显存与调度开销增大。",
            "值减小: 降低资源压力，但超过上限的请求进入等待队列。",
        ],
        "constraints": ["必须 <= max_num_batched_tokens"],
        "combo_effects": ["建议满足 max_num_seqs * data_parallel_size >= 实际并发目标"],
        "performance_tradeoffs": [
            "增大提升并发容量，但可能增加 TPOT 与排队抖动",
            "减小可提升稳态效率，但高并发下等待时长上升",
        ],
        "failure_signals": ["ValueError: max_num_batched_tokens must be >= max_num_seqs"],
        "evidence_refs": [
            "vllm/config/scheduler.py:55",
            "vllm/config/scheduler.py:258",
            "vllm-ascend/docs/source/tutorials/models/Qwen3-235B-A22B.md:137",
        ],
        "completion_status": "done",
    },
    "--block-size": {
        "value_shape": "enum_numeric",
        "accepted_values": [1, 8, 16, 32, 64, 128, 256],
        "default_behavior": "平台侧会在未设置时选择合适默认值。",
        "value_effects": [
            "较小 block: 更细粒度，管理开销更高。",
            "较大 block: 管理开销更低，但可能造成浪费或兼容限制。",
        ],
        "constraints": [
            "CUDA 通常仅支持 <=32",
            "Xlite graph 要求 block_size = 128",
            "CP interleave 需满足 block_size 可整除 cp_kv_cache_interleave_size",
        ],
        "combo_effects": [
            "与 attention backend 支持的 kernel block size 联动",
            "与 context parallel / graph mode 存在硬约束",
        ],
        "performance_tradeoffs": [
            "值过大可能增加内存颗粒浪费",
            "值过小可能增加调度/映射开销",
        ],
        "failure_signals": [
            "RuntimeError: Xlite graph mode is only compatible with block_size of 128",
            "AssertionError: block_size should be divisible by cp_kv_cache_interleave_size",
        ],
        "evidence_refs": [
            "vllm/config/cache.py:22",
            "vllm/config/cache.py:42",
            "vllm/config/cache.py:44",
            "vllm-ascend/vllm_ascend/ascend_config.py:340",
            "vllm/config/vllm.py:925",
        ],
        "completion_status": "done",
    },
    "--gpu-memory-utilization": {
        "value_shape": "numeric_ratio",
        "accepted_values": ["0 < value <= 1"],
        "default_behavior": "默认 0.9；按实例生效。",
        "value_effects": [
            "值增大: 可用 KV cache 预算增加，吞吐潜力提升。",
            "值减小: 预留内存更多，OOM 风险下降，但 KV cache 容量变小。",
        ],
        "constraints": ["当设置 kv_cache_memory_bytes 时会忽略该参数"],
        "combo_effects": [
            "与 max_num_batched_tokens 联动决定 profile 峰值后可分配 KV cache",
        ],
        "performance_tradeoffs": [
            "过高在真实负载（如 EP 不均匀）下可能触发 OOM",
            "过低会限制缓存并拉低吞吐",
        ],
        "failure_signals": ["运行期 OOM、服务重启或显存不足告警"],
        "evidence_refs": [
            "vllm/config/cache.py:49",
            "vllm/config/cache.py:152",
            "vllm-ascend/docs/source/tutorials/models/Qwen3-235B-A22B.md:142",
        ],
        "completion_status": "done",
    },
    "--max-model-len": {
        "value_shape": "numeric_or_auto",
        "accepted_values": ["int >= 1", "-1", "auto", "k/m/g 后缀"],
        "default_behavior": "未设置时按模型配置推导；-1/auto 触发自动适配。",
        "value_effects": [
            "值增大: 支持更长上下文，但 KV cache 与算力压力显著上升。",
            "值减小: 资源压力降低，但长请求可能被拒绝或截断。",
        ],
        "constraints": [
            "超过模型推导上限需设置 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 才允许",
            "过大在 RoPE/绝对位置编码模型上可能导致 NaN/OOB 风险",
        ],
        "combo_effects": [
            "与 max_num_batched_tokens、sliding_window、speculative 配置联动",
        ],
        "performance_tradeoffs": [
            "大上下文提升能力但降低单位资源吞吐",
            "小上下文更稳但能力边界更低",
        ],
        "failure_signals": [
            "ValueError: user-specified max_model_len greater than derived limit",
            "超长位置导致 NaN 或越界",
        ],
        "evidence_refs": [
            "vllm/config/model.py:173",
            "vllm/config/model.py:177",
            "vllm/config/model.py:1984",
            "vllm/config/model.py:2024",
        ],
        "completion_status": "done",
    },
    "--additional-config": {
        "value_shape": "json_object",
        "accepted_values": [
            "xlite_graph_config",
            "weight_prefetch_config",
            "finegrained_tp_config",
            "ascend_compilation_config",
            "eplb_config",
            "refresh",
            "dump_config_path",
            "enable_async_exponential",
            "enable_shared_expert_dp",
            "multistream_overlap_shared_expert",
            "multistream_overlap_gate",
            "recompute_scheduler_enable",
            "enable_cpu_binding",
            "SLO_limits_for_dynamic_batch",
            "pa_shape_list",
            "enable_kv_nz",
            "layer_sharding",
            "sp_threshold",
        ],
        "default_behavior": "默认空字典，不启用额外插件特性。",
        "value_effects": [
            "按子字段启用 Ascend 扩展能力（图模式、预取、细粒度 TP、动态调度等）。",
        ],
        "constraints": [
            "xlite_graph_config.enabled=true 时要求 block_size=128 且不兼容 speculative decoding",
            "weight_prefetch_config 需结合并发与模型类型调优 prefetch_ratio",
        ],
        "combo_effects": [
            "与 --compilation-config、--block-size、并行参数共同决定最终执行路径",
            "与部分环境变量存在兼容层（如 VLLM_ASCEND_ENABLE_PREFETCH_MLP）",
        ],
        "performance_tradeoffs": [
            "开启更多优化项可能提升吞吐，但会提高配置复杂度和不兼容风险",
        ],
        "failure_signals": [
            "RuntimeError: Xlite graph mode incompatible with current setup",
            "ValueError/AssertionError: finegrained_tp_config 或 eplb_config 校验失败",
        ],
        "evidence_refs": [
            "vllm-ascend/docs/source/user_guide/configuration/additional_config.md:25",
            "vllm-ascend/vllm_ascend/ascend_config.py:34",
            "vllm-ascend/vllm_ascend/ascend_config.py:327",
            "vllm-ascend/vllm_ascend/ascend_config.py:346",
        ],
        "completion_status": "done",
    },
    "--enable-prefix-caching": {
        "value_shape": "binary_or_auto",
        "accepted_values": ["enabled", "disabled", "unset(auto)"],
        "default_behavior": "EngineArgs 未显式设置时会使用模型/后端默认值；CacheConfig 默认值为 True。",
        "value_effects": [
            "enabled: 尝试开启前缀复用以减少重复 prefill 计算。",
            "disabled: 强制关闭前缀缓存，减少缓存管理开销与兼容性风险。",
            "unset(auto): 交由默认策略决策。",
        ],
        "constraints": [
            "在 pooling 等不官方支持场景强行开启会产生风险告警。",
            "在 POWER/S390X/RISC-V CPU 上会被强制关闭。",
            "若设置 mamba-block-size，需要开启前缀缓存。",
        ],
        "combo_effects": [
            "与 mamba_cache_mode/mamba_block_size 联动。",
            "与 prefix_caching_hash_algo 联动决定命中键生成方式。",
        ],
        "performance_tradeoffs": [
            "高重复前缀场景通常明显降 TTFT；低复用场景收益有限。",
        ],
        "failure_signals": [
            "warning: model does not officially support prefix caching",
            "ValueError: --mamba-block-size can only be set with --enable-prefix-caching",
        ],
        "evidence_refs": [
            "vllm/config/cache.py:76",
            "vllm/engine/arg_utils.py:1967",
            "vllm/engine/arg_utils.py:1979",
            "vllm/engine/arg_utils.py:1988",
            "vllm/config/vllm.py:1475",
        ],
        "completion_status": "done",
    },
    "--prefix-caching-hash-algo": {
        "value_shape": "enum",
        "accepted_values": ["sha256", "sha256_cbor", "xxhash", "xxhash_cbor"],
        "default_behavior": "默认 sha256。",
        "value_effects": [
            "sha256/sha256_cbor: 更偏安全性，哈希冲突风险最低。",
            "xxhash/xxhash_cbor: 更偏性能，哈希更快但非密码学安全。",
        ],
        "constraints": [
            "xxhash/xxhash_cbor 需要安装可选 xxhash 依赖。",
            "多租户场景使用非加密哈希存在碰撞与潜在信息泄露风险。",
        ],
        "combo_effects": [
            "与 --enable-prefix-caching 联动；关闭前缀缓存时该值不产生实际效果。",
        ],
        "performance_tradeoffs": [
            "xxhash 通常更快；sha256 通常更稳妥安全。",
        ],
        "failure_signals": ["缺少 xxhash 依赖时启用 xxhash 系列会失败。"],
        "evidence_refs": [
            "vllm/config/cache.py:34",
            "vllm/config/cache.py:78",
            "vllm/config/cache.py:85",
            "vllm/config/cache.py:93",
        ],
        "completion_status": "done",
    },
    "--enforce-eager": {
        "value_shape": "binary_toggle",
        "accepted_values": ["enabled", "disabled"],
        "default_behavior": "默认 disabled（False）。",
        "value_effects": [
            "enabled: 强制 eager 执行，禁用 cudagraph 并将优化级别回退到 O0。",
            "disabled: 允许图模式与 eager 混合路径。",
        ],
        "constraints": [
            "开启后 compilation_config.cudagraph_mode 会被覆盖为 NONE。",
            "bitsandbytes 8bit 或部分 ROCm 场景可能被自动切到 eager。",
        ],
        "combo_effects": [
            "与 --compilation-config / --cudagraph-* 强耦合，开启 eager 后相关图参数失效。",
        ],
        "performance_tradeoffs": [
            "兼容性更高，但通常牺牲图优化带来的吞吐和时延收益。",
        ],
        "failure_signals": [
            "warning: Enforce eager set, overriding optimization level to -O0",
            "info: Cudagraph is disabled under eager mode",
        ],
        "evidence_refs": [
            "vllm/config/model.py:194",
            "vllm/config/model.py:960",
            "vllm/config/model.py:988",
            "vllm/config/vllm.py:723",
            "vllm/config/vllm.py:843",
        ],
        "completion_status": "done",
    },
    "--compilation-config": {
        "value_shape": "json_object",
        "accepted_values": [
            "mode",
            "backend",
            "custom_ops",
            "cudagraph_mode",
            "cudagraph_capture_sizes",
            "max_cudagraph_capture_size",
            "cudagraph_num_of_warmups",
            "pass_config",
            "use_inductor_graph_partition",
        ],
        "default_behavior": "默认空对象，系统按 optimization_level 自动补全 mode/cudagraph 默认值。",
        "value_effects": [
            "可细粒度控制编译后端、图模式、capture 尺寸与 pass 行为。",
        ],
        "constraints": [
            "--cudagraph-capture-sizes 与 compilation_config.cudagraph_capture_sizes 互斥。",
            "--max-cudagraph-capture-size 与 compilation_config.max_cudagraph_capture_size 互斥。",
            "若 cudagraph_mode 需要 piecewise，但 mode 非 VLLM_COMPILE，会被覆盖到 NONE。",
        ],
        "combo_effects": [
            "与 --enforce-eager 冲突：eager 打开会清空 cudagraph 相关设置。",
            "与 --optimization-level 叠加决定最终编译策略。",
        ],
        "performance_tradeoffs": [
            "正确配置可显著降开销；错误配置会导致启动失败或图优化被回退。",
        ],
        "failure_signals": [
            "ValueError: cudagraph_capture_sizes ... mutually exclusive",
            "ValueError: max_cudagraph_capture_size ... mutually exclusive",
        ],
        "evidence_refs": [
            "vllm/config/vllm.py:243",
            "vllm/config/vllm.py:757",
            "vllm/config/vllm.py:781",
            "vllm/config/compilation.py:486",
            "vllm/engine/arg_utils.py:1775",
        ],
        "completion_status": "done",
    },
    "--cudagraph-capture-sizes": {
        "value_shape": "list_numeric",
        "accepted_values": ["list[int] (non-empty when cudagraph enabled)"],
        "default_behavior": "默认 None；未指定时按 max_num_seqs/max_num_batched_tokens 自动生成候选 sizes。",
        "value_effects": [
            "显式设置: 强制使用给定 capture sizes（会去重、排序并按 token 上限裁剪）。",
            "不设置: 使用系统默认分段规则生成。",
        ],
        "constraints": [
            "与 compilation_config.cudagraph_capture_sizes 互斥。",
            "开启 cudagraph 时列表不能为空。",
        ],
        "combo_effects": [
            "与 --max-cudagraph-capture-size 需一致，否则可能触发告警或错误。",
            "与 speculative/sequence parallel 参数联动后，sizes 可能被重新对齐。",
        ],
        "performance_tradeoffs": [
            "列表越大覆盖越广但启动捕获成本和内存占用更高。",
        ],
        "failure_signals": [
            "ValueError: cudagraph_capture_sizes and compilation_config... are mutually exclusive",
            "AssertionError: cudagraph_capture_sizes should contain at least one element",
        ],
        "evidence_refs": [
            "vllm/engine/arg_utils.py:1776",
            "vllm/config/vllm.py:1239",
            "vllm/config/vllm.py:1241",
            "vllm/config/vllm.py:1308",
        ],
        "completion_status": "done",
    },
    "--max-cudagraph-capture-size": {
        "value_shape": "numeric",
        "accepted_values": ["int >= 1 when cudagraph enabled"],
        "default_behavior": "默认 None；未设时按 min(max_num_seqs * decode_query_len * 2, 512) 自动估算。",
        "value_effects": [
            "增大: 捕获更大 batch 的图，潜在提升高并发 decode 性能。",
            "减小: 降低启动图捕获成本与显存压力。",
        ],
        "constraints": [
            "与 compilation_config.max_cudagraph_capture_size 互斥。",
            "若同时显式给出 cudagraph_capture_sizes，需与其最大值一致，否则报错。",
        ],
        "combo_effects": [
            "与 --cudagraph-capture-sizes、--max-num-batched-tokens 联动决定最终可用 sizes。",
        ],
        "performance_tradeoffs": [
            "上限过高可能拖慢启动并增加内存占用；过低可能错失大 batch 图收益。",
        ],
        "failure_signals": [
            "ValueError: max_cudagraph_capture_size and compilation_config... are mutually exclusive",
            "ValueError: customized max_cudagraph_capture_size ... should be consistent ...",
        ],
        "evidence_refs": [
            "vllm/engine/arg_utils.py:1783",
            "vllm/config/compilation.py:571",
            "vllm/config/vllm.py:1227",
            "vllm/config/vllm.py:1286",
        ],
        "completion_status": "done",
    },
    "--speculative-config": {
        "value_shape": "json_object",
        "accepted_values": [
            "method",
            "model",
            "num_speculative_tokens",
            "draft_tensor_parallel_size",
            "quantization",
            "disable_by_batch_size",
            "disable_padded_drafter_batch",
            "parallel_drafting",
        ],
        "default_behavior": "默认 None（关闭 speculative decoding）。",
        "value_effects": [
            "配置后启用投机解码路径，通常目标是降低解码时延。",
        ],
        "constraints": [
            "num_speculative_tokens 必须 > 0。",
            "speculative_config 内不允许 tensor_parallel_size 字段，需使用 draft_tensor_parallel_size。",
            "draft_tensor_parallel_size 仅允许 1 或 target TP。",
        ],
        "combo_effects": [
            "与 --async-scheduling 联动：仅 EAGLE/MTP/draft_model 且 disable_padded_drafter_batch=False 才兼容。",
            "与 cudagraph size 计算联动：num_speculative_tokens 会影响 decode_query_len 与 graph size 上限。",
        ],
        "performance_tradeoffs": [
            "配置正确可降 TPOT；配置不当会带来额外校验/回退开销甚至启动失败。",
        ],
        "failure_signals": [
            "ValueError: num_speculative_tokens must be provided ...",
            "ValueError: 'tensor_parallel_size' is not a valid argument in speculative_config",
            "ValueError: async scheduling is only supported with EAGLE/MTP/Draft ...",
        ],
        "evidence_refs": [
            "vllm/config/speculative.py:57",
            "vllm/config/speculative.py:63",
            "vllm/config/speculative.py:625",
            "vllm/config/speculative.py:658",
            "vllm/config/vllm.py:622",
            "vllm/config/vllm.py:631",
        ],
        "completion_status": "done",
    },
    "--pipeline-parallel-size": {
        "value_shape": "numeric",
        "accepted_values": ["int >= 1"],
        "default_behavior": "默认 1（不开启 PP）。",
        "value_effects": [
            "值 > 1: 将模型按 pipeline stage 切分，降低单设备承载压力。",
            "值 = 1: 单 stage 路径。",
        ],
        "constraints": [
            "模型必须支持 PP（SupportsPP），否则抛 NotImplementedError。",
            "非 Ray/MP/external_launcher 的后端下，PP>1 可能被判定为不支持。",
        ],
        "combo_effects": [
            "与 tensor_parallel_size、data_parallel_size 共同决定 world_size。",
        ],
        "performance_tradeoffs": [
            "可提升超大模型可部署性，但会引入 pipeline 气泡与跨 stage 通信开销。",
        ],
        "failure_signals": [
            "NotImplementedError: Pipeline parallelism is not supported for this model",
            "unsupported: Pipeline Parallelism without Ray/mp/external launcher",
        ],
        "evidence_refs": [
            "vllm/config/parallel.py:97",
            "vllm/config/model.py:1059",
            "vllm/config/model.py:1063",
            "vllm/engine/arg_utils.py:1830",
        ],
        "completion_status": "done",
    },
    "--tensor-parallel-size": {
        "value_shape": "numeric",
        "accepted_values": ["int >= 1"],
        "default_behavior": "默认 1（不开启 TP）。",
        "value_effects": [
            "值增大: 模型按张量维切分，单模型可用算力和显存池扩大。",
            "值为 1: 单卡/单分片执行。",
        ],
        "constraints": [
            "模型 attention head 总数必须可被 TP 整除。",
            "decode_context_parallel_size 必须整除 TP（tp_size % dcp_size == 0）。",
        ],
        "combo_effects": [
            "与 data/pipeline/expert parallel 联动决定并行拓扑。",
            "与 all2all_backend、enable_sp 等编译路径联动影响图模式可用性。",
        ],
        "performance_tradeoffs": [
            "可扩容量与吞吐，但通信成本上升；小 batch 下可能收益有限。",
        ],
        "failure_signals": [
            "ValueError: Total number of attention heads ... must be divisible by tensor parallel size",
            "ValueError: tp_size must be divisible by dcp_size",
        ],
        "evidence_refs": [
            "vllm/config/parallel.py:99",
            "vllm/config/parallel.py:352",
            "vllm/config/model.py:1047",
            "vllm/config/model.py:1050",
        ],
        "completion_status": "done",
    },
    "--data-parallel-size": {
        "value_shape": "numeric",
        "accepted_values": ["int >= 1"],
        "default_behavior": "默认 1（不开启 DP）。",
        "value_effects": [
            "值增大: 增加副本并行，提升总体吞吐上限。",
            "值为 1: 单副本。",
        ],
        "constraints": [
            "data_parallel_size_local 必须 <= data_parallel_size。",
            "在外部 LB 模式下，data_parallel_external_lb 仅在 data_parallel_size > 1 时允许。",
            "部分离线 dense 场景（env fallback）不支持/不建议 DP > 1。",
        ],
        "combo_effects": [
            "与 data_parallel_backend/rank/local_size/hybrid_lb 联动决定路由与拓扑。",
            "与 pipeline/tensor parallel 共同决定总 world_size。",
        ],
        "performance_tradeoffs": [
            "吞吐可线性扩展潜力高，但副本间同步和负载均衡复杂度上升。",
        ],
        "failure_signals": [
            "ValueError: data_parallel_size_local ... must be <= data_parallel_size",
            "ValueError: data_parallel_external_lb can only be set when data_parallel_size > 1",
        ],
        "evidence_refs": [
            "vllm/config/parallel.py:103",
            "vllm/config/parallel.py:313",
            "vllm/config/parallel.py:319",
            "vllm/config/parallel.py:588",
        ],
        "completion_status": "done",
    },
    "--data-parallel-size-local": {
        "value_shape": "numeric_or_auto",
        "accepted_values": ["int >= 1", "unset(auto infer)"],
        "default_behavior": "默认 None 时按节点拓扑、后端与 LB 模式自动推导。",
        "value_effects": [
            "设置明确值: 固定每节点本地 DP 宽度。",
            "不设置: 系统根据 nnodes/backend/LB 策略推导。",
        ],
        "constraints": [
            "必须 <= data_parallel_size。",
            "当 data_parallel_rank 显式给出（外部 LB）时，local size 只能是 1 或 None。",
            "启用 hybrid_lb 时必须可推导到有效 local size。",
        ],
        "combo_effects": [
            "与 data_parallel_hybrid_lb / data_parallel_external_lb 强耦合。",
        ],
        "performance_tradeoffs": [
            "local size 增大有助于节点内吞吐，但跨节点协调和 LB 复杂度提升。",
        ],
        "failure_signals": [
            "AssertionError: data_parallel_size_local must be 1 or None when data_parallel_rank is set",
            "AssertionError: data_parallel_size_local must be set to use data_parallel_hybrid_lb",
        ],
        "evidence_refs": [
            "vllm/config/parallel.py:106",
            "vllm/config/parallel.py:313",
            "vllm/engine/arg_utils.py:1532",
            "vllm/engine/arg_utils.py:1571",
        ],
        "completion_status": "done",
    },
    "--data-parallel-backend": {
        "value_shape": "enum",
        "accepted_values": ["mp", "ray"],
        "default_behavior": "默认 mp。",
        "value_effects": [
            "mp: 走多进程/本地分布式路径。",
            "ray: 走 Ray 调度与地址管理路径。",
        ],
        "constraints": [
            "nnodes > 1 仅支持 data_parallel_backend=mp。",
            "非法值会在 backend 选择阶段断言失败。",
        ],
        "combo_effects": [
            "与 distributed_executor_backend、nnodes、data_parallel_address 推导逻辑联动。",
        ],
        "performance_tradeoffs": [
            "ray 便于弹性编排；mp 路径更直接、部署复杂度相对低。",
        ],
        "failure_signals": [
            "AssertionError: nnodes > 1 is only supported with data_parallel_backend=mp",
            "AssertionError: data_parallel_backend can only be ray or mp",
        ],
        "evidence_refs": [
            "vllm/config/parallel.py:37",
            "vllm/config/parallel.py:119",
            "vllm/engine/arg_utils.py:1488",
            "vllm/engine/arg_utils.py:1594",
        ],
        "completion_status": "done",
    },
    "--all2all-backend": {
        "value_shape": "enum",
        "accepted_values": [
            "naive",
            "pplx",
            "deepep_high_throughput",
            "deepep_low_latency",
            "mori",
            "allgather_reducescatter",
            "flashinfer_all2allv",
        ],
        "default_behavior": "默认 allgather_reducescatter。",
        "value_effects": [
            "决定 MoE expert 并行通信实现路径和吞吐/时延曲线。",
        ],
        "constraints": [
            "主要在 enable_expert_parallel 场景生效。",
        ],
        "combo_effects": [
            "与 enable_expert_parallel + TP>1 + DP>1 时，部分后端会触发 sequence parallel MoE 路径。",
        ],
        "performance_tradeoffs": [
            "不同后端在高吞吐与低时延场景表现差异明显，需要按硬件与负载压测选择。",
        ],
        "failure_signals": ["非法枚举值会在参数解析阶段报错。"],
        "evidence_refs": [
            "vllm/config/parallel.py:39",
            "vllm/config/parallel.py:150",
            "vllm/config/parallel.py:447",
        ],
        "completion_status": "done",
    },
    "--enable-expert-parallel": {
        "value_shape": "binary_toggle",
        "accepted_values": ["enabled", "disabled"],
        "default_behavior": "默认 disabled（False）。",
        "value_effects": [
            "enabled: MoE 层采用专家并行而非纯 TP 路径。",
            "disabled: 保持常规并行策略。",
        ],
        "constraints": [
            "仅 MoE 模型可开启；dense 模型会报错。",
        ],
        "combo_effects": [
            "与 --all2all-backend、--enable-eplb、TP/DP 参数强耦合。",
            "qwen3-32b-w8a8 profile 下与 EP 组合应硬阻断（见 combo rule）。",
        ],
        "performance_tradeoffs": [
            "适合 MoE 扩展吞吐，但对通信拓扑和负载均衡要求更高。",
        ],
        "failure_signals": [
            "ValueError: Number of experts in the model must be greater than 0 when expert parallelism is enabled.",
        ],
        "evidence_refs": [
            "vllm/config/parallel.py:135",
            "vllm/config/model.py:996",
            "vllm/config/model.py:1056",
        ],
        "completion_status": "done",
    },
    "--enable-eplb": {
        "value_shape": "binary_toggle",
        "accepted_values": ["enabled", "disabled"],
        "default_behavior": "默认 disabled（False）。",
        "value_effects": [
            "enabled: 开启 MoE expert 负载均衡策略。",
            "disabled: 不进行 EPLB 调整。",
        ],
        "constraints": [
            "仅 CUDA/ROCm 设备支持。",
            "必须同时 enable_expert_parallel=True。",
            "要求 TP*DP > 1。",
        ],
        "combo_effects": [
            "与 --eplb-config 联动决定窗口、异步策略与冗余专家策略。",
        ],
        "performance_tradeoffs": [
            "可改善专家负载偏斜，但会引入额外调度/通信开销。",
        ],
        "failure_signals": [
            "ValueError: enable_expert_parallel must be True to use EPLB.",
            "ValueError: EPLB requires tensor_parallel_size or data_parallel_size to be greater than 1.",
        ],
        "evidence_refs": [
            "vllm/config/parallel.py:137",
            "vllm/config/parallel.py:324",
            "vllm/config/parallel.py:330",
            "vllm/config/parallel.py:332",
        ],
        "completion_status": "done",
    },
    "--eplb-config": {
        "value_shape": "json_object",
        "accepted_values": [
            "window_size",
            "step_interval",
            "num_redundant_experts",
            "log_balancedness",
            "log_balancedness_interval",
            "use_async",
            "policy",
        ],
        "default_behavior": "默认使用 EPLBConfig 默认值（window=1000, step_interval=3000, policy=default 等）。",
        "value_effects": [
            "通过窗口、重排间隔和异步策略控制专家负载均衡强度。",
        ],
        "constraints": [
            "use_async=True 仅支持 policy=default。",
            "log_balancedness=True 时 log_balancedness_interval 必须 > 0。",
            "当 enable_eplb=False 且 num_redundant_experts!=0 会报错。",
        ],
        "combo_effects": [
            "仅在 --enable-eplb 打开时完整生效。",
        ],
        "performance_tradeoffs": [
            "更激进的均衡策略可改善热点专家，但也可能带来更多通信/重排开销。",
        ],
        "failure_signals": [
            "ValueError: Async EPLB is only supported with the default policy.",
            "ValueError: num_redundant_experts ... but EPLB is not enabled",
        ],
        "evidence_refs": [
            "vllm/config/parallel.py:51",
            "vllm/config/parallel.py:84",
            "vllm/config/parallel.py:339",
            "vllm/engine/arg_utils.py:605",
        ],
        "completion_status": "done",
    },
    "--expert-placement-strategy": {
        "value_shape": "enum",
        "accepted_values": ["linear", "round_robin"],
        "default_behavior": "默认 linear。",
        "value_effects": [
            "linear: 连续专家分配，映射更直观。",
            "round_robin: 轮转分配，通常更利于负载均衡。",
        ],
        "constraints": ["主要在 MoE + expert parallel 场景生效。"],
        "combo_effects": [
            "与 --enable-expert-parallel / --enable-eplb 配合决定最终专家负载分布。",
        ],
        "performance_tradeoffs": [
            "round_robin 在分组专家模型上常更均衡，但可能改变通信局部性。",
        ],
        "failure_signals": ["非法枚举值会在配置解析阶段报错。"],
        "evidence_refs": [
            "vllm/config/parallel.py:35",
            "vllm/config/parallel.py:141",
        ],
        "completion_status": "done",
    },
    "--disable-nccl-for-dp-synchronization": {
        "value_shape": "binary_or_auto",
        "accepted_values": ["enabled", "disabled", "unset(auto)"],
        "default_behavior": "默认 unset(None)：async scheduling 开启时自动设为 True，否则 False。",
        "value_effects": [
            "enabled: DP 同步优先走 Gloo 而非 NCCL。",
            "disabled: DP 同步优先走 NCCL。",
            "unset(auto): 按 async_scheduling 自动决策。",
        ],
        "constraints": ["该项主要影响 DP 同步实现，非 DP 场景影响有限。"],
        "combo_effects": [
            "与 --async-scheduling、--data-parallel-size 联动最明显。",
        ],
        "performance_tradeoffs": [
            "切换到 Gloo 可绕过部分 NCCL 问题，但吞吐与时延可能变化。",
        ],
        "failure_signals": ["通信栈不匹配时可能出现同步性能下降或超时。"],
        "evidence_refs": [
            "vllm/config/parallel.py:185",
            "vllm/config/vllm.py:694",
            "vllm/config/vllm.py:704",
            "vllm/config/vllm.py:706",
        ],
        "completion_status": "done",
    },
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": {
        "value_shape": "binary_toggle",
        "accepted_values": ["0", "1"],
        "default_behavior": "默认 0（不允许超过模型推导上限的 max_model_len）。",
        "value_effects": [
            "1: 允许将 --max-model-len 设到推导上限之上（仅告警，不直接阻断）。",
            "0: 超出推导上限时直接报错阻断启动。",
        ],
        "constraints": [
            "仅在用户 max_model_len > derived_max_model_len 时生效。",
            "官方明确提示该开关需极度谨慎使用（RoPE 可能 NaN，绝对位置编码可能 OOB）。",
        ],
        "combo_effects": [
            "与 --max-model-len 强耦合；不开该开关时超长配置会报错。",
        ],
        "performance_tradeoffs": [
            "放宽上限可覆盖更长上下文，但数值稳定性和正确性风险显著上升。",
        ],
        "failure_signals": [
            "ValueError: User-specified max_model_len ... To allow overriding this maximum, set VLLM_ALLOW_LONG_MAX_MODEL_LEN=1",
            "warning: positions exceeding derived_max_model_len may lead to NaN/OOB",
        ],
        "evidence_refs": [
            "vllm/envs.py:815",
            "vllm/envs.py:819",
            "vllm/config/model.py:2004",
            "vllm/config/model.py:2017",
            "vllm/config/model.py:2024",
        ],
        "completion_status": "done",
    },
    "VLLM_DP_SIZE": {
        "value_shape": "numeric",
        "accepted_values": ["int >= 1"],
        "default_behavior": "默认 1。",
        "value_effects": [
            "增大: 提升 DP 副本规模（离线 SPMD fallback 场景从 env 注入）。",
            "为 1: 单副本路径。",
        ],
        "constraints": [
            "通过 env fallback 注入时，dense 模型离线 DP>1 会报错。",
            "需与 VLLM_DP_RANK / VLLM_DP_MASTER_* 协同配置。",
        ],
        "combo_effects": [
            "与 VLLM_DP_RANK、VLLM_DP_RANK_LOCAL、VLLM_DP_MASTER_IP/PORT 联动形成完整 DP 拓扑。",
        ],
        "performance_tradeoffs": [
            "副本数增大可提高吞吐，但跨副本同步/调度复杂度增加。",
        ],
        "failure_signals": [
            "ValueError: Offline data parallel mode is not supported/useful for dense models.",
            "ValueError: data_parallel_rank ... must be in the range [0, VLLM_DP_SIZE)",
        ],
        "evidence_refs": [
            "vllm/envs.py:1047",
            "vllm/config/parallel.py:582",
            "vllm/config/parallel.py:588",
            "vllm/config/parallel.py:575",
        ],
        "completion_status": "done",
    },
    "VLLM_DP_RANK": {
        "value_shape": "numeric",
        "accepted_values": ["int >= 0"],
        "default_behavior": "默认 0。",
        "value_effects": [
            "指定当前进程在 DP 组内的全局 rank。",
        ],
        "constraints": [
            "必须满足 0 <= rank < VLLM_DP_SIZE。",
        ],
        "combo_effects": [
            "与 VLLM_DP_SIZE、VLLM_DP_RANK_LOCAL 一起决定本地/全局 DP 映射。",
        ],
        "performance_tradeoffs": [
            "rank 本身不影响性能，但错误配置会导致 DP 拓扑不可用。",
        ],
        "failure_signals": [
            "ValueError: data_parallel_rank ... must be in the range [0, data_parallel_size)",
        ],
        "evidence_refs": [
            "vllm/envs.py:1040",
            "vllm/config/parallel.py:583",
            "vllm/config/parallel.py:575",
        ],
        "completion_status": "done",
    },
    "VLLM_DP_RANK_LOCAL": {
        "value_shape": "numeric_or_default",
        "accepted_values": ["int >= -1"],
        "default_behavior": "未设置时默认跟随 VLLM_DP_RANK。",
        "value_effects": [
            "用于标识节点内 DP 本地 rank，服务于本地调度与通信映射。",
        ],
        "constraints": [
            "需与 VLLM_DP_SIZE、VLLM_DP_RANK 保持一致，避免本地拓扑错配。",
        ],
        "combo_effects": [
            "与 VLLM_DP_RANK/VLLM_DP_SIZE 联动决定 local rank 视图。",
        ],
        "performance_tradeoffs": [
            "本地 rank 只影响拓扑映射，不直接改变算子性能。",
        ],
        "failure_signals": [
            "本地 rank 与实际拓扑不匹配时会导致通信/映射异常。",
        ],
        "evidence_refs": [
            "vllm/envs.py:1042",
            "vllm/envs.py:1043",
            "vllm/config/parallel.py:584",
        ],
        "completion_status": "done",
    },
    "VLLM_DP_MASTER_IP": {
        "value_shape": "string_ip",
        "accepted_values": ["IPv4/hostname"],
        "default_behavior": "默认 127.0.0.1。",
        "value_effects": [
            "指定 DP 主节点地址，用于分布式控制与套接字通信。",
        ],
        "constraints": [
            "多节点场景需保证地址可达且与端口一致。",
        ],
        "combo_effects": [
            "与 VLLM_DP_MASTER_PORT、VLLM_DP_SIZE、VLLM_DP_RANK 联动。",
        ],
        "performance_tradeoffs": [
            "地址错误会直接导致连接失败，不属于性能微调项。",
        ],
        "failure_signals": [
            "DP 组初始化连接失败、RPC/Socket 无法建立。",
        ],
        "evidence_refs": [
            "vllm/envs.py:1049",
            "vllm/config/parallel.py:585",
        ],
        "completion_status": "done",
    },
    "VLLM_DP_MASTER_PORT": {
        "value_shape": "numeric_port",
        "accepted_values": ["int >= 0 (建议有效监听端口)"],
        "default_behavior": "默认 0（由系统/逻辑进一步分配）。",
        "value_effects": [
            "指定 DP 主节点端口，用于控制面通信。",
        ],
        "constraints": [
            "端口需可用且与主节点地址匹配，冲突会导致初始化失败。",
        ],
        "combo_effects": [
            "与 VLLM_DP_MASTER_IP、VLLM_DP_SIZE、VLLM_DP_RANK 联动。",
        ],
        "performance_tradeoffs": [
            "端口配置主要影响可用性，不直接影响推理性能。",
        ],
        "failure_signals": [
            "端口占用/连接失败导致 DP 初始化错误。",
        ],
        "evidence_refs": [
            "vllm/envs.py:1051",
            "vllm/config/parallel.py:586",
            "vllm/utils/network_utils.py:159",
        ],
        "completion_status": "done",
    },
    "VLLM_WORKER_MULTIPROC_METHOD": {
        "value_shape": "enum",
        "accepted_values": ["fork", "spawn"],
        "default_behavior": "env 默认 fork；CLI 启动路径若未设置会主动注入 spawn。",
        "value_effects": [
            "fork: 启动开销低，但在 CUDA/WSL/部分模型下稳定性风险更高。",
            "spawn: 兼容性更好，但进程启动成本通常更高。",
        ],
        "constraints": [
            "在 Ray actor、CUDA 已初始化或 WSL 场景，系统可能强制覆盖为 spawn。",
            "Whisper 场景使用 fork 可能启动挂起并触发告警建议切 spawn。",
        ],
        "combo_effects": [
            "与运行上下文（Ray/CUDA 初始化状态/WSL）联动，而非单纯静态配置。",
        ],
        "performance_tradeoffs": [
            "fork 启动更快；spawn 稳定性更高。",
        ],
        "failure_signals": [
            "warning: Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'",
            "warning: Whisper is known to have issues with forked workers",
        ],
        "evidence_refs": [
            "vllm/envs.py:712",
            "vllm/entrypoints/utils.py:170",
            "vllm/utils/system_utils.py:114",
            "vllm/utils/system_utils.py:140",
            "vllm/config/vllm.py:879",
        ],
        "completion_status": "done",
    },
    "VLLM_API_KEY": {
        "value_shape": "string_secret",
        "accepted_values": ["non-empty token string"],
        "default_behavior": "默认 None（不从环境注入鉴权 token）。",
        "value_effects": [
            "设置后可作为 OpenAI API Server 鉴权 token 来源。",
        ],
        "constraints": [
            "CLI `--api-key` 优先级高于环境变量。",
            "密钥应通过安全渠道注入，避免出现在日志或命令历史。",
        ],
        "combo_effects": [
            "与 serve 启动参数共同决定 AuthenticationMiddleware 是否启用。",
        ],
        "performance_tradeoffs": [
            "对推理性能影响极小，主要影响访问控制。",
        ],
        "failure_signals": [
            "未设置且无 CLI key 时接口可能处于无鉴权状态（依部署策略）。",
        ],
        "evidence_refs": [
            "vllm/envs.py:614",
            "vllm/entrypoints/openai/api_server.py:239",
            "vllm/entrypoints/openai/api_server.py:243",
        ],
        "completion_status": "done",
    },
    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": {
        "value_shape": "binary_toggle",
        "accepted_values": ["0", "1"],
        "default_behavior": "默认 0（关闭运行时 LoRA 动态加载/卸载）。",
        "value_effects": [
            "1: 开启运行时 LoRA 动态管理接口（开发/实验场景）。",
            "0: 不暴露动态 LoRA 路由。",
        ],
        "constraints": [
            "api_server_count > 1 时不允许开启，会直接报错。",
            "官方警告该能力应仅用于本地开发场景。",
        ],
        "combo_effects": [
            "与 LoRA resolver 相关环境变量联动（插件与仓库解析）。",
        ],
        "performance_tradeoffs": [
            "增强动态性，但引入运行时状态管理复杂度与安全风险。",
        ],
        "failure_signals": [
            "ValueError: VLLM_ALLOW_RUNTIME_LORA_UPDATING cannot be used with api_server_count > 1",
            "warning: LoRA dynamic loading & unloading is enabled ... ONLY be used for local development",
        ],
        "evidence_refs": [
            "vllm/envs.py:861",
            "vllm/entrypoints/cli/serve.py:234",
            "vllm/entrypoints/serve/lora/api_router.py:27",
            "vllm/entrypoints/serve/lora/api_router.py:30",
        ],
        "completion_status": "done",
    },
    "CUDA_VISIBLE_DEVICES": {
        "value_shape": "gpu_id_list",
        "accepted_values": ["comma-separated GPU ids, e.g. 0,1,2,3", "unset(None)"],
        "default_behavior": "默认 None（由运行时/编排系统决定可见设备）。",
        "value_effects": [
            "设置后限制当前进程可见 GPU 集合，直接影响 worker 设备映射与可用卡数统计。",
        ],
        "constraints": [
            "配置过窄会导致 world_size 大于可见 GPU 数并触发启动失败。",
            "Ray 场景下该变量可能由调度器注入或在 worker 生命周期中重写。",
        ],
        "combo_effects": [
            "与 tensor/pipeline/data parallel world_size 约束直接耦合。",
        ],
        "performance_tradeoffs": [
            "主要影响资源分配与隔离，不是直接的算子性能开关。",
        ],
        "failure_signals": [
            "ValueError: World size (...) is larger than the number of available GPUs (...)",
        ],
        "evidence_refs": [
            "vllm/envs.py:603",
            "vllm/config/parallel.py:617",
            "vllm/utils/torch_utils.py:633",
            "vllm/v1/engine/core.py:1540",
        ],
        "completion_status": "done",
    },
    "VLLM_ASCEND_ENABLE_PREFETCH_MLP": {
        "value_shape": "binary_toggle",
        "accepted_values": ["0", "1"],
        "default_behavior": "默认 0（关闭）。",
        "value_effects": [
            "1: 启用旧版 MLP 预取兼容路径，并读取 gate_up/down 预取大小。",
            "0: 不启用该兼容路径，推荐使用 additional_config.weight_prefetch_config。",
        ],
        "constraints": ["该变量已标记弃用，后续版本将移除"],
        "combo_effects": [
            "与 VLLM_ASCEND_MLP_GATE_UP_PREFETCH_SIZE / DOWN_PREFETCH_SIZE 联动",
        ],
        "performance_tradeoffs": [
            "小并发可能收益有限；错误预取大小会造成资源争用",
        ],
        "failure_signals": ["DeprecationWarning: VLLM_ASCEND_ENABLE_PREFETCH_MLP is deprecated"],
        "evidence_refs": [
            "vllm-ascend/vllm_ascend/envs.py:80",
            "vllm-ascend/vllm_ascend/ascend_config.py:150",
            "vllm-ascend/vllm_ascend/ascend_config.py:163",
        ],
        "completion_status": "done",
    },
    "VLLM_ASCEND_ENABLE_FLASHCOMM1": {
        "value_shape": "binary_toggle",
        "accepted_values": ["0", "1"],
        "default_behavior": "默认 0（关闭）。",
        "value_effects": [
            "1: 启用 FlashComm1 通信优化，适合高并发场景。",
            "0: 使用常规通信路径。",
        ],
        "constraints": ["主要在 MoE 且 tp_size > 1 场景有收益"],
        "combo_effects": [
            "与 prefill_context_parallel_size、tensor_parallel_size 联动时会约束 max_num_batched_tokens 对齐",
        ],
        "performance_tradeoffs": [
            "高并发可提升吞吐；低并发或不匹配场景收益有限",
        ],
        "failure_signals": ["不满足约束时会触发参数对齐告警或收益不稳定"],
        "evidence_refs": [
            "vllm-ascend/vllm_ascend/envs.py:72",
            "vllm-ascend/vllm_ascend/ascend_config.py:76",
            "vllm-ascend/docs/source/tutorials/models/Qwen3-235B-A22B.md:149",
        ],
        "completion_status": "done",
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


def _infer_value_shape(value_type: str, valid_values: list[str]) -> str:
    if value_type == "bool":
        return "binary_toggle"
    if valid_values:
        return "enum"
    if value_type in {"int", "float"}:
        return "numeric"
    if value_type == "json":
        return "json_object"
    if value_type == "list":
        return "list"
    return "free_form"


def _default_value_semantics(name: str, raw: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    value_type = raw.get("type", "string")
    valid_values = [str(item) for item in raw.get("valid_values", [])]
    default_value = raw.get("default")
    help_text = raw.get("help_text")

    accepted_values: list[Any]
    if valid_values:
        accepted_values = valid_values
    elif value_type == "bool":
        accepted_values = ["enabled", "disabled"]
    elif value_type in {"int", "float"}:
        accepted_values = [f"{value_type} value"]
    elif value_type == "json":
        accepted_values = ["JSON object"]
    elif value_type == "list":
        accepted_values = ["list value"]
    else:
        accepted_values = ["string value"]

    default_behavior = "使用默认值。" if default_value is not None else "未显式设置时使用系统默认行为。"
    if isinstance(help_text, str) and help_text.strip():
        default_behavior = help_text.strip()

    return {
        "value_shape": _infer_value_shape(value_type, valid_values),
        "accepted_values": accepted_values,
        "default_value": default_value,
        "default_behavior": default_behavior,
        "value_effects": [defaults["semantics"]],
        "constraints": defaults.get("incompatibilities", []),
        "combo_effects": [],
        "performance_tradeoffs": [],
        "failure_signals": defaults.get("failure_modes", []),
        "evidence_refs": [],
        "completion_status": "todo",
    }


def _merge_value_semantics(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        merged[key] = value
    return merged


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
        value_semantics = _default_value_semantics(name, raw, defaults)
        if name in VALUE_SEMANTICS_OVERRIDES:
            value_semantics = _merge_value_semantics(value_semantics, VALUE_SEMANTICS_OVERRIDES[name])
        if not value_semantics.get("evidence_refs"):
            value_semantics["evidence_refs"] = (definition_ref + effect_ref + read_ref)[:6]

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
            "value_semantics": value_semantics,
            "value_semantics_completion": value_semantics.get("completion_status", "todo"),
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
            "value_shape": entry["value_semantics"].get("value_shape"),
            "value_semantics_completion": entry.get("value_semantics_completion", "todo"),
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
    value_semantics_done = sum(1 for item in entries if item.get("value_semantics_completion") == "done")
    value_semantics_todo = sum(1 for item in entries if item.get("value_semantics_completion") != "done")

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
        "value_semantics_progress": {
            "done": value_semantics_done,
            "todo": value_semantics_todo,
            "ratio": round(value_semantics_done / max(1, total), 4),
        },
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
                f"- Value semantics done: **{validation_report['value_semantics_progress']['done']}**",
                f"- Value semantics ratio: **{validation_report['value_semantics_progress']['ratio']}**",
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
    _write_json(out_deploy / "global_value_semantics_progress.json", validation_report["value_semantics_progress"])
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
        "value_semantics_done": validation_report["value_semantics_progress"]["done"],
        "value_semantics_ratio": validation_report["value_semantics_progress"]["ratio"],
        "generated_at": now,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
