from __future__ import annotations

import re
from typing import Any

from .types import ParsedRequest


_MODEL_PATTERNS: list[tuple[str, list[str]]] = [
    ('owen3', ['owen3', 'owen 3', 'owen']),
    ('qwen3.5', ['qwen3.5', 'qwen3 5', 'qwen3_5']),
    ('qwen2-vl', ['qwen2-vl', 'qwen2 vl']),
    ('deepseek-v3.1', ['deepseek-v3.1', 'deepseek v3.1']),
    ('qwen3', ['qwen3']),
]

_HARDWARES = ['310p', 'a2', 'a3', 'a5']
_QUANTS = ['w4a4', 'w4a8', 'w8a8', 'bf16', 'fp16', 'float16']


def _extract_first_int(pattern: str, text: str) -> int | None:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def parse_request(text: str, *, overrides: dict[str, Any] | None = None) -> ParsedRequest:
    lower = text.lower()
    intent = 'deployment' if any(k in lower for k in ['部署', '脚本', '命令', 'serve', 'server']) else 'unknown'

    model_name = None
    for canonical, patterns in _MODEL_PATTERNS:
        if any(p in lower for p in patterns):
            model_name = canonical
            break

    size_b = None
    m = re.search(r'(?<![\d.])(\d+(?:\.\d+)?)\s*[bB]\b', text)
    if m:
        try:
            size_b = float(m.group(1))
        except ValueError:
            size_b = None

    hardware = None
    for hw in _HARDWARES:
        if hw in lower:
            hardware = hw.upper()
            break

    cards = None
    if '双卡' in text:
        cards = 2
    if '单卡' in text:
        cards = 1
    m = re.search(r'(\d+)\s*卡', text)
    if m:
        cards = int(m.group(1))

    quant = None
    for q in _QUANTS:
        if q in lower:
            quant = q.upper()
            break

    objective = 'unknown'
    if '高吞吐' in text or 'throughput' in lower:
        objective = 'throughput'
    elif '低时延' in text or 'latency' in lower or 'tpot' in lower:
        objective = 'latency'
    elif '平衡' in text:
        objective = 'balanced'

    avg_in = None
    avg_out = None
    max_ctx = None
    tpot_limit = None

    # Average input/output like 3.5k / 1.5k
    m = re.search(r'平均[^\d]*(\d+(?:\.\d+)?)\s*[kK]?\s*输入[^\d]*(\d+(?:\.\d+)?)\s*[kK]?\s*输出', text)
    if m:
        avg_in = int(float(m.group(1)) * (1000 if 'k' in m.group(0).lower() else 1))
        avg_out = int(float(m.group(2)) * (1000 if 'k' in m.group(0).lower() else 1))
    else:
        m = re.search(r'(\d+(?:\.\d+)?)\s*[kK]?\s*输入\s*[/.，,， ]+\s*(\d+(?:\.\d+)?)\s*[kK]?\s*输出', text)
        if m:
            avg_in = int(float(m.group(1)) * (1000 if 'k' in m.group(0).lower() else 1))
            avg_out = int(float(m.group(2)) * (1000 if 'k' in m.group(0).lower() else 1))

    m = re.search(r'最大上下文[^\d]*(\d+(?:\.\d+)?)\s*[kK]?', text)
    if m:
        max_ctx = int(float(m.group(1)) * (1000 if 'k' in m.group(0).lower() else 1))
    else:
        m = re.search(r'context[^\d]*(\d+(?:\.\d+)?)\s*[kK]?', lower)
        if m:
            max_ctx = int(float(m.group(1)) * (1000 if 'k' in m.group(0).lower() else 1))

    m = re.search(r'tpot[^\d]*(\d+)\s*ms', lower)
    if m:
        tpot_limit = int(m.group(1))

    path_match = re.search(r'(/[\w./\-]+)', text)
    weight_path = path_match.group(1) if path_match else None

    local_weights = None
    if '本地有权重' in text or '本地权重' in text:
        local_weights = True
    elif '网上的权重' in text or '远端权重' in text:
        local_weights = False

    existing_quant_weights = None
    if '已有w4a4权重' in text.lower() or '已有 w4a4 权重' in text.lower() or '给了w4a4的权重' in text.lower():
        existing_quant_weights = True
    elif '先量化' in text or '帮我量化' in text:
        existing_quant_weights = False

    accepts_experimental = any(k in text for k in ['试一下', '跑不起来也没关系', '实验', '候选也可以'])

    req = ParsedRequest(
        raw_text=text,
        intent=intent,
        model_name=model_name,
        model_size_b=size_b,
        hardware=hardware,
        cards=cards,
        quantization=quant,
        weight_path=weight_path,
        wants_script='脚本' in text or 'script' in lower,
        wants_command='命令' in text or 'command' in lower,
        objective=objective,
        average_input_len=avg_in,
        average_output_len=avg_out,
        max_context=max_ctx,
        accepts_experimental=accepts_experimental,
        has_existing_quantized_weights=existing_quant_weights,
        single_instance=True,
        tpot_limit_ms=tpot_limit,
        local_weights=local_weights,
    )

    if overrides:
        for k, v in overrides.items():
            setattr(req, k, v)
    return req
