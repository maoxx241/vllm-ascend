from __future__ import annotations

import re
from .models import RequestFacts


def _parse_token_amount(raw: str | None) -> int | None:
    if not raw:
        return None
    m = re.search(r'(\d+(?:\.\d+)?)\s*([kK])?', raw)
    if not m:
        return None
    value = float(m.group(1))
    if m.group(2):
        value *= 1000
    return int(value)


def _search_path(text: str) -> str | None:
    m = re.search(r'(/[A-Za-z0-9_./\-]+)', text)
    return m.group(1) if m else None


def parse_request(text: str) -> RequestFacts:
    lower = text.lower()
    req = RequestFacts(raw_text=text)

    if any(k in lower for k in ['部署', '命令', '脚本', 'serve']):
        req.intent = 'deployment'

    if 'owen' in lower:
        req.alias_suspect = 'owen'

    m = re.search(r'glm\s*4\.(5|6|7)', lower)
    if m:
        v = m.group(1)
        req.model_family = 'glm4.x'
        req.model_variant = f'glm4.{v}'
    elif 'glm4.x' in lower:
        req.model_family = 'glm4.x'
        req.model_variant = 'glm4.x'
    elif 'qwen3.5' in lower or 'qwen 3.5' in lower or 'qwen3_5' in lower:
        req.model_family = 'qwen3.5'
        req.model_variant = 'qwen3.5'
    elif 'qwen2 vl' in lower or 'qwen2-vl' in lower:
        req.model_family = 'qwen2-vl'
        req.model_variant = 'qwen2-vl'
    elif 'deepseek v3.1' in lower or 'deepseek-v3.1' in lower:
        req.model_family = 'deepseek-v3.1'
        req.model_variant = 'deepseek-v3.1'
    elif re.search(r'qwen\s*3\b', lower) or 'qwen3' in lower:
        req.model_family = 'qwen3'
        req.model_variant = 'qwen3'

    size = re.search(r'(\d+(?:\.\d+)?)\s*[bB]\b', text)
    if size:
        req.model_size_b = float(size.group(1))

    if '310p' in lower:
        req.hardware = '310P'
    elif 'a3' in lower:
        req.hardware = 'A3'
    elif 'a2' in lower:
        req.hardware = 'A2'
    elif 'a5' in lower:
        req.hardware = 'A5'

    if '双卡' in text:
        req.cards = 2
    elif '单卡' in text:
        req.cards = 1
    m = re.search(r'(\d+)\s*卡', text)
    if m:
        req.cards = int(m.group(1))

    req.weight_path = _search_path(text)

    for q in ['w4a4', 'w4a8', 'w8a8', 'bf16', 'fp16', 'float16']:
        if q in lower:
            req.quantization = q.upper()
            break

    if any(k in lower for k in ['已有权重', '权重已在本地', '本地有权重']):
        req.existing_quantized_weights = True if req.quantization else None
    if any(k in lower for k in ['帮我量化', '先量化', '量化一下']):
        req.existing_quantized_weights = False

    pair = re.search(r'(\d+(?:\.\d+)?\s*[kK]?)\s*输入\s*(\d+(?:\.\d+)?\s*[kK]?)\s*输出', text)
    pair2 = re.search(r'平均输入\s*(\d+(?:\.\d+)?\s*[kK]?).{0,8}?输出\s*(\d+(?:\.\d+)?\s*[kK]?)', text)
    pair3 = re.search(r'input\s*(\d+(?:\.\d+)?\s*[kK]?).{0,8}?output\s*(\d+(?:\.\d+)?\s*[kK]?)', lower)
    if pair:
        req.avg_input_tokens = _parse_token_amount(pair.group(1))
        req.avg_output_tokens = _parse_token_amount(pair.group(2))
    elif pair2:
        req.avg_input_tokens = _parse_token_amount(pair2.group(1))
        req.avg_output_tokens = _parse_token_amount(pair2.group(2))
    elif pair3:
        req.avg_input_tokens = _parse_token_amount(pair3.group(1))
        req.avg_output_tokens = _parse_token_amount(pair3.group(2))

    ctx = re.search(r'(?:最大上下文|max(?:imum)?\s*context|max-model-len|max model len)[^\n]{0,10}?(\d+(?:\.\d+)?\s*[kK]?)', lower)
    if ctx:
        req.max_context_tokens = _parse_token_amount(ctx.group(1))

    tpot = re.search(r'(?:tpot|sla)[^\n]{0,8}?(\d+(?:\.\d+)?)\s*ms', lower)
    if tpot:
        req.tpot_ms = float(tpot.group(1))

    if '低时延' in text or 'latency' in lower:
        req.explicit_pref = 'low_latency'
    elif '高吞吐' in text or 'throughput' in lower:
        req.explicit_pref = 'high_throughput'

    if any(k in text for k in ['试一下', '跑不起来也没关系', '接受风险', '实验']) or 'experimental' in lower:
        req.accepts_candidate = True

    return req
