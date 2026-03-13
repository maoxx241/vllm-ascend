from __future__ import annotations

import re

CARD_COUNT_RE = re.compile(r"(?:(?<!\d)(\d+)\s*卡|(?<!\d)(\d+)\s*cards?\b)", re.I)
CHINESE_CARD_COUNT_RE = re.compile(r"(一|二|两|三|四|五|六|七|八|九|十)\s*卡")
CHINESE_CARD_VALUES = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def detect_requested_card_count(text: str) -> int | None:
    match = CARD_COUNT_RE.search(text)
    if match:
        value = match.group(1) or match.group(2)
        if value is not None:
            return int(value)
    chinese_match = CHINESE_CARD_COUNT_RE.search(text)
    if chinese_match:
        return CHINESE_CARD_VALUES[chinese_match.group(1)]
    if re.search(r"(single[- ]?card|单卡)", text, re.I):
        return 1
    return None


def requested_card_count_from_features(features: list[str]) -> int | None:
    for feature in features:
        if feature.startswith("cards_"):
            suffix = feature.removeprefix("cards_")
            if suffix.isdigit():
                return int(suffix)
    if "single_card" in features:
        return 1
    return None


def logical_npus_for_hw(hw: str | None, physical_cards: int | None) -> int | None:
    if physical_cards is None:
        return None
    if hw == "A3":
        return physical_cards * 2
    return physical_cards


def visible_devices(count: int) -> str:
    return ",".join(str(index) for index in range(count))


__all__ = [
    "detect_requested_card_count",
    "logical_npus_for_hw",
    "requested_card_count_from_features",
    "visible_devices",
]
