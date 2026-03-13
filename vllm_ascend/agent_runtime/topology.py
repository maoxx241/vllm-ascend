from __future__ import annotations

import re

CARD_COUNT_RE = re.compile(r"(?:(?<!\d)(\d+)\s*卡|(?<!\d)(\d+)\s*cards?\b)", re.I)


def detect_requested_card_count(text: str) -> int | None:
    match = CARD_COUNT_RE.search(text)
    if not match:
        if re.search(r"(single[- ]?card|单卡)", text, re.I):
            return 1
        return None
    value = match.group(1) or match.group(2)
    if value is None:
        return None
    return int(value)


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
