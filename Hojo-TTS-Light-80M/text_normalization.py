"""Text normalization for characters the released tokenizer cannot speak reliably."""

from __future__ import annotations

import re

_NUMBER_RE = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?")
_HAN_RE = re.compile(r"[\u3400-\u9fff]")

_EN_ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
)
_EN_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_ZH_DIGITS = "零一二三四五六七八九"


def _english_integer(value: int) -> str:
    if value < 20:
        return _EN_ONES[value]
    if value < 100:
        tens, ones = divmod(value, 10)
        return _EN_TENS[tens] + (f" {_EN_ONES[ones]}" if ones else "")
    if value < 1000:
        hundreds, rest = divmod(value, 100)
        return f"{_EN_ONES[hundreds]} hundred" + (f" {_english_integer(rest)}" if rest else "")
    for scale, name in ((1_000_000_000, "billion"), (1_000_000, "million"), (1000, "thousand")):
        if value >= scale:
            head, rest = divmod(value, scale)
            return f"{_english_integer(head)} {name}" + (f" {_english_integer(rest)}" if rest else "")
    return " ".join(_EN_ONES[int(digit)] for digit in str(value))


def _english_number(raw: str) -> str:
    negative = raw.startswith("-")
    raw = raw.lstrip("-").replace(",", "")
    whole, dot, fraction = raw.partition(".")
    spoken = _english_integer(int(whole))
    if dot:
        spoken += " point " + " ".join(_EN_ONES[int(digit)] for digit in fraction)
    return ("minus " if negative else "") + spoken


def _chinese_section(value: int) -> str:
    units = ("", "十", "百", "千")
    result = ""
    zero_pending = False
    for position in range(3, -1, -1):
        divisor = 10 ** position
        digit, value = divmod(value, divisor)
        if digit:
            if zero_pending and result:
                result += "零"
            if not (digit == 1 and position == 1 and not result):
                result += _ZH_DIGITS[digit]
            result += units[position]
            zero_pending = False
        elif result and value:
            zero_pending = True
    return result or "零"


def _chinese_integer(value: int) -> str:
    if value == 0:
        return "零"
    parts = []
    for scale, unit in ((100_000_000, "亿"), (10_000, "万")):
        if value >= scale:
            head, value = divmod(value, scale)
            parts.append(_chinese_section(head) + unit)
            if value and value < scale // 10:
                parts.append("零")
    if value:
        parts.append(_chinese_section(value))
    return "".join(parts)


def _chinese_number(raw: str) -> str:
    negative = raw.startswith("-")
    raw = raw.lstrip("-").replace(",", "")
    whole, dot, fraction = raw.partition(".")
    spoken = _chinese_integer(int(whole))
    if dot:
        spoken += "点" + "".join(_ZH_DIGITS[int(digit)] for digit in fraction)
    return ("负" if negative else "") + spoken


def normalize_text(text: str) -> str:
    """Expand Arabic numerals and percent signs before tokenizer encoding."""
    use_chinese = bool(_HAN_RE.search(text))

    def replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        percent = raw.endswith("%")
        number = raw[:-1] if percent else raw
        spoken = _chinese_number(number) if use_chinese else _english_number(number)
        if percent:
            return f"百分之{spoken}" if use_chinese else f"{spoken} percent"
        return spoken

    return _NUMBER_RE.sub(replace, text).replace("%", "百分比" if use_chinese else " percent ")
