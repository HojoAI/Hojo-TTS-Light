"""Text normalization via WeTextProcessing (Chinese / English)."""

from __future__ import annotations

import re
import threading

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_PINYIN_TOKEN_RE = re.compile(r"<zh_[^<>]+>")
_lock = threading.Lock()
_zh_normalizer = None
_en_normalizer = None


def detect_tn_language(text: str) -> str:
    """Return ``zh`` when the text contains CJK, otherwise ``en``."""
    return "zh" if _CJK_RE.search(text) else "en"


def _get_zh_normalizer():
    global _zh_normalizer
    if _zh_normalizer is None:
        from tn.chinese.normalizer import Normalizer

        _zh_normalizer = Normalizer(remove_erhua=False)
    return _zh_normalizer


def _get_en_normalizer():
    global _en_normalizer
    if _en_normalizer is None:
        from tn.english.normalizer import Normalizer

        _en_normalizer = Normalizer()
    return _en_normalizer


def _normalize_keeping_pinyin(text: str, normalizer) -> str:
    """Run TN on the text around ``<zh_*>`` tokens, leaving those tokens intact."""
    parts = _PINYIN_TOKEN_RE.split(text)
    tokens = _PINYIN_TOKEN_RE.findall(text)
    if not tokens:
        return normalizer.normalize(text)
    pieces: list[str] = []
    for index, part in enumerate(parts):
        if part:
            pieces.append(normalizer.normalize(part))
        if index < len(tokens):
            pieces.append(tokens[index])
    return "".join(pieces)


def normalize_text(text: str, *, language: str = "auto") -> str:
    """Normalize ``text`` for TTS.

    ``language`` is ``auto``, ``zh``, or ``en``. ``auto`` uses Chinese TN when
    the string contains CJK characters, otherwise English TN. Tone digits inside
    ``<zh_*>`` pinyin tokens are left unchanged.
    """
    if text is None:
        return ""
    raw = str(text).strip()
    if not raw:
        return raw

    lang = language if language != "auto" else detect_tn_language(raw)
    if lang not in ("zh", "en"):
        raise ValueError(f"language must be auto, zh, or en, got {language!r}.")

    with _lock:
        normalizer = _get_zh_normalizer() if lang == "zh" else _get_en_normalizer()
        return _normalize_keeping_pinyin(raw, normalizer)
