"""Locale vocabulary and canonicalization shared by the classifier and the LLM router.

Both routing tiers must produce the *same* slot value for the same request ("es-MX", never
"es_mx", "ES-mx" or a bare "es"), otherwise downstream behaviour - and exact-match evaluation -
depends on which tier happened to answer.
"""

from __future__ import annotations

import re

LANGUAGES = {
    "spanish": "es", "german": "de", "french": "fr", "portuguese": "pt", "japanese": "ja",
    "english": "en", "italian": "it", "dutch": "nl", "hindi": "hi", "chinese": "zh",
    "korean": "ko", "polish": "pl",
}  # fmt: skip
COUNTRIES = {
    "mexic": ("MX", "es"), "spain": ("ES", "es"), "brazil": ("BR", "pt"), "portugal": ("PT", "pt"),
    "germany": ("DE", "de"), "france": ("FR", "fr"), "japan": ("JP", "ja"),
    "united kingdom": ("GB", "en"), " uk": ("GB", "en"), "britain": ("GB", "en"),
    "united states": ("US", "en"), " usa": ("US", "en"), "india": ("IN", "en"),
    "canada": ("CA", "en"), "argentin": ("AR", "es"), "colombia": ("CO", "es"),
    "chile": ("CL", "es"), "netherlands": ("NL", "nl"), "ital": ("IT", "it"),
}  # fmt: skip
# Market assumed when only a language is named.
DEFAULT_REGION = {
    "es": "ES", "de": "DE", "fr": "FR", "pt": "BR", "ja": "JP", "en": "US", "it": "IT",
    "nl": "NL", "hi": "IN", "zh": "CN", "ko": "KR", "pl": "PL",
}  # fmt: skip
_CODE = re.compile(r"^\s*([a-zA-Z]{2,3})(?:[-_\s]+([a-zA-Z]{2}|\d{3}))?\s*$")


def extract_locale(text: str) -> str | None:
    """Locale from natural language: 'Spanish for the Mexican market' -> 'es-MX'."""
    low = f" {text.lower()} "
    lang = next((code for word, code in LANGUAGES.items() if word in low), None)
    region, region_lang = next(
        ((r, rl) for word, (r, rl) in COUNTRIES.items() if word in low), (None, None)
    )
    if lang is None and region_lang is None:
        return None
    lang = lang or region_lang
    return f"{lang}-{region or DEFAULT_REGION.get(lang, lang.upper())}"


def canonical_locale(value: str | None) -> str:
    """Normalize any locale-ish value to 'll-RR'; '' when unrecognizable.

    'es_mx' -> 'es-MX', 'DE' -> 'de-DE', 'Spanish (Mexico)' -> 'es-MX', '' -> ''.
    """
    if not value or not value.strip():
        return ""
    match = _CODE.match(value)
    if match:
        lang = match.group(1).lower()
        region = (match.group(2) or DEFAULT_REGION.get(lang, lang)).upper()
        return f"{lang}-{region}"
    return extract_locale(value) or ""
