"""
Multilingual language detection and utilities for Indian legal consultation.

Supports major Indian languages (Telugu, Hindi, Tamil, Kannada, Marathi, Bengali,
Gujarati, Malayalam, Punjabi, Odia, English) and mixed script/transliteration.
"""

from __future__ import annotations

import re

# Supported ISO-639-1 language codes mapped to human names
SUPPORTED_LANGUAGES: dict[str, str] = {
    "te": "Telugu",
    "hi": "Hindi",
    "ta": "Tamil",
    "kn": "Kannada",
    "mr": "Marathi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
    "en": "English",
}

# Regex ranges for Indian native scripts
_SCRIPT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"[\u0C00-\u0C7F]"), "te"),  # Telugu
    (re.compile(r"[\u0B80-\u0BFF]"), "ta"),  # Tamil
    (re.compile(r"[\u0C80-\u0CFF]"), "kn"),  # Kannada
    (re.compile(r"[\u0980-\u09FF]"), "bn"),  # Bengali
    (re.compile(r"[\u0A80-\u0AFF]"), "gu"),  # Gujarati
    (re.compile(r"[\u0D00-\u0D7F]"), "ml"),  # Malayalam
    (re.compile(r"[\u0A00-\u0A7F]"), "pa"),  # Gurmukhi / Punjabi
    (re.compile(r"[\u0B00-\u0B7F]"), "or"),  # Odia
    (re.compile(r"[\u0900-\u097F]"), "hi"),  # Devanagari (Hindi / Marathi)
]

# Common Latin-script transliterated Indic tokens
_TELUGLISH_TOKENS = {
    "nenu", "naa", "naaku", "chesanu", "chesaru", "ivvaledhu", "ledhu",
    "jeetham", "cheyali", "meeru", "undhi", "unnaru", "teliyadhu", "daggara",
    "pani", "manishi", "raavadam", "emiti", "evaru", "ippudu",
}

_HINGLISH_TOKENS = {
    "mera", "meri", "mere", "mujhe", "kya", "nahin", "nahi", "diya", "kiye",
    "karein", "karna", "hoga", "raha", "rahe", "chahiye", "paisa", "paise",
    "nokri", "naukri", "kaam", "saal", "mahine", "aap", "hum",
}

_TAMGLISH_TOKENS = {
    "naan", "enakku", "pannanum", "pannanga", "sambalam", "theriyathu",
    "irukku", "illa", "illai", "kudukkala", "velai", "solranga",
}


def detect_language(text: str, default: str = "en") -> str:
    """
    Detect the primary language of the input text.
    
    1. Checks for Indic Unicode script characters first (highest confidence).
    2. Checks for transliterated Indic tokens in Latin script.
    3. Falls back to default (English).
    """
    if not text or not text.strip():
        return default

    cleaned = text.strip()

    # 1. Check Unicode script ranges
    for pattern, lang_code in _SCRIPT_PATTERNS:
        # If there are at least 2 script characters or significant presence
        matches = pattern.findall(cleaned)
        if len(matches) >= 2 or (len(cleaned) <= 10 and len(matches) >= 1):
            return lang_code

    # 2. Check transliterated tokens in Latin script
    words = set(re.findall(r"\b[a-zA-Z]+\b", cleaned.lower()))
    if not words:
        return default

    telugu_overlap = words & _TELUGLISH_TOKENS
    if len(telugu_overlap) >= 2 or (len(words) <= 5 and len(telugu_overlap) >= 1):
        return "te"

    hinglish_overlap = words & _HINGLISH_TOKENS
    if len(hinglish_overlap) >= 2 or (len(words) <= 5 and len(hinglish_overlap) >= 1):
        return "hi"

    tamglish_overlap = words & _TAMGLISH_TOKENS
    if len(tamglish_overlap) >= 2 or (len(words) <= 5 and len(tamglish_overlap) >= 1):
        return "ta"

    return default


def normalize_language_code(code: str | None, default: str = "en") -> str:
    """Normalize input language code or return default."""
    if not code or code.strip().lower() in ("auto", "none", ""):
        return default
    c = code.strip().lower()
    return c if c in SUPPORTED_LANGUAGES else default


def get_language_name(code: str) -> str:
    """Get human-readable language name from ISO code."""
    return SUPPORTED_LANGUAGES.get(code, "English")


def is_indic_language(code: str) -> bool:
    """Check if the language is an Indian regional language."""
    return code in SUPPORTED_LANGUAGES and code != "en"


def resolve_turn_language(text: str, current_lang: str = "en") -> str:
    """
    Resolve the active language for the current turn, supporting dynamic language switching.
    
    1. If user uses native Indic script (Telugu, Hindi, Tamil, etc.), switch to that language.
    2. If user uses transliterated tokens (Hinglish, Teluglish), switch to that language.
    3. If user sends a short factual reply (1-3 words like 'Yes', 'No', '2 months', 'Hyderabad'),
       retain the current conversation language so flow is not disrupted.
    4. If user writes a longer English message (> 3 words) without Indic script/tokens, switch to English.
    """
    if not text or not text.strip():
        return current_lang or "en"

    cleaned = text.strip()

    # 1. Native Indic script check (immediate switch)
    for pattern, lang_code in _SCRIPT_PATTERNS:
        matches = pattern.findall(cleaned)
        if len(matches) >= 2 or (len(cleaned) <= 10 and len(matches) >= 1):
            return lang_code

    # 2. Check transliterated Indic tokens
    words = re.findall(r"\b[a-zA-Z]+\b", cleaned.lower())
    words_set = set(words)

    if words_set & _TELUGLISH_TOKENS:
        return "te"
    if words_set & _HINGLISH_TOKENS:
        return "hi"
    if words_set & _TAMGLISH_TOKENS:
        return "ta"

    # 3. Short answers (e.g., "Yes", "2 months", "Bangalore") retain active regional language
    if len(words) <= 3 and current_lang and current_lang != "en":
        return current_lang

    # 4. Otherwise user wrote in English
    return "en"
