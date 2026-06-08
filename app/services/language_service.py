from __future__ import annotations

import re

BANGLA_RE = re.compile(r"[\u0980-\u09FF]")
HINDI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")
LATIN_TOKEN_RE = re.compile(r"[A-Za-z']+")

BANGLISH_MARKERS = {
    "ache",
    "ajke",
    "bangladeshe",
    "bolche",
    "bolse",
    "chilo",
    "chiriakhanae",
    "dhakai",
    "dhakar",
    "ekhon",
    "hobe",
    "hoyeche",
    "keu",
    "korche",
    "kore",
    "mohish",
    "na",
    "naki",
    "namer",
    "shob",
    "theke",
}

HINGLISH_MARKERS = {
    "abhi",
    "aisa",
    "aisa",
    "apna",
    "bahut",
    "bilkul",
    "hai",
    "hoga",
    "kaise",
    "kya",
    "kyun",
    "mera",
    "nahi",
    "sach",
    "shayad",
    "wala",
    "wahan",
    "yahaan",
}


def _latin_tokens(value: str) -> list[str]:
    return [token.lower() for token in LATIN_TOKEN_RE.findall(value)]


def _marker_score(tokens: list[str], markers: set[str]) -> int:
    return sum(1 for token in tokens if token in markers)


def detect_language(value: str) -> str:
    has_bangla = bool(BANGLA_RE.search(value))
    has_hindi = bool(HINDI_RE.search(value))
    has_latin = bool(LATIN_RE.search(value))

    if has_bangla and has_latin and not has_hindi:
        return "Banglish"
    if has_hindi and has_latin and not has_bangla:
        return "Hinglish"
    if has_bangla and has_hindi:
        return "Mixed"
    if has_bangla:
        return "Bangla"
    if has_hindi:
        return "Hindi"
    if has_latin:
        tokens = _latin_tokens(value)
        banglish_score = _marker_score(tokens, BANGLISH_MARKERS)
        hinglish_score = _marker_score(tokens, HINGLISH_MARKERS)
        if banglish_score >= 2 and banglish_score > hinglish_score:
            return "Banglish"
        if hinglish_score >= 2 and hinglish_score > banglish_score:
            return "Hinglish"
        if banglish_score >= 2 and hinglish_score >= 2:
            return "Mixed"
        return "English"
    return "Mixed"
