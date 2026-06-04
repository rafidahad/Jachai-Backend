from __future__ import annotations

import re

BANGLA_RE = re.compile(r"[\u0980-\u09FF]")
HINDI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")


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
    return "English" if has_latin else "Mixed"
