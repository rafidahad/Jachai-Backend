from __future__ import annotations

import re

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s-]{8,}\d)(?!\d)")


def mask_pii(value: str) -> str:
    value = EMAIL_RE.sub("[EMAIL_REDACTED]", value)
    value = PHONE_RE.sub("[PHONE_REDACTED]", value)
    return value
