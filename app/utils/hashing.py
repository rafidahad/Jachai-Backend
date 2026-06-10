from __future__ import annotations

import hashlib


def normalize_for_hash(value: str) -> str:
    collapsed = " ".join(value.lower().strip().split())
    return collapsed


def normalized_hash(value: str) -> str:
    normalized = normalize_for_hash(value)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
