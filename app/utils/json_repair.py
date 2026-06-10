from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any


def load_json_with_repair(payload: str) -> dict[str, Any]:
    try:
        return json.loads(payload)
    except JSONDecodeError:
        cleaned = payload.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])
