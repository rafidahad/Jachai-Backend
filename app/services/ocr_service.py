from __future__ import annotations

import re

from app.core.config import settings
from app.services.nvidia_llm_service import extract_text_with_kimi_ocr
from app.services.text_cleaning_service import clean_text
from app.utils.errors import AppError

_SHORT_UI_NOISE = {
    "follow",
    "following",
    "like",
    "reply",
    "share",
    "comment",
    "comments",
    "subscribe",
    "subscribed",
    "search",
    "menu",
    "home",
    "send",
    "copy",
    "result",
    "response",
    "preview",
    "json",
    "code",
    "view source",
    "open in new",
}
_UI_NOISE_PATTERN = re.compile(
    r"\b("
    r"follow|following|like|reply|share|comment|comments|subscribe|subscribed|"
    r"search|menu|home|send|sponsored|verified|views?|followers?|following|"
    r"copy|result|response|preview|json|code|open in new|view source|"
    r"api playground|privacy policy|terms of use|cookie settings"
    r")\b",
    re.IGNORECASE,
)
_TIMESTAMP_ONLY_PATTERN = re.compile(
    r"^(?:"
    r"\d{1,2}:\d{2}(?:\s?[ap]m)?|"
    r"\d+\s*(?:s|m|h|d|w|mo|y)|"
    r"today|yesterday|just now|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,?\s+\d{4})?"
    r")$",
    re.IGNORECASE,
)


def _looks_like_usable_ocr(text: str) -> bool:
    if len(text) < settings.ocr_min_characters:
        return False
    useful_characters = sum(1 for char in text if char.isalnum())
    return useful_characters / max(len(text), 1) >= 0.35


def _is_noise_token(token: str) -> bool:
    normalized = token.strip("()[]{}<>.,:;!?\"'").lower()
    if not normalized:
        return True
    if normalized in _SHORT_UI_NOISE:
        return True
    if normalized.startswith(("http://", "https://", "www.", "@", "#")):
        return True
    if _TIMESTAMP_ONLY_PATTERN.fullmatch(normalized):
        return True
    return False


def _trim_edge_noise_tokens(line: str) -> str:
    tokens = line.split()
    while tokens and _is_noise_token(tokens[0]):
        tokens.pop(0)
    while tokens and _is_noise_token(tokens[-1]):
        tokens.pop()
    return " ".join(tokens)


def _clean_ocr_lines(text: str) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for raw_line in str(text).splitlines():
        line = _trim_edge_noise_tokens(clean_text(raw_line))
        if not line:
            continue
        normalized = line.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        lines.append(line)
    return lines


def _is_probable_ui_noise_line(line: str) -> bool:
    lowered = line.lower().strip()
    word_count = len(lowered.split())
    if lowered.startswith(("http://", "https://", "www.")):
        return True
    if lowered.startswith(("@", "#")) and word_count <= 5:
        return True
    if lowered in _SHORT_UI_NOISE:
        return True
    if _TIMESTAMP_ONLY_PATTERN.fullmatch(lowered):
        return True
    if sum(char.isalnum() for char in line) / max(len(line), 1) < 0.35:
        return True
    ui_hits = len(_UI_NOISE_PATTERN.findall(lowered))
    if ui_hits >= 2 and word_count <= 8:
        return True
    if ui_hits >= 1 and word_count <= 4:
        return True
    return False


def _claim_line_score(line: str) -> int:
    lowered = line.lower()
    words = lowered.split()
    score = min(len(words), 12)
    if len(words) >= 4:
        score += 4
    if any(marker in line for marker in (".", "?", "!", ":", "।")):
        score += 2
    if any(char.isdigit() for char in line):
        score += 1
    if lowered.startswith(("breaking", "claim", "fact", "headline")):
        score += 1
    if lowered.startswith(("@", "#")):
        score -= 4
    if "http://" in lowered or "https://" in lowered or "www." in lowered:
        score -= 4
    if _is_probable_ui_noise_line(line):
        score -= 8
    return score


def prepare_ocr_text_for_claim_extraction(text: str) -> str:
    lines = _clean_ocr_lines(text)
    if not lines:
        return ""
    filtered_lines = [line for line in lines if not _is_probable_ui_noise_line(line)]
    candidates = filtered_lines or lines
    if len(candidates) == 1:
        return candidates[0]

    scored_candidates = [
        (index, _claim_line_score(line), line)
        for index, line in enumerate(candidates)
    ]
    sorted_candidates = sorted(
        scored_candidates,
        key=lambda item: (item[1], len(item[2])),
        reverse=True,
    )
    best_index, best_score, _ = sorted_candidates[0]
    if best_score <= 0:
        return clean_text(" ".join(candidates[: min(3, len(candidates))]))

    selected_indexes = {best_index}
    for neighbor in (best_index - 1, best_index + 1):
        if 0 <= neighbor < len(candidates) and _claim_line_score(candidates[neighbor]) >= 2:
            selected_indexes.add(neighbor)

    strong_threshold = max(6, best_score - 2)
    for index, score, _ in sorted_candidates[1:]:
        if score >= strong_threshold:
            selected_indexes.add(index)

    return clean_text(" ".join(candidates[index] for index in sorted(selected_indexes)))


async def extract_text_from_image(
    image_bytes: bytes,
    *,
    mime_type: str | None = None,
) -> str:
    text = await extract_text_with_kimi_ocr(image_bytes, mime_type=mime_type)
    if not text:
        raise AppError(
            status_code=422,
            code="OCR_FAILED",
            message="Kimi OCR could not extract text from the image.",
        )
    if not _looks_like_usable_ocr(text):
        raise AppError(
            status_code=422,
            code="OCR_FAILED",
            message="Kimi OCR could not extract enough text from the image.",
        )
    return text
