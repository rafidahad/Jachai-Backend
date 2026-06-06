from __future__ import annotations

from enum import Enum

from app.core.config import settings


class NVIDIAModelTask(str, Enum):
    CLAIM_EXTRACTION = "claim_extraction"
    SEARCH_QUERY_GENERATION = "search_query_generation"
    CLAIM_REASONING = "claim_reasoning"
    EVIDENCE_RERANKING = "evidence_reranking"
    IMAGE_OCR_FALLBACK = "image_ocr_fallback"


def get_model_for_task(task: NVIDIAModelTask) -> str | None:
    if task == NVIDIAModelTask.CLAIM_EXTRACTION:
        return settings.active_nvidia_claim_extraction_model
    if task == NVIDIAModelTask.SEARCH_QUERY_GENERATION:
        return settings.active_nvidia_query_model
    if task == NVIDIAModelTask.CLAIM_REASONING:
        return settings.active_nvidia_reasoning_model
    if task == NVIDIAModelTask.EVIDENCE_RERANKING:
        return settings.active_nvidia_rerank_model
    if task == NVIDIAModelTask.IMAGE_OCR_FALLBACK:
        return settings.active_nvidia_vision_model if settings.nvidia_vision_enabled else None
    return None


def is_task_enabled(task: NVIDIAModelTask) -> bool:
    return get_model_for_task(task) is not None
