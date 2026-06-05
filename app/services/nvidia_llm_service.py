from __future__ import annotations

import base64
import json
from typing import Any
from uuid import UUID

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.verdict_schema import LLMVerdictSchema, VisionOCRSchema
from app.services.ai_model_router import NVIDIAModelTask, get_model_for_task, is_task_enabled
from app.services.nvidia_client import call_nvidia_chat
from app.services.text_cleaning_service import clean_text

logger = get_logger(__name__)

SYSTEM_PROMPT = """
You are a misinformation verification assistant for South Asia.
Return strict JSON only with keys:
verdict, confidence, explanation, reasoning, summary, source_ids.

Rules:
- Use only the supplied evidence.
- If evidence is weak or conflicting, choose "Not Enough Evidence" or "Misleading".
- Confidence must be between 0 and 1.
- source_ids must contain UUID strings chosen from the evidence list only.
""".strip()

VISION_SYSTEM_PROMPT = """
You are a careful OCR fallback assistant.
Extract only the readable factual text from the provided image.
Return strict JSON only with key: extracted_text.

Rules:
- Preserve the claim meaning while fixing obvious OCR formatting noise.
- Do not summarize, explain, or fact-check.
- Do not invent missing text.
- If the image text is unreadable, return the longest clearly readable text segment you can recover.
""".strip()


def fallback_verdict(reason: str, source_ids: list[UUID]) -> LLMVerdictSchema:
    return LLMVerdictSchema(
        verdict="Not Enough Evidence",
        confidence=0.2,
        explanation=reason,
        reasoning="The system could not safely complete evidence-grounded reasoning.",
        summary="Not enough evidence is available to verify this claim right now.",
        source_ids=source_ids,
    )


async def generate_verdict(
    claim_text: str,
    language: str,
    evidence: list[dict[str, Any]],
) -> LLMVerdictSchema:
    source_ids = [UUID(item["source_id"]) for item in evidence if item.get("source_id")]
    if not settings.nvidia_api_key:
        return fallback_verdict("NVIDIA API key is not configured.", source_ids)
    if not get_model_for_task(NVIDIAModelTask.CLAIM_REASONING):
        return fallback_verdict("No NVIDIA reasoning model is configured.", source_ids)

    evidence_blob = json.dumps(evidence, default=str)
    user_prompt = (
        f"Language: {language}\n"
        f"Claim: {claim_text}\n"
        f"Evidence: {evidence_blob}\n"
        "Respond with JSON only."
    )

    try:
        content = await call_nvidia_chat(
            task=NVIDIAModelTask.CLAIM_REASONING,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
            max_tokens=1200,
        )
        payload = json.loads(content or "{}")
        return LLMVerdictSchema.model_validate(payload)
    except Exception:
        logger.exception("nvidia_reasoning_failed")

    return fallback_verdict("The language model could not return a valid verdict.", source_ids)


async def extract_text_with_vision_fallback(
    image_bytes: bytes,
    *,
    mime_type: str | None = None,
) -> str | None:
    if not is_task_enabled(NVIDIAModelTask.IMAGE_OCR_FALLBACK):
        return None

    content_type = mime_type if mime_type and mime_type.startswith("image/") else "image/png"
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    messages = [
        {"role": "system", "content": VISION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Extract the text from this screenshot and return strict JSON only "
                        "with the key extracted_text."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{content_type};base64,{image_b64}"},
                },
            ],
        },
    ]

    try:
        content = await call_nvidia_chat(
            task=NVIDIAModelTask.IMAGE_OCR_FALLBACK,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=600,
        )
        payload = json.loads(content or "{}")
        result = VisionOCRSchema.model_validate(payload)
        extracted_text = clean_text(result.extracted_text)
        if len(extracted_text) < settings.ocr_min_characters:
            return None
        return extracted_text
    except Exception:
        logger.exception("nvidia_vision_fallback_failed")
        return None
