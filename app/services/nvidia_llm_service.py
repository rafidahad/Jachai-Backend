from __future__ import annotations

import base64
from typing import Any
from uuid import UUID

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.verdict_schema import LLMVerdictSchema, VisionOCRSchema
from app.services.ai_model_router import NVIDIAModelTask, get_model_for_task, is_task_enabled
from app.services.evidence_context_builder import build_evidence_context
from app.services.nvidia_client import call_nvidia_chat
from app.services.text_cleaning_service import clean_text
from app.utils.json_repair import load_json_with_repair

logger = get_logger(__name__)

SYSTEM_PROMPT = """
You are JachAI, a strict multilingual fact-checking assistant for South Asian misinformation.

The user input may contain Bangla, English, Hindi, Banglish, Hinglish, OCR noise, social media slang, or forwarded-message formatting.

Your job:
1. Extract the core factual claim.
2. Identify the language/style.
3. Classify the claim category.
4. Compare the claim only against the provided evidence.
5. Return a verdict.
6. Write a short user-facing explanation.

Important rules:
- Use ONLY the provided evidence.
- Do not use your internal knowledge as proof.
- If evidence is missing, weak, irrelevant, or conflicting, return "Not Enough Evidence".
- Do not invent sources.
- Do not overstate certainty.
- Keep the explanation short and clear.
- Return valid JSON only.
- Do not include markdown.
- Do not include extra text outside JSON.

Allowed verdicts:
- Likely True
- Likely False
- Misleading
- Not Enough Evidence

Allowed confidence labels:
- Low
- Medium
- High

Return JSON exactly in this schema:
{
  "extracted_claim": "string",
  "detected_language": "Bangla | English | Hindi | Banglish | Hinglish | Mixed | Unknown",
  "category": "Politics | Health | Disaster | Crime | Finance | Cybersecurity | Entertainment | Religion | Education | Other",
  "verdict": "Likely True | Likely False | Misleading | Not Enough Evidence",
  "confidence": 0.0,
  "confidence_label": "Low | Medium | High",
  "explanation": "string",
  "user_response": "string",
  "used_source_ids": ["string"]
}
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

ALLOWED_VERDICTS = {"Likely True", "Likely False", "Misleading", "Not Enough Evidence"}
ALLOWED_CONFIDENCE_LABELS = {"Low", "Medium", "High"}


def _default_user_response(reason: str) -> str:
    return f"JachAI could not confidently verify this claim yet. {reason}"


def fallback_verdict(
    reason: str,
    source_ids: list[UUID],
    *,
    claim_text: str,
    language: str,
) -> LLMVerdictSchema:
    extracted_claim = clean_text(claim_text) or "Claim text unavailable."
    detected_language = clean_text(language) or "Unknown"
    return LLMVerdictSchema(
        extracted_claim=extracted_claim,
        detected_language=detected_language,
        category="Other",
        verdict="Not Enough Evidence",
        confidence=0.2,
        confidence_label="Low",
        explanation=reason,
        user_response=_default_user_response(reason),
        used_source_ids=source_ids,
    )


def _confidence_label_from_score(score: Any) -> str:
    try:
        numeric = float(score)
    except (TypeError, ValueError):
        return "Low"
    if numeric >= 0.8:
        return "High"
    if numeric >= 0.5:
        return "Medium"
    return "Low"


def _normalize_payload(
    payload: dict[str, Any],
    *,
    claim_text: str,
    language: str,
    valid_source_ids: set[str],
) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["extracted_claim"] = clean_text(str(normalized.get("extracted_claim") or claim_text))
    normalized["detected_language"] = clean_text(str(normalized.get("detected_language") or language or "Unknown"))
    normalized["category"] = clean_text(str(normalized.get("category") or "Other")) or "Other"

    verdict = clean_text(str(normalized.get("verdict") or "Not Enough Evidence"))
    normalized["verdict"] = verdict if verdict in ALLOWED_VERDICTS else "Not Enough Evidence"

    try:
        normalized["confidence"] = float(normalized.get("confidence", 0.2))
    except (TypeError, ValueError):
        normalized["confidence"] = 0.2

    confidence_label = clean_text(str(normalized.get("confidence_label") or ""))
    if confidence_label not in ALLOWED_CONFIDENCE_LABELS:
        confidence_label = _confidence_label_from_score(normalized["confidence"])
    normalized["confidence_label"] = confidence_label

    explanation = clean_text(str(normalized.get("explanation") or ""))
    if not explanation:
        explanation = "The verification pipeline could not produce a reliable structured result."
    normalized["explanation"] = explanation

    user_response = clean_text(str(normalized.get("user_response") or ""))
    if not user_response:
        user_response = explanation
    normalized["user_response"] = user_response

    raw_source_ids = normalized.get("used_source_ids")
    valid_ids: list[str] = []
    if isinstance(raw_source_ids, list):
        for source_id in raw_source_ids:
            source_str = str(source_id).strip()
            if source_str in valid_source_ids:
                valid_ids.append(source_str)
    normalized["used_source_ids"] = valid_ids
    return normalized


async def generate_verdict(
    claim_text: str,
    language: str,
    evidence: list[dict[str, Any]],
) -> tuple[LLMVerdictSchema, dict[str, Any]]:
    source_ids = [UUID(str(item["source_id"])) for item in evidence if item.get("source_id")]
    valid_source_ids = {str(source_id) for source_id in source_ids}
    model = get_model_for_task(NVIDIAModelTask.CLAIM_REASONING)

    if not evidence:
        return fallback_verdict(
            "JachAI could not find any supporting evidence to verify this claim.",
            [],
            claim_text=claim_text,
            language=language,
        ), {"model": model, "call_count": 0}
    if not settings.nvidia_api_key:
        return fallback_verdict(
            "NVIDIA API key is not configured.",
            source_ids,
            claim_text=claim_text,
            language=language,
        ), {"model": model, "call_count": 0}
    if not model:
        return fallback_verdict(
            "No NVIDIA reasoning model is configured.",
            source_ids,
            claim_text=claim_text,
            language=language,
        ), {"model": None, "call_count": 0}

    evidence_context = build_evidence_context(evidence)
    user_prompt = (
        f"User input:\n{claim_text}\n\n"
        f"Detected language hint:\n{language}\n\n"
        f"Evidence:\n{evidence_context}\n"
    )

    attempted_call = False
    try:
        attempted_call = True
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
        payload = load_json_with_repair(content or "{}")
        normalized = _normalize_payload(
            payload,
            claim_text=claim_text,
            language=language,
            valid_source_ids=valid_source_ids,
        )
        return LLMVerdictSchema.model_validate(normalized), {"model": model, "call_count": 1}
    except Exception:
        logger.exception("nvidia_reasoning_failed")

    return fallback_verdict(
        "The verification pipeline could not produce a reliable structured result.",
        source_ids,
        claim_text=claim_text,
        language=language,
    ), {"model": model, "call_count": 1 if attempted_call else 0}


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
        payload = load_json_with_repair(content or "{}")
        result = VisionOCRSchema.model_validate(payload)
        extracted_text = clean_text(result.extracted_text)
        if len(extracted_text) < settings.ocr_min_characters:
            return None
        return extracted_text
    except Exception:
        logger.exception("nvidia_vision_fallback_failed")
        return None
