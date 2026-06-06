from __future__ import annotations

import base64
from typing import Any
from uuid import UUID

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.verdict_schema import (
    ClaimExtractionSchema,
    LLMVerdictSchema,
    ReasoningVerdictSchema,
    SearchQueryGenerationSchema,
    VisionOCRSchema,
)
from app.services.ai_model_router import NVIDIAModelTask, get_model_for_task, is_task_enabled
from app.services.cache_service import cache_service
from app.services.evidence_context_builder import build_evidence_context
from app.services.nvidia_client import call_nvidia_chat
from app.services.text_cleaning_service import clean_text
from app.utils.json_repair import load_json_with_repair

logger = get_logger(__name__)

CLAIM_EXTRACTION_SYSTEM_PROMPT = """
You are JachAI's claim extraction model.

Your only job is to normalize messy user input into a structured claim for downstream retrieval.

The user input may contain Bangla, English, Hindi, Banglish, Hinglish, OCR noise, slang, emojis, forwarded-message formatting, or repeated text.

Important rules:
- Extract the single clearest factual claim that can be checked.
- Remove greetings, hashtags, calls to action, and repeated noise.
- Keep the extracted claim faithful to the user's meaning.
- Do not verify, judge, or score the claim.
- Do not mention evidence, verdicts, or confidence.
- If the input contains several related statements, return the most central verifiable claim.
- If the text is noisy, still return the best recoverable factual claim.
- Return valid JSON only.
- Do not include markdown or extra text outside JSON.

Allowed detected_language values:
- Bangla
- English
- Hindi
- Banglish
- Hinglish
- Mixed
- Unknown

Allowed category values:
- Politics
- Health
- Disaster
- Crime
- Finance
- Cybersecurity
- Entertainment
- Religion
- Education
- Other

Return JSON exactly in this schema:
{
  "extracted_claim": "string",
  "detected_language": "Bangla | English | Hindi | Banglish | Hinglish | Mixed | Unknown",
  "category": "Politics | Health | Disaster | Crime | Finance | Cybersecurity | Entertainment | Religion | Education | Other"
}
""".strip()

SEARCH_QUERY_SYSTEM_PROMPT = """
You are JachAI's search query generation model.

Your only job is to create concise search queries that help retrieve fact-check and trusted-source evidence.

Important rules:
- Do not verify the claim.
- Do not give a verdict, confidence score, explanation, or source citation.
- Do not invent URLs.
- Prefer short, high-signal queries over long sentences.
- Include the original-language claim phrasing when useful.
- Include an English query when translation would help international fact-check or news search.
- Include relevant named entities, locations, dates, organizations, and event keywords.
- Do not include private personal identifiers.
- Return valid JSON only.
- Do not include markdown or extra text outside JSON.

Return JSON exactly in this schema:
{
  "search_queries": ["string"]
}
""".strip()

REASONING_SYSTEM_PROMPT = """
You are JachAI's evidence reasoning model.

Your only job is to decide a verdict for the provided claim using only the supplied evidence.

Important rules:
- Use ONLY the provided evidence.
- Do not extract or rewrite the claim unless needed for understanding.
- Do not use your internal knowledge as proof.
- Do not invent sources or facts.
- If evidence is missing, weak, loosely related, or conflicting, return "Not Enough Evidence".
- Do not overstate certainty.
- Keep the explanation short and concrete.
- Return valid JSON only.
- Do not include extra text outside JSON.
- Do not include markdown.

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


def _dedupe_queries(values: list[str]) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()
    for value in values:
        query = clean_text(value)
        if len(query) < 5:
            continue
        normalized = query.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        queries.append(query)
    return queries[:6]


def fallback_search_queries(
    *,
    extraction: ClaimExtractionSchema,
    original_text: str,
) -> SearchQueryGenerationSchema:
    return SearchQueryGenerationSchema(
        search_queries=_dedupe_queries([extraction.extracted_claim, original_text]),
    )


def fallback_claim_extraction(
    claim_text: str,
    *,
    language_hint: str,
) -> ClaimExtractionSchema:
    extracted_claim = clean_text(claim_text) or "Claim text unavailable."
    detected_language = clean_text(language_hint) or "Unknown"
    return ClaimExtractionSchema(
        extracted_claim=extracted_claim,
        detected_language=detected_language,
        category="Other",
    )


def _default_user_response(reason: str) -> str:
    return f"JachAI could not confidently verify this claim yet. {reason}"


def fallback_verdict(
    reason: str,
    source_ids: list[UUID],
    *,
    extraction: ClaimExtractionSchema,
) -> LLMVerdictSchema:
    return LLMVerdictSchema(
        extracted_claim=extraction.extracted_claim,
        detected_language=extraction.detected_language,
        category=extraction.category,
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


def _normalize_extraction_payload(
    payload: dict[str, Any],
    *,
    claim_text: str,
    language_hint: str,
) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["extracted_claim"] = clean_text(str(normalized.get("extracted_claim") or claim_text))
    normalized["detected_language"] = clean_text(str(normalized.get("detected_language") or language_hint or "Unknown"))
    normalized["category"] = clean_text(str(normalized.get("category") or "Other")) or "Other"
    return normalized


def _normalize_payload(
    payload: dict[str, Any],
    *,
    extraction: ClaimExtractionSchema,
    valid_source_ids: set[str],
) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["extracted_claim"] = extraction.extracted_claim
    normalized["detected_language"] = extraction.detected_language
    normalized["category"] = extraction.category

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


async def extract_claim_context(
    claim_text: str,
    language_hint: str,
    *,
    cache_key: str | None = None,
) -> tuple[ClaimExtractionSchema, dict[str, Any]]:
    model = get_model_for_task(NVIDIAModelTask.CLAIM_EXTRACTION)
    if not settings.nvidia_api_key:
        return fallback_claim_extraction(claim_text, language_hint=language_hint), {"model": model, "call_count": 0}
    if not model:
        return fallback_claim_extraction(claim_text, language_hint=language_hint), {"model": None, "call_count": 0}

    cache_task = f"{NVIDIAModelTask.CLAIM_EXTRACTION.value}:{model}"
    if cache_key:
        cached_payload = await cache_service.get_ai_task_payload(cache_task, cache_key)
        if cached_payload:
            try:
                return ClaimExtractionSchema.model_validate(cached_payload), {
                    "model": model,
                    "call_count": 0,
                    "cached": True,
                }
            except Exception:
                logger.warning("cached_claim_extraction_invalid cache_key=%s", cache_key)

    user_prompt = (
        f"Raw user input:\n{claim_text}\n\n"
        f"Language hint:\n{language_hint}\n"
    )

    attempted_call = False
    try:
        attempted_call = True
        content = await call_nvidia_chat(
            task=NVIDIAModelTask.CLAIM_EXTRACTION,
            messages=[
                {"role": "system", "content": CLAIM_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=500,
        )
        payload = load_json_with_repair(content or "{}")
        normalized = _normalize_extraction_payload(
            payload,
            claim_text=claim_text,
            language_hint=language_hint,
        )
        extraction = ClaimExtractionSchema.model_validate(normalized)
        if cache_key:
            await cache_service.set_ai_task_payload(cache_task, cache_key, extraction.model_dump(mode="json"))
        return extraction, {"model": model, "call_count": 1, "cached": False}
    except Exception:
        logger.exception("nvidia_claim_extraction_failed")

    return fallback_claim_extraction(claim_text, language_hint=language_hint), {
        "model": model,
        "call_count": 1 if attempted_call else 0,
        "cached": False,
    }


async def generate_search_queries(
    *,
    extraction: ClaimExtractionSchema,
    original_text: str,
    cache_key: str | None = None,
) -> tuple[SearchQueryGenerationSchema, dict[str, Any]]:
    model = get_model_for_task(NVIDIAModelTask.SEARCH_QUERY_GENERATION)
    fallback = fallback_search_queries(extraction=extraction, original_text=original_text)
    if not settings.nvidia_api_key:
        return fallback, {"model": model, "call_count": 0}
    if not model:
        return fallback, {"model": None, "call_count": 0}

    cache_task = f"{NVIDIAModelTask.SEARCH_QUERY_GENERATION.value}:{model}"
    if cache_key:
        cached_payload = await cache_service.get_ai_task_payload(cache_task, cache_key)
        if cached_payload:
            try:
                return SearchQueryGenerationSchema.model_validate(cached_payload), {
                    "model": model,
                    "call_count": 0,
                    "cached": True,
                }
            except Exception:
                logger.warning("cached_search_queries_invalid cache_key=%s", cache_key)

    user_prompt = (
        f"Extracted claim:\n{extraction.extracted_claim}\n\n"
        f"Detected language:\n{extraction.detected_language}\n\n"
        f"Claim category:\n{extraction.category}\n\n"
        f"Original user input:\n{original_text}\n"
    )

    attempted_call = False
    try:
        attempted_call = True
        content = await call_nvidia_chat(
            task=NVIDIAModelTask.SEARCH_QUERY_GENERATION,
            messages=[
                {"role": "system", "content": SEARCH_QUERY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
            max_tokens=500,
        )
        payload = load_json_with_repair(content or "{}")
        raw_queries = payload.get("search_queries") if isinstance(payload, dict) else None
        queries = raw_queries if isinstance(raw_queries, list) else []
        normalized = SearchQueryGenerationSchema(search_queries=_dedupe_queries([str(item) for item in queries]))
        if cache_key:
            await cache_service.set_ai_task_payload(cache_task, cache_key, normalized.model_dump(mode="json"))
        return normalized, {"model": model, "call_count": 1, "cached": False}
    except Exception:
        logger.exception("nvidia_search_query_generation_failed")

    return fallback, {
        "model": model,
        "call_count": 1 if attempted_call else 0,
        "cached": False,
    }


async def generate_verdict(
    extraction: ClaimExtractionSchema,
    evidence: list[dict[str, Any]],
) -> tuple[LLMVerdictSchema, dict[str, Any]]:
    source_ids = [UUID(str(item["source_id"])) for item in evidence if item.get("source_id")]
    valid_source_ids = {str(source_id) for source_id in source_ids}
    model = get_model_for_task(NVIDIAModelTask.CLAIM_REASONING)

    if not evidence:
        return fallback_verdict(
            "JachAI could not find any supporting evidence to verify this claim.",
            [],
            extraction=extraction,
        ), {"model": model, "call_count": 0}
    if not settings.nvidia_api_key:
        return fallback_verdict(
            "NVIDIA API key is not configured.",
            source_ids,
            extraction=extraction,
        ), {"model": model, "call_count": 0}
    if not model:
        return fallback_verdict(
            "No NVIDIA reasoning model is configured.",
            source_ids,
            extraction=extraction,
        ), {"model": None, "call_count": 0}

    evidence_context = build_evidence_context(evidence)
    user_prompt = (
        f"Extracted claim:\n{extraction.extracted_claim}\n\n"
        f"Detected language:\n{extraction.detected_language}\n\n"
        f"Claim category:\n{extraction.category}\n\n"
        f"Evidence:\n{evidence_context}\n"
    )

    attempted_call = False
    try:
        attempted_call = True
        content = await call_nvidia_chat(
            task=NVIDIAModelTask.CLAIM_REASONING,
            messages=[
                {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
            max_tokens=1200,
        )
        payload = load_json_with_repair(content or "{}")
        normalized = _normalize_payload(
            payload,
            extraction=extraction,
            valid_source_ids=valid_source_ids,
        )
        reasoning = ReasoningVerdictSchema.model_validate(
            {
                "verdict": normalized["verdict"],
                "confidence": normalized["confidence"],
                "confidence_label": normalized["confidence_label"],
                "explanation": normalized["explanation"],
                "user_response": normalized["user_response"],
                "used_source_ids": normalized["used_source_ids"],
            }
        )
        return LLMVerdictSchema(
            extracted_claim=extraction.extracted_claim,
            detected_language=extraction.detected_language,
            category=extraction.category,
            verdict=reasoning.verdict,
            confidence=reasoning.confidence,
            confidence_label=reasoning.confidence_label,
            explanation=reasoning.explanation,
            user_response=reasoning.user_response,
            used_source_ids=reasoning.used_source_ids,
        ), {"model": model, "call_count": 1}
    except Exception:
        logger.exception("nvidia_reasoning_failed")

    return fallback_verdict(
        "The verification pipeline could not produce a reliable structured result.",
        source_ids,
        extraction=extraction,
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
