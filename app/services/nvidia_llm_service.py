from __future__ import annotations

import asyncio
import base64
import json
from typing import Any
from uuid import UUID

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.verdict_schema import (
    ALLOWED_PIPELINE_VERDICTS,
    PIPELINE_TO_LEGACY_VERDICT,
    ClaimExtractionSchema,
    ClassifiedChunkSchema,
    LLMVerdictSchema,
    ReasoningVerdictSchema,
    SearchQueryGenerationSchema,
    SearchQuerySchema,
    VisionOCRSchema,
)
from app.services.ai_model_router import NVIDIAModelTask, get_model_for_task, is_task_enabled
from app.services.cache_service import cache_service
from app.services.evidence_context_builder import build_evidence_context
from app.services.nvidia_client import call_nvidia_chat
from app.services.text_cleaning_service import clean_text
from app.utils.json_repair import load_json_with_repair

logger = get_logger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

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
- Do not use your internal knowledge to assess the claim.
- If the input contains several related statements, return the most central verifiable claim in detected_claims with priority=1.
- If the text is noisy, still return the best recoverable factual claim.
- Return valid JSON only.
- Do not include markdown or extra text outside JSON.

Allowed detected_language values:
Bangla, English, Hindi, Banglish, Hinglish, Mixed, Unknown

Allowed category values:
Politics, Health, Disaster, Crime, Finance, Cybersecurity, Entertainment, Religion, Education, Other

Allowed verification_strategy values:
official_source_first, news_source_first, academic_source_first, general_web

Return JSON exactly in this schema:
{
  "extracted_claim": "string",
  "detected_language": "Bangla | English | Hindi | Banglish | Hinglish | Mixed | Unknown",
  "category": "Politics | Health | Disaster | Crime | Finance | Cybersecurity | Entertainment | Religion | Education | Other",
  "entities": ["string"],
  "time_context": "string or null",
  "location_context": "string or null",
  "requires_freshness": true,
  "verification_strategy": "official_source_first | news_source_first | academic_source_first | general_web",
  "detected_claims": [
    {"claim": "string", "priority": 1}
  ]
}
""".strip()

SEARCH_QUERY_SYSTEM_PROMPT = """
You are JachAI's search query generation model.

Your only job is to create 3-5 concise search queries that help retrieve fact-check and trusted-source evidence.

Important rules:
- Do not verify the claim.
- Do not give a verdict, confidence score, explanation, or source citation.
- Do not invent URLs.
- Prefer short, high-signal queries over long sentences.
- Include the original-language claim phrasing when useful.
- Include an English query when translation would help international fact-check or news search.
- Include relevant named entities, locations, dates, organizations, and event keywords.
- For scientific/medical claims: include academic or official-source queries.
- For political/legal claims: include government or reputable news queries.
- For company/product claims: include official company name queries.
- Do not include private personal identifiers.
- Return valid JSON only.
- Do not include markdown or extra text outside JSON.

Return JSON exactly in this schema:
{
  "queries": [
    {"query": "string", "purpose": "general | official | refutation | recent | background", "priority": 1}
  ]
}
""".strip()

STANCE_CLASSIFICATION_SYSTEM_PROMPT = """
You are JachAI's evidence stance classification model.

Your job is to classify how a single evidence passage relates to the given claim.

Important rules:
- Use ONLY the provided passage. Do not use outside knowledge.
- Do not fabricate or invent information.
- Compare the passage directly against the claim.
- If the passage directly confirms the key claim, mark "supports".
- If the passage directly contradicts the key claim, mark "refutes".
- If the passage only mentions related topics without proving or disproving the claim, mark "neutral" or "background".
- Keep rationale very short (1-2 sentences).
- quoted_evidence must be a direct quote or very close paraphrase from the passage.
- Return valid JSON only. No markdown or extra text outside JSON.

Return JSON exactly in this schema:
{
  "stance": "supports | refutes | neutral | background",
  "confidence": 0.0,
  "rationale": "string",
  "quoted_evidence": "string"
}
""".strip()

REASONING_SYSTEM_PROMPT = """
You are JachAI's evidence reasoning model.

Your only job is to decide a verdict for the provided claim using ONLY the supplied evidence.

Important rules:
- Use ONLY the provided evidence. Do not use your internal knowledge as proof.
- Do not invent sources or facts.
- If evidence directly confirms the claim from credible sources, return "supported".
- If credible evidence directly contradicts the claim, return "refuted".
- If the claim has partial truth but misses key context or adds false details, return "misleading" or "partially_true".
- If the claim was once true but the evidence shows it is no longer current, return "outdated".
- If the claim is vague, subjective, opinion-based, or cannot be verified, return "unverifiable".
- If evidence is missing, too weak, only about related topics, or only from low-credibility sources, return "insufficient_evidence".
- Never produce a confident verdict from snippet-only or low-credibility evidence.
- Prefer "insufficient_evidence" over guessing.
- Keep explanation short and concrete.
- Return valid JSON only. No markdown or extra text outside JSON.

Allowed verdicts: supported, refuted, misleading, partially_true, outdated, insufficient_evidence, unverifiable

Allowed confidence labels: Low, Medium, High

Return JSON exactly in this schema:
{
  "verdict": "supported | refuted | misleading | partially_true | outdated | insufficient_evidence | unverifiable",
  "confidence": 0.0,
  "confidence_label": "Low | Medium | High",
  "explanation": "string",
  "user_response": "string",
  "used_source_ids": ["string"],
  "key_evidence_ids": ["string"],
  "warnings": ["string"]
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
ALLOWED_STANCES = {"supports", "refutes", "neutral", "background"}
ALLOWED_QUERY_PURPOSES = {"general", "official", "refutation", "recent", "background"}


# ── Helper utilities ──────────────────────────────────────────────────────────

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


def _call_with_retry(call_fn, *, retries: int = 1):
    """Decorator-style helper — not used; retry logic is inline for async."""
    pass


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
    # New fields with safe defaults
    entities = normalized.get("entities")
    normalized["entities"] = [str(e) for e in entities] if isinstance(entities, list) else []
    normalized["time_context"] = str(normalized["time_context"]).strip() if normalized.get("time_context") else None
    normalized["location_context"] = str(normalized["location_context"]).strip() if normalized.get("location_context") else None
    normalized["requires_freshness"] = bool(normalized.get("requires_freshness", False))
    strategy = str(normalized.get("verification_strategy") or "general_web")
    valid_strategies = {"official_source_first", "news_source_first", "academic_source_first", "general_web"}
    normalized["verification_strategy"] = strategy if strategy in valid_strategies else "general_web"
    detected = normalized.get("detected_claims")
    normalized["detected_claims"] = detected if isinstance(detected, list) else []
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

    # Support new pipeline verdict labels
    raw_verdict = clean_text(str(normalized.get("verdict") or "insufficient_evidence"))
    if raw_verdict in ALLOWED_PIPELINE_VERDICTS:
        normalized["pipeline_verdict"] = raw_verdict
        normalized["verdict"] = PIPELINE_TO_LEGACY_VERDICT.get(raw_verdict, "Not Enough Evidence")
    elif raw_verdict in ALLOWED_VERDICTS:
        normalized["pipeline_verdict"] = None
        normalized["verdict"] = raw_verdict
    else:
        normalized["pipeline_verdict"] = "insufficient_evidence"
        normalized["verdict"] = "Not Enough Evidence"

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
    valid_ids: list[UUID] = []
    if isinstance(raw_source_ids, list):
        for source_id in raw_source_ids:
            source_str = str(source_id).strip()
            if source_str in valid_source_ids:
                valid_ids.append(UUID(source_str))
    normalized["used_source_ids"] = valid_ids

    key_evidence_ids = normalized.get("key_evidence_ids")
    normalized["key_evidence_ids"] = [str(k) for k in key_evidence_ids] if isinstance(key_evidence_ids, list) else []

    warnings = normalized.get("warnings")
    normalized["warnings"] = [str(w) for w in warnings] if isinstance(warnings, list) else []

    return normalized


async def _call_llm_with_retry(
    *,
    task: NVIDIAModelTask,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
) -> str:
    """Call NVIDIA LLM with one retry on failure."""
    retries = max(0, settings.llm_json_retry_count)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return await call_nvidia_chat(
                task=task,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
            )
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                logger.warning(
                    "llm_call_retry task=%s attempt=%d/%d error=%s",
                    task.value, attempt + 1, retries + 1, exc,
                )
                await asyncio.sleep(0.5)
    raise last_error or RuntimeError("LLM call failed")


# ── Fallback constructors ─────────────────────────────────────────────────────

def fallback_search_queries(
    *,
    extraction: ClaimExtractionSchema,
    original_text: str,
) -> SearchQueryGenerationSchema:
    flat = _dedupe_queries([extraction.extracted_claim, original_text])
    queries = [SearchQuerySchema(query=q, purpose="general", priority=i + 1) for i, q in enumerate(flat)]
    return SearchQueryGenerationSchema(queries=queries, search_queries=flat)


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
        pipeline_verdict="insufficient_evidence",
        warnings=[reason],
    )


# ── Public LLM service functions ──────────────────────────────────────────────

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
        content = await _call_llm_with_retry(
            task=NVIDIAModelTask.CLAIM_EXTRACTION,
            messages=[
                {"role": "system", "content": CLAIM_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=800,
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

    entities_str = ", ".join(extraction.entities) if extraction.entities else "None detected"
    user_prompt = (
        f"Extracted claim:\n{extraction.extracted_claim}\n\n"
        f"Detected language:\n{extraction.detected_language}\n\n"
        f"Claim category:\n{extraction.category}\n\n"
        f"Key entities:\n{entities_str}\n\n"
        f"Time context:\n{extraction.time_context or 'unknown'}\n\n"
        f"Location context:\n{extraction.location_context or 'unknown'}\n\n"
        f"Requires freshness:\n{extraction.requires_freshness}\n\n"
        f"Verification strategy:\n{extraction.verification_strategy}\n\n"
        f"Original user input:\n{original_text}\n"
    )

    attempted_call = False
    try:
        attempted_call = True
        content = await _call_llm_with_retry(
            task=NVIDIAModelTask.SEARCH_QUERY_GENERATION,
            messages=[
                {"role": "system", "content": SEARCH_QUERY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        payload = load_json_with_repair(content or "{}")
        raw_queries_list = payload.get("queries") if isinstance(payload, dict) else None
        parsed_queries: list[SearchQuerySchema] = []
        if isinstance(raw_queries_list, list):
            for item in raw_queries_list:
                if isinstance(item, dict):
                    q = clean_text(str(item.get("query") or ""))
                    purpose = str(item.get("purpose") or "general")
                    if purpose not in ALLOWED_QUERY_PURPOSES:
                        purpose = "general"
                    try:
                        priority = int(item.get("priority") or 1)
                    except (TypeError, ValueError):
                        priority = 1
                    if len(q) >= 5:
                        parsed_queries.append(SearchQuerySchema(query=q, purpose=purpose, priority=priority))
                elif isinstance(item, str):
                    q = clean_text(item)
                    if len(q) >= 5:
                        parsed_queries.append(SearchQuerySchema(query=q, purpose="general", priority=1))

        if not parsed_queries:
            # Try legacy flat list format
            raw_flat = payload.get("search_queries") if isinstance(payload, dict) else None
            if isinstance(raw_flat, list):
                for item in raw_flat:
                    q = clean_text(str(item))
                    if len(q) >= 5:
                        parsed_queries.append(SearchQuerySchema(query=q, purpose="general", priority=1))

        if not parsed_queries:
            return fallback, {"model": model, "call_count": 1, "cached": False}

        # Sort by priority, dedupe by query text
        parsed_queries.sort(key=lambda x: x.priority)
        seen: set[str] = set()
        deduped: list[SearchQuerySchema] = []
        for q in parsed_queries:
            normalized = q.query.lower()
            if normalized not in seen:
                seen.add(normalized)
                deduped.append(q)
        deduped = deduped[:6]

        flat_queries = [q.query for q in deduped]
        result = SearchQueryGenerationSchema(queries=deduped, search_queries=flat_queries)
        if cache_key:
            await cache_service.set_ai_task_payload(cache_task, cache_key, result.model_dump(mode="json"))
        return result, {"model": model, "call_count": 1, "cached": False}
    except Exception:
        logger.exception("nvidia_search_query_generation_failed")

    return fallback, {
        "model": model,
        "call_count": 1 if attempted_call else 0,
        "cached": False,
    }


async def classify_evidence_stance(
    chunk: dict[str, Any],
    *,
    normalized_claim: str,
) -> dict[str, Any]:
    """
    Classify a single evidence chunk's stance relative to the normalized claim.

    Returns a dict with: stance, stance_confidence, rationale, quoted_evidence.
    Falls back to neutral on any failure.
    """
    chunk_id = str(chunk.get("chunk_id") or "")
    chunk_text = str(chunk.get("text") or chunk.get("snippet") or "")
    source_id = str(chunk.get("source_id") or "")

    fallback_result = {
        **chunk,
        "stance": "neutral",
        "stance_confidence": 0.0,
        "rationale": "Classification unavailable.",
        "quoted_evidence": "",
    }

    if not chunk_text or not normalized_claim:
        return fallback_result

    if not settings.nvidia_api_key:
        return fallback_result

    model = get_model_for_task(NVIDIAModelTask.CLAIM_REASONING)
    if not model:
        return fallback_result

    user_prompt = (
        f"Claim to check:\n{normalized_claim}\n\n"
        f"Evidence passage:\n{chunk_text[:2000]}\n"
        f"\nSource: {chunk.get('url', '')}"
    )

    try:
        content = await _call_llm_with_retry(
            task=NVIDIAModelTask.CLAIM_REASONING,
            messages=[
                {"role": "system", "content": STANCE_CLASSIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        payload = load_json_with_repair(content or "{}")
        stance = str(payload.get("stance") or "neutral")
        if stance not in ALLOWED_STANCES:
            stance = "neutral"
        try:
            stance_confidence = float(payload.get("confidence") or 0.0)
        except (TypeError, ValueError):
            stance_confidence = 0.0
        rationale = clean_text(str(payload.get("rationale") or ""))
        quoted_evidence = clean_text(str(payload.get("quoted_evidence") or ""))
        return {
            **chunk,
            "stance": stance,
            "stance_confidence": stance_confidence,
            "rationale": rationale,
            "quoted_evidence": quoted_evidence,
        }
    except Exception:
        logger.exception(
            "nvidia_stance_classification_failed chunk_id=%s source_id=%s",
            chunk_id,
            source_id,
        )
        return fallback_result


async def classify_evidence_stances(
    chunks: list[dict[str, Any]],
    *,
    normalized_claim: str,
) -> list[dict[str, Any]]:
    """
    Classify all chunks concurrently. Falls back to neutral per chunk on failure.
    """
    if not chunks:
        return []
    results = await asyncio.gather(
        *[classify_evidence_stance(chunk, normalized_claim=normalized_claim) for chunk in chunks],
        return_exceptions=True,
    )
    classified: list[dict[str, Any]] = []
    for chunk, result in zip(chunks, results):
        if isinstance(result, Exception):
            logger.warning("stance_classification_exception chunk_id=%s error=%s", chunk.get("chunk_id"), result)
            classified.append({
                **chunk,
                "stance": "neutral",
                "stance_confidence": 0.0,
                "rationale": "Classification failed.",
                "quoted_evidence": "",
            })
        else:
            classified.append(result)  # type: ignore[arg-type]
    return classified


async def generate_verdict(
    extraction: ClaimExtractionSchema,
    evidence: list[dict[str, Any]],
    *,
    classified_chunks: list[dict[str, Any]] | None = None,
    credibility_scores: list[dict[str, Any]] | None = None,
    pipeline_warnings: list[str] | None = None,
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

    evidence_context = build_evidence_context(
        evidence,
        classified_chunks=classified_chunks,
        credibility_scores=credibility_scores,
    )
    warnings_text = ""
    if pipeline_warnings:
        warnings_text = "\nWarnings:\n" + "\n".join(f"- {w}" for w in pipeline_warnings) + "\n"

    user_prompt = (
        f"Claim to verify:\n{extraction.extracted_claim}\n\n"
        f"Claim type:\n{extraction.category}\n\n"
        f"Entities:\n{', '.join(extraction.entities) if extraction.entities else 'None'}\n\n"
        f"Time context:\n{extraction.time_context or 'Not specified'}\n\n"
        f"Location context:\n{extraction.location_context or 'Not specified'}\n\n"
        f"Requires freshness:\n{extraction.requires_freshness}\n\n"
        f"Detected language:\n{extraction.detected_language}\n\n"
        f"{warnings_text}"
        f"Evidence:\n{evidence_context}\n"
    )

    attempted_call = False
    try:
        attempted_call = True
        content = await _call_llm_with_retry(
            task=NVIDIAModelTask.CLAIM_REASONING,
            messages=[
                {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1400,
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
                "pipeline_verdict": normalized.get("pipeline_verdict"),
                "key_evidence_ids": normalized.get("key_evidence_ids", []),
                "warnings": normalized.get("warnings", []),
            }
        )
        llm_verdict = LLMVerdictSchema(
            extracted_claim=extraction.extracted_claim,
            detected_language=extraction.detected_language,
            category=extraction.category,
            verdict=reasoning.verdict,
            confidence=reasoning.confidence,
            confidence_label=reasoning.confidence_label,
            explanation=reasoning.explanation,
            user_response=reasoning.user_response,
            used_source_ids=reasoning.used_source_ids,
            pipeline_verdict=reasoning.pipeline_verdict,
            warnings=reasoning.warnings,
        )
        return llm_verdict, {"model": model, "call_count": 1}
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
