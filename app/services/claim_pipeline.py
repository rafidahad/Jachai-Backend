from __future__ import annotations

import math
import re
import time
from typing import Any
from uuid import UUID

import httpx
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.logging import get_logger
from app.db.session import SessionLocal
from app.models.audit_log import AuditLog
from app.models.claim import Claim, ClaimEvidenceLink
from app.models.verification_job import VerificationJob
from app.schemas.ai_schema import AIUsageSchema
from app.schemas.claim_schema import (
    ClaimResponseSchema,
    ClaimSubmissionResponseSchema,
    VerificationJobSchema,
    VerificationLiveStatusSchema,
)
from app.schemas.rerank_schema import EvidenceCandidateSchema
from app.schemas.verdict_schema import EvidenceSnippetSchema, LLMVerdictSchema
from app.services.ai_model_router import NVIDIAModelTask, get_model_for_task
from app.services.cache_service import cache_service
from app.services.cluster_service import get_or_create_cluster
from app.services.embedding_service import embed_text
from app.services.evidence_chunker import chunk_evidence_documents
from app.services.evidence_index_service import get_evidence_index_status
from app.services.language_service import detect_language
from app.services.live_evidence_service import hydrate_live_evidence
from app.services.nvidia_llm_service import (
    classify_evidence_stances,
    extract_claim_context,
    generate_search_queries,
    generate_verdict,
)
from app.services.nvidia_rerank_service import rerank_evidence
from app.services.ocr_service import extract_text_from_image, prepare_ocr_text_for_claim_extraction
from app.services.pii_service import mask_pii
from app.services.retrieval_service import retrieve_evidence
from app.services.search_service import score_source_credibility
from app.services.text_cleaning_service import clean_text, text_from_html
from app.utils.errors import AppError
from app.utils.hashing import normalized_hash
from app.utils.time import utc_now

logger = get_logger(__name__)

TOKEN_PATTERN = re.compile(r"[\w']+", re.UNICODE)
MATCH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by",
    "for", "from", "has", "have", "in", "into", "is", "it",
    "its", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "with",
}
MATCH_TOKEN_ALIASES = {
    "buffaloes": "buffalo",
    "cattle": "buffalo",
    "cow": "buffalo",
    "cows": "buffalo",
    "mahish": "buffalo",
    "mohis": "buffalo",
    "mohish": "buffalo",
    "mohishh": "buffalo",
    "mosh": "buffalo",
    "মহিষ": "buffalo",
    "eidaladha": "sacrifice",
    "eiduladha": "sacrifice",
    "korban": "sacrifice",
    "korbani": "sacrifice",
    "kurban": "sacrifice",
    "qorbani": "sacrifice",
    "qurbani": "sacrifice",
    "sacrificed": "sacrifice",
    "sacrificing": "sacrifice",
    "কোরবানি": "sacrifice",
    "কুরবানি": "sacrifice",
}

LIVE_STAGE_SEQUENCE = (
    "queued",
    "extracting_claim",
    "searching_sources",
    "comparing_evidence",
    "generating_verdict",
)
LIVE_STAGE_METADATA = {
    "queued": {
        "label": "Reading input",
        "message": "Preparing the incoming submission...",
        "index": 0,
        "progress": 10.0,
    },
    "extracting_claim": {
        "label": "Extracting claim",
        "message": "Normalizing the claim and isolating the factual core...",
        "index": 1,
        "progress": 28.0,
    },
    "searching_sources": {
        "label": "Searching trusted sources",
        "message": "Generating retrieval queries and collecting live evidence...",
        "index": 2,
        "progress": 56.0,
    },
    "comparing_evidence": {
        "label": "Comparing evidence",
        "message": "Ranking sources and measuring support versus contradiction...",
        "index": 3,
        "progress": 78.0,
    },
    "generating_verdict": {
        "label": "Generating verdict",
        "message": "Synthesizing grounded evidence into the final decision...",
        "index": 4,
        "progress": 92.0,
    },
    "completed": {
        "label": "Verification complete",
        "message": "Evidence locked. Preparing your verdict...",
        "index": 4,
        "progress": 100.0,
    },
    "failed": {
        "label": "Verification failed",
        "message": "The verification engine hit an error while processing this claim.",
        "index": 4,
        "progress": 100.0,
    },
}
MAX_LIVE_EVENTS = 18


# ── Utility helpers ───────────────────────────────────────────────────────────

def _parse_uuid(value: UUID | str, *, code: str, message: str) -> UUID:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except ValueError as exc:
        raise AppError(status_code=422, code=code, message=message) from exc


def _confidence_label_from_score(score: float) -> str:
    if score >= 0.8:
        return "High"
    if score >= 0.5:
        return "Medium"
    return "Low"


def _canonical_match_token(token: str) -> str:
    normalized = token.lower().strip("'")
    if normalized.endswith("'s"):
        normalized = normalized[:-2]
    normalized = normalized.replace("-", "")
    return MATCH_TOKEN_ALIASES.get(normalized, normalized)


def _tokens_for_match(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in TOKEN_PATTERN.findall(text):
        canonical = _canonical_match_token(token)
        if len(canonical) < 3 or canonical in MATCH_STOPWORDS:
            continue
        tokens.add(canonical)
    return tokens


def _token_overlap_score(claim_text: str, evidence_text: str) -> float:
    claim_tokens = _tokens_for_match(claim_text)
    if not claim_tokens:
        return 0.0
    evidence_tokens = _tokens_for_match(evidence_text)
    if not evidence_tokens:
        return 0.0
    overlap = claim_tokens & evidence_tokens
    return min(1.0, len(overlap) / len(claim_tokens))


def _normalized_search_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score <= 0:
        return 0.0
    if score <= 1:
        return min(1.0, math.sqrt(score))
    return min(1.0, score / 100.0)


def _calculate_match_score(
    *,
    claim_text: str,
    title: str,
    snippet: str,
    text_content: str,
    similarity_score: float,
    search_score: Any,
) -> tuple[float, float]:
    compact_content = text_content[:4000]
    body_overlap = _token_overlap_score(claim_text, f"{title} {snippet} {compact_content}")
    title_overlap = _token_overlap_score(claim_text, title)
    lexical_score = max(body_overlap, title_overlap * 0.9)
    retrieval_score = max(0.0, min(1.0, float(similarity_score)))
    normalized_search = _normalized_search_score(search_score)
    match_score = max(retrieval_score, lexical_score, normalized_search * 0.7)
    return round(min(1.0, match_score), 4), round(lexical_score, 4)


def _evidence_relevance_score(item: dict[str, Any]) -> float:
    scores: list[float] = []
    for key in ("match_score", "similarity_score", "lexical_overlap_score", "relevance_score"):
        try:
            scores.append(float(item.get(key) or 0.0))
        except (TypeError, ValueError):
            continue
    return max(scores) if scores else 0.0


def _is_relevant_evidence(item: dict[str, Any]) -> bool:
    return _evidence_relevance_score(item) >= settings.min_relevant_similarity


def _looks_like_unsupported_detail(explanation: str, user_response: str) -> bool:
    text = f"{explanation} {user_response}".lower()
    markers = (
        "does not mention", "doesn't mention", "do not mention",
        "no mention", "not mention", "does not support",
        "do not support", "unsupported", "but not",
    )
    return any(marker in text for marker in markers)


def _is_poor_verdict_summary(summary: str, *, claim_text: str) -> bool:
    normalized = clean_text(summary)
    if not normalized:
        return True
    lowered = normalized.lower()
    claim_lowered = clean_text(claim_text).lower()
    if lowered.startswith(("claim to verify:", "claim:", "input claim:", "extracted claim:")):
        return True
    if lowered in {claim_lowered, f"claim to verify: {claim_lowered}"}:
        return True
    if len(normalized) < 45 and claim_lowered and claim_lowered in lowered:
        return True
    return False


def _build_claim_aware_summary(
    *,
    verdict_label: str,
    claim_text: str,
    explanation: str,
    evidence: list[dict[str, Any]],
) -> str:
    claim = clean_text(claim_text)
    source_count = len(evidence)
    source_phrase = f"{source_count} source{'s' if source_count != 1 else ''}" if source_count else "the available sources"
    compact_explanation = clean_text(explanation)

    if verdict_label == "Likely True":
        lead = f"JachAI found reliable evidence supporting the claim: \"{claim}\"."
    elif verdict_label == "Likely False":
        lead = f"JachAI found reliable evidence contradicting the claim: \"{claim}\"."
    elif verdict_label == "Misleading":
        lead = f"JachAI found that the claim \"{claim}\" is misleading or missing important context."
    else:
        lead = f"JachAI could not verify the claim \"{claim}\" from reliable matching evidence."

    if compact_explanation and not _is_poor_verdict_summary(compact_explanation, claim_text=claim):
        return f"{lead} {compact_explanation}"

    if verdict_label == "Not Enough Evidence":
        return (
            f"{lead} {source_phrase.capitalize()} did not directly confirm the claim, "
            "so the result remains Not Enough Evidence."
        )
    return f"{lead} The verdict is based on the strongest matching evidence JachAI retrieved."


def _source_ids_from_evidence(evidence: list[dict[str, Any]], *, limit: int = 3) -> list[UUID]:
    source_ids: list[UUID] = []
    for item in evidence:
        try:
            source_ids.append(UUID(str(item["source_id"])))
        except (KeyError, TypeError, ValueError):
            continue
        if len(source_ids) >= limit:
            break
    return source_ids


def _filter_selected_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    relevant = [item for item in evidence if _is_relevant_evidence(item)]
    if not relevant:
        return evidence

    filtered: list[dict[str, Any]] = []
    for final_rank, item in enumerate(relevant[: settings.final_evidence_top_k], start=1):
        ranked_item = dict(item)
        ranked_item["final_rank"] = final_rank
        filtered.append(ranked_item)
    return filtered


def _preview_text(value: str | None, *, limit: int = 900) -> str | None:
    if not value:
        return None
    compact = re.sub(r"[ \t]+", " ", str(value)).strip()
    compact = re.sub(r"\n{3,}", "\n\n", compact)
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def _dedupe_text_variants(values: list[str | None]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = clean_text(value or "")
        if len(cleaned) < 5:
            continue
        normalized = cleaned.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(cleaned)
    return ordered


def _merge_retrieved_result_sets(
    result_sets: list[list[tuple[Any, float]]],
    *,
    limit: int,
) -> list[tuple[Any, float]]:
    merged: dict[str, tuple[Any, float]] = {}
    for result_set in result_sets:
        for source, score in result_set:
            source_id = str(getattr(source, "id", ""))
            if not source_id:
                continue
            existing = merged.get(source_id)
            if existing is None or float(score) > float(existing[1]):
                merged[source_id] = (source, float(score))
    ordered = sorted(merged.values(), key=lambda item: item[1], reverse=True)
    return ordered[:limit]


def _live_stage_payload(stage_key: str, *, message: str | None = None) -> dict[str, Any]:
    stage = LIVE_STAGE_METADATA.get(stage_key, LIVE_STAGE_METADATA["queued"])
    return {
        "stage_key": stage_key,
        "stage_label": stage["label"],
        "stage_message": message or stage["message"],
        "stage_index": stage["index"],
        "total_stages": len(LIVE_STAGE_SEQUENCE),
        "progress_percent": stage["progress"],
    }


def _touch_live_status(live_status: dict[str, Any]) -> dict[str, Any]:
    live_status["updated_at"] = utc_now().isoformat()
    return live_status


def _set_live_stage(
    live_status: dict[str, Any],
    stage_key: str,
    *,
    message: str | None = None,
) -> dict[str, Any]:
    live_status.update(_live_stage_payload(stage_key, message=message))
    return _touch_live_status(live_status)


def _append_live_event(
    live_status: dict[str, Any],
    *,
    key: str,
    label: str,
    status: str,
    summary: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    events = list(live_status.get("events") or [])
    events.append(
        {
            "key": key,
            "label": label,
            "status": status,
            "summary": summary,
            "timestamp": utc_now().isoformat(),
            "metadata": metadata or {},
        }
    )
    live_status["events"] = events[-MAX_LIVE_EVENTS:]
    return _touch_live_status(live_status)


def _merge_live_details(
    live_status: dict[str, Any],
    section: str,
    values: dict[str, Any],
) -> dict[str, Any]:
    details = dict(live_status.get("details") or {})
    current = details.get(section)
    merged = dict(current) if isinstance(current, dict) else {}
    merged.update(values)
    details[section] = merged
    live_status["details"] = details
    return _touch_live_status(live_status)


def _set_live_warning_messages(
    live_status: dict[str, Any],
    warnings: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    current = [str(item) for item in live_status.get("warnings") or [] if str(item).strip()]
    for warning in warnings:
        normalized = str(warning).strip()
        if normalized and normalized not in current:
            current.append(normalized)
    live_status["warnings"] = current
    return _touch_live_status(live_status)


def _truncate_metadata(value: Any, *, limit: int = 220) -> Any:
    if isinstance(value, str):
        return _preview_text(value, limit=limit) or ""
    if isinstance(value, list):
        return [_truncate_metadata(item, limit=limit) for item in value[:6]]
    if isinstance(value, dict):
        return {str(key): _truncate_metadata(item, limit=limit) for key, item in list(value.items())[:10]}
    return value


def _build_initial_live_status(
    *,
    input_type: str,
    raw_input: str | None = None,
    source_url: str | None = None,
    context_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = dict(context_payload or {})
    ocr_model = metadata.get("ocr_model") or get_model_for_task(NVIDIAModelTask.IMAGE_OCR)
    live_status: dict[str, Any] = {
        **_live_stage_payload("queued"),
        "raw_input": _preview_text(raw_input),
        "cleaned_input": None,
        "masked_input": None,
        "extracted_claim": None,
        "retrieval_query": None,
        "detected_language": None,
        "search_queries": [],
        "warnings": [],
        "details": {
            "input": {
                "input_type": input_type,
                "source_url": source_url,
                "filename": metadata.get("filename"),
                "content_type": metadata.get("content_type"),
                "ocr_method": metadata.get("ocr_method"),
                "ocr_model": ocr_model if input_type == "image" else None,
                "external_id": metadata.get("external_id"),
            },
            "models": {
                "claim_extraction_model": settings.active_gemini_claim_extraction_model
                or get_model_for_task(NVIDIAModelTask.CLAIM_EXTRACTION),
                "reasoning_candidates": settings.active_gemini_reasoning_models
                or ([get_model_for_task(NVIDIAModelTask.CLAIM_REASONING)] if get_model_for_task(NVIDIAModelTask.CLAIM_REASONING) else []),
                "search_provider": settings.search_provider,
                "embedding_model": settings.embedding_model,
                "backup_gemini_key_count": max(0, len(settings.active_gemini_api_keys) - 1),
            },
        },
        "events": [],
        "updated_at": utc_now().isoformat(),
    }
    if source_url and not live_status["raw_input"]:
        live_status["raw_input"] = source_url
    if metadata.get("ocr_warning"):
        _set_live_warning_messages(live_status, [str(metadata["ocr_warning"])])
    return _append_live_event(
        live_status,
        key="job_queued",
        label="Job queued",
        status="running",
        summary="Verification job created and waiting for engine execution.",
        metadata={"input_type": input_type},
    )


def _normalize_supporting_text(value: str | None) -> str | None:
    cleaned = clean_text(value or "")
    return cleaned or None


def _compose_claim_source_input(
    *,
    primary_text: str,
    supporting_text: str | None = None,
    source_label: str,
) -> str:
    cleaned_primary = clean_text(primary_text)
    cleaned_supporting = _normalize_supporting_text(supporting_text)

    if cleaned_supporting and cleaned_primary:
        return f"User note:\n{cleaned_supporting}\n\n{source_label}:\n{cleaned_primary}"
    if cleaned_primary:
        return cleaned_primary
    return cleaned_supporting or ""


def _evidence_preview(items: list[dict[str, Any]], *, limit: int = 4) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []
    for item in items[:limit]:
        previews.append(
            {
                "source_id": item.get("source_id"),
                "title": _preview_text(str(item.get("title") or ""), limit=120),
                "publisher": _preview_text(str(item.get("publisher") or ""), limit=60),
                "url": item.get("url"),
                "snippet": _preview_text(str(item.get("snippet") or ""), limit=180),
                "match_score": item.get("match_score"),
                "similarity_score": item.get("similarity_score"),
                "rerank_score": item.get("rerank_score"),
                "search_score": item.get("search_score"),
                "trust_score": item.get("trust_score"),
                "stance": item.get("stance"),
            }
        )
    return previews


def _build_reasoning_text(
    verdict: LLMVerdictSchema,
    evidence: list[dict[str, Any]],
    *,
    rerank_applied: bool,
    classified_chunks: list[dict[str, Any]] | None = None,
) -> str:
    relevant_count = sum(1 for item in evidence if _is_relevant_evidence(item))
    chunk_info = f", {len(classified_chunks)} classified evidence chunks" if classified_chunks else ""
    pipeline_verdict = getattr(verdict, "pipeline_verdict", None)
    verdict_info = f" (pipeline verdict: {pipeline_verdict})" if pipeline_verdict else ""
    return (
        f"Evaluated {len(evidence)} evidence source(s){chunk_info}, "
        f"used {len(verdict.used_source_ids)} source(s) in the final answer, "
        f"{relevant_count} met the relevance threshold of {settings.min_relevant_similarity:.2f}, "
        f"and reranking was {'applied' if rerank_applied else 'not applied'}{verdict_info}."
    )


def _build_evidence_candidates(
    retrieved: list[tuple[Any, float]],
    *,
    claim_text: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for initial_rank, (source, similarity_score) in enumerate(retrieved, start=1):
        metadata = source.source_meta or {}
        match_score, lexical_overlap_score = _calculate_match_score(
            claim_text=claim_text,
            title=source.title,
            snippet=source.snippet,
            text_content=source.text_content,
            similarity_score=float(similarity_score),
            search_score=metadata.get("search_score"),
        )
        candidate = EvidenceCandidateSchema(
            source_id=source.id,
            title=source.title,
            url=source.url,
            publisher=source.publisher,
            language=source.language,
            source_type=source.source_type,
            snippet=source.snippet,
            similarity_score=float(similarity_score),
            match_score=match_score,
            initial_rank=initial_rank,
        )
        candidate_payload = candidate.model_dump(mode="json")
        candidate_payload.update(
            {
                "lexical_overlap_score": lexical_overlap_score,
                "search_score": metadata.get("search_score"),
                "trust_score": metadata.get("trust_score"),
                "provider": metadata.get("provider"),
            }
        )
        candidates.append(candidate_payload)
    return candidates


def _build_ai_usage_payload(
    *,
    claim_extraction_model: str | None,
    query_generation_model: str | None,
    reasoning_model: str | None,
    rerank_model: str | None,
    vision_model: str | None,
    search_provider: str | None,
    llm_call_count: int,
    rerank_call_count: int,
    vision_call_count: int,
    tavily_request_count: int,
) -> dict[str, Any]:
    return AIUsageSchema(
        embedding_model=settings.embedding_model,
        claim_extraction_model=claim_extraction_model,
        query_generation_model=query_generation_model,
        rerank_model=rerank_model,
        reasoning_model=reasoning_model,
        vision_model=vision_model,
        search_provider=search_provider,
        llm_call_count=llm_call_count,
        rerank_call_count=rerank_call_count,
        vision_call_count=vision_call_count,
        tavily_request_count=tavily_request_count,
    ).model_dump(mode="json")


def _apply_verdict_guardrails(
    verdict: LLMVerdictSchema,
    *,
    claim_text: str,
    language: str,
    evidence: list[dict[str, Any]],
    classified_chunks: list[dict[str, Any]] | None = None,
) -> LLMVerdictSchema:
    """
    Apply post-hoc safety guardrails to the LLM verdict.

    Ensures:
    - Source IDs cited are actually in the evidence set
    - Insufficient evidence cases are correctly downgraded
    - Confidence is bounded by evidence quality
    """
    valid_source_ids = {str(item["source_id"]) for item in evidence if item.get("source_id")}
    filtered_source_ids = [
        source_id for source_id in verdict.used_source_ids if str(source_id) in valid_source_ids
    ]
    relevant_evidence = [item for item in evidence if _is_relevant_evidence(item)]
    trusted_evidence = [
        item for item in relevant_evidence if float(item.get("trust_score") or 0.50) > 0.50
    ]

    extracted_claim = verdict.extracted_claim or claim_text
    detected_language = verdict.detected_language or language or "Unknown"
    category = verdict.category or "Other"
    explanation = verdict.explanation or (
        "JachAI could not find enough reliable evidence from trusted sources to verify this claim."
    )
    user_response = verdict.user_response or explanation
    verdict_label = verdict.verdict
    pipeline_verdict = getattr(verdict, "pipeline_verdict", None)
    confidence = min(max(float(verdict.confidence), 0.0), 1.0)
    confidence_label = verdict.confidence_label
    warnings = list(getattr(verdict, "warnings", []))

    # ── Stance-based insufficient_evidence check ──────────────────────────────
    if classified_chunks:
        supports_count = sum(1 for c in classified_chunks if c.get("stance") == "supports")
        refutes_count = sum(1 for c in classified_chunks if c.get("stance") == "refutes")
        total_chunks = len(classified_chunks)
        strong_chunks = supports_count + refutes_count
        # If less than 20% of chunks are directly relevant (support/refute), evidence is weak
        if total_chunks > 0 and strong_chunks / total_chunks < 0.2:
            if verdict_label not in {"Not Enough Evidence"}:
                verdict_label = "Not Enough Evidence"
                pipeline_verdict = "insufficient_evidence"
                confidence = min(confidence, 0.40)
                confidence_label = "Low"
                explanation = (
                    "Most retrieved evidence does not directly support or refute the claim."
                )
                user_response = (
                    "JachAI Verdict: Insufficient Evidence. "
                    "The available sources do not directly address this claim."
                )
                warnings.append(
                    f"Only {strong_chunks}/{total_chunks} evidence chunks directly relevant."
                )

    # ── No evidence at all ────────────────────────────────────────────────────
    if not evidence:
        verdict_label = "Not Enough Evidence"
        pipeline_verdict = "insufficient_evidence"
        confidence = min(confidence, 0.35)
        confidence_label = "Low"
        explanation = "JachAI could not find enough reliable evidence from trusted sources to verify this claim."
        user_response = (
            "JachAI Verdict: Insufficient Evidence. "
            "We could not find enough reliable sources to verify this claim yet."
        )
        filtered_source_ids = []
    elif not relevant_evidence:
        verdict_label = "Not Enough Evidence"
        pipeline_verdict = "insufficient_evidence"
        confidence = min(confidence, 0.45)
        confidence_label = "Low"
        explanation = "The retrieved evidence is too weak or only loosely related to this claim."
        user_response = (
            "JachAI Verdict: Insufficient Evidence. "
            "The available sources are not strong enough to verify this claim."
        )
        filtered_source_ids = []

    # ── Misleading detection via text signals ─────────────────────────────────
    if (
        verdict_label == "Not Enough Evidence"
        and trusted_evidence
        and _looks_like_unsupported_detail(explanation, user_response)
    ):
        verdict_label = "Misleading"
        pipeline_verdict = "misleading"
        confidence = max(min(confidence, 0.65), 0.55)
        confidence_label = "Medium"
        explanation = (
            "Trusted sources confirm a related real story, but they do not support the claim's decisive added detail."
        )
        user_response = (
            "JachAI Verdict: Misleading. Trusted sources match the same broad story, but the evidence does not "
            "support the added detail in the claim."
        )
        if not filtered_source_ids:
            filtered_source_ids = _source_ids_from_evidence(trusted_evidence)

    # ── Require at least one cited source for strong verdicts ─────────────────
    if verdict_label in {"Likely True", "Likely False"} and not filtered_source_ids:
        verdict_label = "Not Enough Evidence"
        pipeline_verdict = "insufficient_evidence"
        confidence = min(confidence, 0.35)
        confidence_label = "Low"
        explanation = "A strong true or false verdict needs at least one valid supporting source."
        user_response = (
            "JachAI Verdict: Insufficient Evidence. "
            "We need at least one trustworthy source before giving a strong true or false verdict."
        )

    if not filtered_source_ids and confidence_label == "High":
        confidence_label = "Low"
        confidence = min(confidence, 0.45)

    if not filtered_source_ids and verdict_label == "Misleading" and confidence > 0.65:
        confidence = 0.65
        confidence_label = "Medium"

    if verdict_label in {"Likely True", "Likely False", "Misleading"} and not trusted_evidence:
        confidence = min(confidence, 0.65)
        confidence_label = "Medium" if confidence >= 0.5 else "Low"
        if verdict_label in {"Likely True", "Likely False"}:
            verdict_label = "Not Enough Evidence"
            pipeline_verdict = "insufficient_evidence"
            confidence = min(confidence, 0.45)
            confidence_label = "Low"
            explanation = "A strong verdict needs support from at least one trusted domain."
            user_response = (
                "JachAI Verdict: Insufficient Evidence. "
                "The available sources are not trusted enough for a strong true or false verdict."
            )

    if not confidence_label:
        confidence_label = _confidence_label_from_score(confidence)

    if _is_poor_verdict_summary(user_response, claim_text=extracted_claim):
        user_response = _build_claim_aware_summary(
            verdict_label=verdict_label,
            claim_text=extracted_claim,
            explanation=explanation,
            evidence=relevant_evidence or evidence,
        )

    return verdict.model_copy(
        update={
            "extracted_claim": extracted_claim,
            "detected_language": detected_language,
            "category": category,
            "verdict": verdict_label,
            "confidence": confidence,
            "confidence_label": confidence_label,
            "explanation": explanation,
            "user_response": user_response,
            "used_source_ids": filtered_source_ids,
            "pipeline_verdict": pipeline_verdict,
            "warnings": warnings,
        }
    )


def _parse_ai_usage(context_payload: dict[str, Any]) -> AIUsageSchema | None:
    payload = context_payload.get("ai_usage")
    if not isinstance(payload, dict):
        return None
    try:
        return AIUsageSchema.model_validate(payload, strict=False)
    except ValidationError:
        return None


# ── Serialization ─────────────────────────────────────────────────────────────

def serialize_job(
    job: VerificationJob,
    live_status: dict[str, Any] | VerificationLiveStatusSchema | None = None,
) -> VerificationJobSchema:
    parsed_live_status: VerificationLiveStatusSchema | None = None
    if isinstance(live_status, VerificationLiveStatusSchema):
        parsed_live_status = live_status
    elif isinstance(live_status, dict) and live_status:
        parsed_live_status = VerificationLiveStatusSchema.model_validate(live_status, strict=False)
    return VerificationJobSchema(
        id=job.id,
        claim_id=job.claim_id,
        input_type=job.input_type,
        status=job.status,
        normalized_hash=job.normalized_hash,
        cached_hit=bool(job.cached_hit),
        error_code=job.error_code,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
        completed_at=job.completed_at,
        live_status=parsed_live_status,
    )


def serialize_claim(claim: Claim) -> ClaimResponseSchema:
    context_payload = claim.context_payload or {}
    evidence_metadata = {
        str(item["source_id"]): item
        for item in context_payload.get("evidence", [])
        if isinstance(item, dict) and item.get("source_id")
    }
    # Build stance lookup from classified_chunks if available
    stance_lookup: dict[str, str] = {}
    credibility_lookup: dict[str, float] = {}
    for chunk in context_payload.get("classified_chunks", []):
        if isinstance(chunk, dict) and chunk.get("source_id"):
            sid = str(chunk["source_id"])
            if "stance" in chunk and sid not in stance_lookup:
                stance_lookup[sid] = str(chunk["stance"])
    for cred in context_payload.get("credibility_scores", []):
        if isinstance(cred, dict) and cred.get("source_id"):
            credibility_lookup[str(cred["source_id"])] = float(cred.get("credibility_score") or 0.5)

    evidence = [
        EvidenceSnippetSchema(
            source_id=link.source.id,
            title=link.source.title,
            url=link.source.url,
            publisher=link.source.publisher,
            language=link.source.language,
            source_type=link.source.source_type,
            snippet=link.source.snippet,
            similarity_score=float(
                evidence_metadata.get(str(link.source.id), {}).get("similarity_score", link.similarity_score)
            ),
            match_score=evidence_metadata.get(str(link.source.id), {}).get("match_score"),
            rerank_score=evidence_metadata.get(str(link.source.id), {}).get("rerank_score"),
            search_score=evidence_metadata.get(str(link.source.id), {}).get("search_score"),
            trust_score=evidence_metadata.get(str(link.source.id), {}).get("trust_score"),
            provider=evidence_metadata.get(str(link.source.id), {}).get("provider"),
            initial_rank=evidence_metadata.get(str(link.source.id), {}).get("initial_rank"),
            final_rank=evidence_metadata.get(str(link.source.id), {}).get("final_rank", link.rank),
            stance=stance_lookup.get(str(link.source.id)),
            credibility_score=credibility_lookup.get(str(link.source.id)),
            snippet_only=bool(evidence_metadata.get(str(link.source.id), {}).get("snippet_only", False)),
        )
        for link in sorted(claim.evidence_links, key=lambda item: item.rank)
        if link.source is not None
    ]
    extracted_claim = str(context_payload.get("extracted_claim") or claim.cleaned_text)
    user_response = str(context_payload.get("user_response") or claim.share_summary)
    if _is_poor_verdict_summary(user_response, claim_text=extracted_claim):
        user_response = _build_claim_aware_summary(
            verdict_label=claim.verdict,
            claim_text=extracted_claim,
            explanation=claim.explanation,
            evidence=[
                {
                    "source_id": str(link.source.id),
                    "trust_score": evidence_metadata.get(str(link.source.id), {}).get("trust_score"),
                    "match_score": evidence_metadata.get(str(link.source.id), {}).get("match_score"),
                    "similarity_score": evidence_metadata.get(str(link.source.id), {}).get("similarity_score"),
                }
                for link in claim.evidence_links
                if link.source is not None
            ],
        )

    return ClaimResponseSchema(
        id=claim.id,
        cluster_id=claim.cluster_id,
        input_type=claim.input_type,
        source_url=claim.source_url,
        raw_text=claim.raw_text,
        cleaned_text=claim.cleaned_text,
        masked_text=claim.masked_text,
        normalized_hash=claim.normalized_hash,
        language=claim.language,
        extracted_claim=extracted_claim,
        detected_language=str(context_payload.get("detected_language") or claim.language),
        category=str(context_payload.get("category") or "Other"),
        review_status=claim.review_status,
        verdict=claim.verdict,
        confidence=claim.confidence,
        confidence_label=str(
            context_payload.get("confidence_label") or _confidence_label_from_score(claim.confidence)
        ),
        explanation=claim.explanation,
        user_response=user_response,
        reasoning=claim.reasoning,
        share_summary=claim.share_summary,
        created_at=claim.created_at,
        updated_at=claim.updated_at,
        evidence=evidence,
        ai_usage=_parse_ai_usage(context_payload),
        context_payload=context_payload,
    )


# ── DB helpers ────────────────────────────────────────────────────────────────

async def _load_claim(session: AsyncSession, claim_id: UUID | str) -> Claim | None:
    parsed_id = _parse_uuid(claim_id, code="INVALID_CLAIM_ID", message="Claim ID must be a valid UUID.")
    statement = (
        select(Claim)
        .options(selectinload(Claim.evidence_links).selectinload(ClaimEvidenceLink.source))
        .where(Claim.id == parsed_id)
    )
    return await session.scalar(statement)


async def _publish_job_status(
    job: VerificationJob,
    live_status: dict[str, Any] | VerificationLiveStatusSchema | None = None,
) -> None:
    await cache_service.set_job_status(
        str(job.id),
        serialize_job(job, live_status=live_status).model_dump(mode="json"),
    )


async def _load_live_status(job_id: UUID | str) -> dict[str, Any] | None:
    cached = await cache_service.get_job_status(str(job_id))
    live_status = cached.get("live_status") if isinstance(cached, dict) else None
    return dict(live_status) if isinstance(live_status, dict) else None


async def _create_job(
    session: AsyncSession,
    input_type: str,
    *,
    status: str = "processing",
    live_status: dict[str, Any] | None = None,
) -> VerificationJob:
    job = VerificationJob(input_type=input_type, status=status, cached_hit=False)
    session.add(job)
    await session.commit()
    await session.refresh(job)
    await _publish_job_status(job, live_status=live_status)
    return job


async def _ensure_job_processing(
    session: AsyncSession,
    job: VerificationJob,
    *,
    live_status: dict[str, Any] | None = None,
) -> VerificationJob:
    if job.status != "processing":
        job.status = "processing"
        job.error_code = None
        job.error_message = None
        await session.commit()
        await session.refresh(job)
    await _publish_job_status(job, live_status=live_status)
    return job


async def _complete_job(
    session: AsyncSession,
    job: VerificationJob,
    *,
    claim_id: UUID | None,
    cached_hit: bool,
    status: str = "completed",
    live_status: dict[str, Any] | None = None,
) -> None:
    job.claim_id = claim_id
    job.cached_hit = cached_hit
    job.status = status
    job.completed_at = utc_now()
    await session.commit()
    await session.refresh(job)
    await _publish_job_status(job, live_status=live_status)


async def _fail_job(
    session: AsyncSession,
    job: VerificationJob,
    *,
    code: str,
    message: str,
    live_status: dict[str, Any] | None = None,
) -> None:
    await session.rollback()
    persisted_job = await session.get(VerificationJob, job.id)
    if persisted_job is None:
        return

    persisted_job.status = "failed"
    persisted_job.error_code = code
    persisted_job.error_message = message
    persisted_job.completed_at = utc_now()
    await session.commit()
    await session.refresh(persisted_job)
    await _publish_job_status(persisted_job, live_status=live_status)


async def _fetch_url_text(url: str) -> tuple[str, dict[str, Any]]:
    """Fetch a URL and extract clean text. Used for URL-input claims."""
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=settings.fetch_timeout_seconds,
            headers={"User-Agent": "JachAI/0.1"},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise AppError(
            status_code=422, code="URL_FETCH_FAILED", message=f"Could not fetch URL: {exc}"
        ) from exc

    text = text_from_html(response.text)
    if len(text) < 20:
        raise AppError(
            status_code=422,
            code="URL_CONTENT_TOO_SHORT",
            message="The URL did not contain enough text to verify.",
        )
    return text, {"fetched_url": str(response.url), "status_code": response.status_code}


async def _return_existing_claim(
    session: AsyncSession,
    job: VerificationJob,
    claim: Claim,
    *,
    cached_hit: bool,
    live_status: dict[str, Any] | None = None,
) -> ClaimSubmissionResponseSchema:
    await _complete_job(session, job, claim_id=claim.id, cached_hit=cached_hit, live_status=live_status)
    serialized = serialize_claim(claim)
    await cache_service.set_duplicate_claim_id(claim.normalized_hash, str(claim.id))
    await cache_service.set_claim_result(
        claim.normalized_hash,
        {"claim_id": str(claim.id), "job_id": str(job.id), "cached": cached_hit},
    )
    return ClaimSubmissionResponseSchema(
        job=serialize_job(job, live_status=live_status),
        claim=serialized,
        cached=cached_hit,
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def _pipeline_from_text(
    session: AsyncSession,
    *,
    input_type: str,
    raw_text: str,
    source_url: str | None = None,
    context_payload: dict[str, Any] | None = None,
    job: VerificationJob | None = None,
    live_status: dict[str, Any] | None = None,
) -> ClaimSubmissionResponseSchema:
    """
    Full evidence-based claim verification pipeline.

    Stages:
    1. Input normalization
    2. Claim extraction (LLM)
    3. Query generation (LLM)
    4. Tavily search + evidence fetching
    5. Evidence chunking
    6. Evidence reranking (NVIDIA or local fallback)
    7. Source credibility scoring
    8. Stance classification (LLM, per chunk)
    9. pgvector retrieval (parallel, legacy path)
    10. Final verdict (LLM, evidence-grounded)
    11. Guardrails
    12. Save + return
    """
    if job is None:
        live_status = live_status or _build_initial_live_status(
            input_type=input_type,
            raw_input=raw_text,
            source_url=source_url,
            context_payload=context_payload,
        )
        job = await _create_job(
            session,
            input_type=input_type,
            status="processing",
            live_status=live_status,
        )
    else:
        live_status = live_status or await _load_live_status(job.id) or _build_initial_live_status(
            input_type=input_type,
            raw_input=raw_text,
            source_url=source_url,
            context_payload=context_payload,
        )
        await _ensure_job_processing(session, job, live_status=live_status)

    try:
        pipeline_start = time.perf_counter()
        claim_context = dict(context_payload or {})
        live_status["raw_input"] = live_status.get("raw_input") or _preview_text(raw_text)
        _set_live_stage(live_status, "queued")
        _merge_live_details(
            live_status,
            "input",
            {
                "input_type": input_type,
                "source_url": source_url,
            },
        )
        await _publish_job_status(job, live_status)

        # ─────────────────────────────────────────────────────────────────────
        # Stage 1: Input normalization
        # ─────────────────────────────────────────────────────────────────────
        stage_start = time.perf_counter()
        cleaned_text = clean_text(raw_text)
        if len(cleaned_text) < 5:
            raise AppError(status_code=422, code="INPUT_TOO_SHORT", message="Claim text is too short.")
        masked_text = mask_pii(cleaned_text)
        language = detect_language(masked_text)
        content_hash = normalized_hash(masked_text)
        hash_value = normalized_hash(f"{settings.verification_pipeline_version}:{content_hash}")
        claim_context.update({
            "content_hash": content_hash,
            "verification_pipeline_version": settings.verification_pipeline_version,
        })
        logger.info(
            "pipeline_stage stage=input_normalization duration_ms=%.1f "
            "hash=%.16s language=%s",
            (time.perf_counter() - stage_start) * 1000, hash_value, language,
        )

        job.normalized_hash = hash_value
        await session.commit()
        await session.refresh(job)
        live_status["cleaned_input"] = _preview_text(cleaned_text)
        live_status["masked_input"] = _preview_text(masked_text)
        live_status["detected_language"] = language
        _merge_live_details(
            live_status,
            "input",
            {
                "language": language,
                "normalized_hash": hash_value,
                "content_hash": content_hash,
                "verification_pipeline_version": settings.verification_pipeline_version,
            },
        )
        _append_live_event(
            live_status,
            key="input_normalized",
            label="Input normalized",
            status="completed",
            summary=f"Detected {language} input and generated a normalized verification hash.",
            metadata={"language": language, "normalized_hash": hash_value[:12]},
        )
        await _publish_job_status(job, live_status)

        # ── Cache / dedup check ───────────────────────────────────────────────
        duplicate_claim_id = await cache_service.get_duplicate_claim_id(hash_value)
        if duplicate_claim_id:
            existing = await _load_claim(session, duplicate_claim_id)
            if existing:
                serialized_existing = serialize_claim(existing)
                _append_live_event(
                    live_status,
                    key="cache_hit",
                    label="Cache hit",
                    status="cached",
                    summary="Matched a previously verified claim and reused the stored verdict.",
                    metadata={"claim_id": str(existing.id)},
                )
                live_status["extracted_claim"] = serialized_existing.extracted_claim
                live_status["retrieval_query"] = serialized_existing.extracted_claim
                live_status["detected_language"] = serialized_existing.detected_language
                _merge_live_details(
                    live_status,
                    "verdict",
                    {
                        "cached_hit": True,
                        "verdict": serialized_existing.verdict,
                        "confidence": serialized_existing.confidence,
                        "confidence_label": serialized_existing.confidence_label,
                        "summary": serialized_existing.user_response,
                    },
                )
                _set_live_stage(live_status, "completed", message="Matched a previously verified claim.")
                return await _return_existing_claim(
                    session,
                    job,
                    existing,
                    cached_hit=True,
                    live_status=live_status,
                )

        cached_payload = await cache_service.get_claim_result(hash_value)
        if cached_payload and cached_payload.get("claim_id"):
            existing = await _load_claim(session, cached_payload["claim_id"])
            if existing:
                serialized_existing = serialize_claim(existing)
                _append_live_event(
                    live_status,
                    key="result_cache_hit",
                    label="Result cache hit",
                    status="cached",
                    summary="Loaded a previously generated verdict from the claim-result cache.",
                    metadata={"claim_id": str(existing.id)},
                )
                live_status["extracted_claim"] = serialized_existing.extracted_claim
                live_status["retrieval_query"] = serialized_existing.extracted_claim
                live_status["detected_language"] = serialized_existing.detected_language
                _merge_live_details(
                    live_status,
                    "verdict",
                    {
                        "cached_hit": True,
                        "verdict": serialized_existing.verdict,
                        "confidence": serialized_existing.confidence,
                        "confidence_label": serialized_existing.confidence_label,
                        "summary": serialized_existing.user_response,
                    },
                )
                _set_live_stage(live_status, "completed", message="Loaded a previous verification result from cache.")
                return await _return_existing_claim(
                    session,
                    job,
                    existing,
                    cached_hit=True,
                    live_status=live_status,
                )

        existing_db_claim = await session.scalar(
            select(Claim.id).where(Claim.normalized_hash == hash_value).limit(1)
        )
        if existing_db_claim:
            existing = await _load_claim(session, existing_db_claim)
            if existing:
                serialized_existing = serialize_claim(existing)
                _append_live_event(
                    live_status,
                    key="database_match",
                    label="Existing verification reused",
                    status="cached",
                    summary="Found an existing verified claim in the database and reused its result.",
                    metadata={"claim_id": str(existing.id)},
                )
                live_status["extracted_claim"] = serialized_existing.extracted_claim
                live_status["retrieval_query"] = serialized_existing.extracted_claim
                live_status["detected_language"] = serialized_existing.detected_language
                _merge_live_details(
                    live_status,
                    "verdict",
                    {
                        "cached_hit": True,
                        "verdict": serialized_existing.verdict,
                        "confidence": serialized_existing.confidence,
                        "confidence_label": serialized_existing.confidence_label,
                        "summary": serialized_existing.user_response,
                    },
                )
                _set_live_stage(live_status, "completed", message="Found an existing verification in the database.")
                return await _return_existing_claim(
                    session,
                    job,
                    existing,
                    cached_hit=True,
                    live_status=live_status,
                )

        if not await cache_service.reserve_uncached_claim_slot():
            raise AppError(
                status_code=429,
                code="AI_RATE_LIMITED",
                message="Verification is busy right now. Please try again shortly.",
            )

        _set_live_stage(live_status, "extracting_claim")
        await _publish_job_status(job, live_status)

        # ─────────────────────────────────────────────────────────────────────
        # Stage 2: Claim extraction
        # ─────────────────────────────────────────────────────────────────────
        stage_start = time.perf_counter()
        extraction, extraction_metadata = await extract_claim_context(
            masked_text,
            language,
            cache_key=hash_value,
            source_kind="image_ocr" if input_type == "image" else "text",
        )
        retrieval_query = extraction.extracted_claim or masked_text
        logger.info(
            "pipeline_stage stage=claim_extraction duration_ms=%.1f "
            "claim=%.80s category=%s entities=%d requires_freshness=%s",
            (time.perf_counter() - stage_start) * 1000,
            retrieval_query, extraction.category,
            len(extraction.entities), extraction.requires_freshness,
        )
        live_status["extracted_claim"] = _preview_text(extraction.extracted_claim)
        live_status["retrieval_query"] = _preview_text(retrieval_query)
        live_status["detected_language"] = extraction.detected_language or language
        _merge_live_details(
            live_status,
            "extraction",
            {
                "category": extraction.category,
                "entities": extraction.entities,
                "time_context": extraction.time_context,
                "location_context": extraction.location_context,
                "requires_freshness": extraction.requires_freshness,
                "verification_strategy": extraction.verification_strategy,
                "model": extraction_metadata.get("model"),
                "provider": extraction_metadata.get("provider"),
                "cached": extraction_metadata.get("cached", False),
            },
        )
        _append_live_event(
            live_status,
            key="claim_extracted",
            label="Claim extracted",
            status="completed",
            summary="Captured the factual claim and translated it into a normalized verification target.",
            metadata={
                "category": extraction.category,
                "language": extraction.detected_language or language,
                "entity_count": len(extraction.entities),
            },
        )
        _set_live_stage(live_status, "searching_sources")
        await _publish_job_status(job, live_status)

        # ─────────────────────────────────────────────────────────────────────
        # Stage 3: Query generation
        # ─────────────────────────────────────────────────────────────────────
        stage_start = time.perf_counter()
        query_generation, query_metadata = await generate_search_queries(
            extraction=extraction,
            original_text=masked_text,
            cache_key=hash_value,
        )
        flat_queries = query_generation.search_queries
        english_retrieval_query = clean_text(str(query_metadata.get("english_query") or "")) or None
        retrieval_variants = _dedupe_text_variants([retrieval_query, english_retrieval_query])
        retrieval_match_query = english_retrieval_query or retrieval_query
        logger.info(
            "pipeline_stage stage=query_generation duration_ms=%.1f query_count=%d queries=%s",
            (time.perf_counter() - stage_start) * 1000,
            len(flat_queries),
            flat_queries[:3],
        )
        live_status["search_queries"] = [str(query) for query in flat_queries[:8]]
        _merge_live_details(
            live_status,
            "retrieval",
            {
                "query_count": len(flat_queries),
                "search_queries": [str(query) for query in flat_queries[:8]],
                "query_model": query_metadata.get("model"),
                "query_cached": query_metadata.get("cached", False),
                "cross_lingual_query": _preview_text(english_retrieval_query, limit=180),
            },
        )
        _append_live_event(
            live_status,
            key="queries_generated",
            label="Search queries generated",
            status="completed",
            summary=f"Built {len(flat_queries)} retrieval quer{'y' if len(flat_queries) == 1 else 'ies'} for evidence search.",
            metadata={"query_count": len(flat_queries)},
        )
        await _publish_job_status(job, live_status)

        # ─────────────────────────────────────────────────────────────────────
        # Stage 4: Tavily search + evidence fetching
        # ─────────────────────────────────────────────────────────────────────
        stage_start = time.perf_counter()
        live_evidence = await hydrate_live_evidence(
            session,
            claim_text=retrieval_query,
            language=extraction.detected_language or language,
            normalized_hash=hash_value,
            search_queries=flat_queries,
            requires_freshness=extraction.requires_freshness,
        )
        claim_context["live_evidence"] = live_evidence
        evidence_documents = live_evidence.get("evidence_documents", [])
        pipeline_warnings: list[str] = list(live_evidence.get("pipeline_warnings", []))
        logger.info(
            "pipeline_stage stage=tavily_search duration_ms=%.1f "
            "tavily_results=%d fetched_docs=%d failed_fetches=%d warnings=%d",
            (time.perf_counter() - stage_start) * 1000,
            live_evidence.get("tavily_result_count", 0),
            len(evidence_documents),
            live_evidence.get("tavily_result_count", 0) - len(evidence_documents),
            len(pipeline_warnings),
        )
        _set_live_warning_messages(live_status, pipeline_warnings)
        _merge_live_details(
            live_status,
            "retrieval",
            {
                "tavily_result_count": int(live_evidence.get("tavily_result_count", 0)),
                "fetched_document_count": len(evidence_documents),
                "failed_fetch_count": max(
                    0,
                    int(live_evidence.get("tavily_result_count", 0)) - len(evidence_documents),
                ),
                "tavily_request_count": int(live_evidence.get("tavily_request_count", 0)),
                "search_answer_context": _truncate_metadata(live_evidence.get("search_answer_context")),
                "top_sources": _evidence_preview(evidence_documents),
            },
        )
        _append_live_event(
            live_status,
            key="live_search_complete",
            label="Live retrieval complete",
            status="completed",
            summary=f"Collected {len(evidence_documents)} live document(s) from Tavily and trusted sources.",
            metadata={
                "tavily_results": int(live_evidence.get("tavily_result_count", 0)),
                "documents": len(evidence_documents),
            },
        )
        await _publish_job_status(job, live_status)

        # ─────────────────────────────────────────────────────────────────────
        # Stage 5: Evidence chunking
        # ─────────────────────────────────────────────────────────────────────
        stage_start = time.perf_counter()
        evidence_chunks = chunk_evidence_documents(evidence_documents)
        logger.info(
            "pipeline_stage stage=evidence_chunking duration_ms=%.1f chunk_count=%d",
            (time.perf_counter() - stage_start) * 1000,
            len(evidence_chunks),
        )
        _merge_live_details(
            live_status,
            "retrieval",
            {
                "chunk_count": len(evidence_chunks),
            },
        )
        _append_live_event(
            live_status,
            key="evidence_chunked",
            label="Evidence chunked",
            status="completed",
            summary=f"Split retrieved documents into {len(evidence_chunks)} comparison-ready evidence chunk(s).",
            metadata={"chunk_count": len(evidence_chunks)},
        )
        _set_live_stage(live_status, "comparing_evidence")
        await _publish_job_status(job, live_status)

        # ─────────────────────────────────────────────────────────────────────
        # Stage 6: Evidence ranking / reranking
        # ─────────────────────────────────────────────────────────────────────
        stage_start = time.perf_counter()
        ranked_chunks, chunk_rerank_metadata = await rerank_evidence(
            retrieval_query,
            evidence_chunks,
            claim_text=retrieval_query,
            use_chunk_format=True,
        )
        logger.info(
            "pipeline_stage stage=evidence_ranking duration_ms=%.1f "
            "input_chunks=%d ranked_chunks=%d rerank_applied=%s",
            (time.perf_counter() - stage_start) * 1000,
            len(evidence_chunks),
            len(ranked_chunks),
            chunk_rerank_metadata.get("applied"),
        )
        _merge_live_details(
            live_status,
            "comparison",
            {
                "ranked_chunk_count": len(ranked_chunks),
                "chunk_rerank_applied": bool(chunk_rerank_metadata.get("applied")),
                "top_ranked_chunks": _evidence_preview(ranked_chunks),
            },
        )
        _append_live_event(
            live_status,
            key="evidence_ranked",
            label="Evidence ranked",
            status="completed",
            summary=f"Ranked {len(ranked_chunks)} chunk(s) by relevance to the extracted claim.",
            metadata={
                "ranked_chunks": len(ranked_chunks),
                "rerank_applied": bool(chunk_rerank_metadata.get("applied")),
            },
        )
        await _publish_job_status(job, live_status)

        # ─────────────────────────────────────────────────────────────────────
        # Stage 7: Source credibility scoring
        # ─────────────────────────────────────────────────────────────────────
        stage_start = time.perf_counter()
        credibility_scores: list[dict[str, Any]] = []
        seen_sources: set[str] = set()
        for chunk in ranked_chunks:
            source_id = str(chunk.get("source_id") or "")
            if source_id and source_id not in seen_sources:
                seen_sources.add(source_id)
                cred = score_source_credibility(
                    source_id=source_id,
                    domain=chunk.get("domain"),
                    snippet_only=bool(chunk.get("snippet_only", False)),
                    published_date=chunk.get("published_date"),
                    requires_freshness=extraction.requires_freshness,
                )
                credibility_scores.append(cred)
        logger.info(
            "pipeline_stage stage=source_credibility duration_ms=%.1f sources_scored=%d",
            (time.perf_counter() - stage_start) * 1000,
            len(credibility_scores),
        )
        _merge_live_details(
            live_status,
            "comparison",
            {
                "credibility_scores": _truncate_metadata(credibility_scores),
                "source_count_scored": len(credibility_scores),
            },
        )
        _append_live_event(
            live_status,
            key="credibility_scored",
            label="Source credibility scored",
            status="completed",
            summary=f"Scored trust signals for {len(credibility_scores)} source(s).",
            metadata={"source_count": len(credibility_scores)},
        )
        await _publish_job_status(job, live_status)

        # ─────────────────────────────────────────────────────────────────────
        # Stage 8: Stance classification
        # ─────────────────────────────────────────────────────────────────────
        stage_start = time.perf_counter()
        # Select top chunks for classification (cap at max_evidence_chunks)
        top_chunks_for_classification = ranked_chunks[: settings.max_evidence_chunks]
        classified_chunks = await classify_evidence_stances(
            top_chunks_for_classification,
            normalized_claim=retrieval_query,
        )
        supports_count = sum(1 for c in classified_chunks if c.get("stance") == "supports")
        refutes_count = sum(1 for c in classified_chunks if c.get("stance") == "refutes")
        logger.info(
            "pipeline_stage stage=stance_classification duration_ms=%.1f "
            "chunks=%d supports=%d refutes=%d neutral=%d",
            (time.perf_counter() - stage_start) * 1000,
            len(classified_chunks),
            supports_count,
            refutes_count,
            len(classified_chunks) - supports_count - refutes_count,
        )
        _merge_live_details(
            live_status,
            "comparison",
            {
                "classified_chunk_count": len(classified_chunks),
                "supports_count": supports_count,
                "refutes_count": refutes_count,
                "neutral_count": len(classified_chunks) - supports_count - refutes_count,
                "classified_chunks": _truncate_metadata(classified_chunks),
            },
        )
        _append_live_event(
            live_status,
            key="stances_classified",
            label="Evidence stance classified",
            status="completed",
            summary=(
                f"Compared the strongest chunks: {supports_count} support, {refutes_count} refute, "
                f"{len(classified_chunks) - supports_count - refutes_count} neutral."
            ),
            metadata={
                "supports": supports_count,
                "refutes": refutes_count,
                "neutral": len(classified_chunks) - supports_count - refutes_count,
            },
        )
        await _publish_job_status(job, live_status)

        # ─────────────────────────────────────────────────────────────────────
        # Stage 9: pgvector retrieval (legacy path — runs in parallel with live evidence)
        # ─────────────────────────────────────────────────────────────────────
        stage_start = time.perf_counter()
        evidence_index = await get_evidence_index_status(session)
        claim_context["evidence_index"] = {
            "ready": evidence_index.ready,
            "total_sources": evidence_index.total_sources,
            "real_sources": evidence_index.real_sources,
            "sample_sources": evidence_index.sample_sources,
            "message": evidence_index.message,
        }

        retrieved_result_sets: list[list[tuple[Any, float]]] = []
        for query_variant in retrieval_variants:
            embedding = await embed_text(
                query_variant,
                task_type="RETRIEVAL_QUERY",
            )
            retrieved_result_sets.append(
                await retrieve_evidence(session, embedding, top_k=settings.pgvector_top_k)
            )
        retrieved = _merge_retrieved_result_sets(
            retrieved_result_sets,
            limit=max(settings.pgvector_top_k, settings.pgvector_top_k * len(retrieval_variants)),
        )
        retrieved_candidates = _build_evidence_candidates(retrieved, claim_text=retrieval_match_query)
        selected_evidence, rerank_metadata = await rerank_evidence(retrieval_match_query, retrieved_candidates)
        selected_evidence = _filter_selected_evidence(selected_evidence)
        logger.info(
            "pipeline_stage stage=pgvector_retrieval duration_ms=%.1f "
            "retrieved=%d selected=%d rerank_applied=%s variants=%d",
            (time.perf_counter() - stage_start) * 1000,
            len(retrieved_candidates),
            len(selected_evidence),
            rerank_metadata.get("applied"),
            len(retrieval_variants),
        )
        _merge_live_details(
            live_status,
            "retrieval",
            {
                "pgvector_ready": evidence_index.ready,
                "index_sources": evidence_index.real_sources,
                "pgvector_candidates": len(retrieved_candidates),
                "selected_evidence_count": len(selected_evidence),
                "selected_evidence": _evidence_preview(selected_evidence),
                "retrieval_variant_count": len(retrieval_variants),
            },
        )
        _merge_live_details(
            live_status,
            "comparison",
            {
                "selected_evidence_count": len(selected_evidence),
                "selected_evidence": _evidence_preview(selected_evidence),
                "rerank_model": rerank_metadata.get("model"),
                "rerank_applied": bool(rerank_metadata.get("applied")),
            },
        )
        _append_live_event(
            live_status,
            key="vector_retrieval_complete",
            label="Vector retrieval complete",
            status="completed",
            summary=f"Selected {len(selected_evidence)} final evidence source(s) for verdict generation.",
            metadata={
                "candidates": len(retrieved_candidates),
                "selected": len(selected_evidence),
            },
        )
        _set_live_stage(live_status, "generating_verdict")
        await _publish_job_status(job, live_status)

        # ─────────────────────────────────────────────────────────────────────
        # Stage 10: Final verdict generation
        # ─────────────────────────────────────────────────────────────────────
        stage_start = time.perf_counter()
        verdict, llm_metadata = await generate_verdict(
            extraction,
            selected_evidence,
            classified_chunks=classified_chunks,
            credibility_scores=credibility_scores,
            pipeline_warnings=pipeline_warnings,
        )
        verdict = _apply_verdict_guardrails(
            verdict,
            claim_text=cleaned_text,
            language=language,
            evidence=selected_evidence,
            classified_chunks=classified_chunks,
        )
        pipeline_verdict = getattr(verdict, "pipeline_verdict", None)
        logger.info(
            "pipeline_stage stage=verdict_generation duration_ms=%.1f "
            "verdict=%s pipeline_verdict=%s confidence=%.2f warnings=%d",
            (time.perf_counter() - stage_start) * 1000,
            verdict.verdict, pipeline_verdict,
            verdict.confidence, len(getattr(verdict, "warnings", [])),
        )
        _set_live_warning_messages(live_status, list(getattr(verdict, "warnings", [])))
        _merge_live_details(
            live_status,
            "verdict",
            {
                "verdict": verdict.verdict,
                "pipeline_verdict": pipeline_verdict,
                "confidence": verdict.confidence,
                "confidence_label": verdict.confidence_label,
                "explanation": _preview_text(verdict.explanation, limit=260),
                "user_response": _preview_text(verdict.user_response, limit=260),
                "used_source_ids": [str(source_id) for source_id in verdict.used_source_ids],
            },
        )
        _append_live_event(
            live_status,
            key="verdict_generated",
            label="Verdict generated",
            status="completed",
            summary=f"Drafted a {verdict.verdict} verdict with {verdict.confidence_label.lower()} confidence.",
            metadata={
                "verdict": verdict.verdict,
                "confidence": verdict.confidence,
                "confidence_label": verdict.confidence_label,
            },
        )
        await _publish_job_status(job, live_status)

        # ─────────────────────────────────────────────────────────────────────
        # Stage 11: Build AI usage payload
        # ─────────────────────────────────────────────────────────────────────
        reasoning_provider = llm_metadata.get("provider") or "nvidia"
        reasoning_model = llm_metadata.get("model") or get_model_for_task(NVIDIAModelTask.CLAIM_REASONING)
        claim_extraction_model = (
            extraction_metadata.get("model") or get_model_for_task(NVIDIAModelTask.CLAIM_EXTRACTION)
        )
        query_generation_model = (
            query_metadata.get("model") or get_model_for_task(NVIDIAModelTask.SEARCH_QUERY_GENERATION)
        )
        rerank_model = rerank_metadata.get("model")
        vision_model = str(claim_context.get("ocr_model")) if claim_context.get("ocr_model") else None
        ai_usage = _build_ai_usage_payload(
            claim_extraction_model=claim_extraction_model,
            query_generation_model=query_generation_model,
            reasoning_model=reasoning_model,
            rerank_model=rerank_model,
            vision_model=vision_model,
            search_provider="tavily",
            llm_call_count=(
                int(extraction_metadata.get("call_count", 0))
                + int(query_metadata.get("call_count", 0))
                + int(llm_metadata.get("call_count", 0))
                + len(classified_chunks)  # one call per classified chunk
            ),
            rerank_call_count=(
                int(rerank_metadata.get("call_count", 0))
                + int(chunk_rerank_metadata.get("call_count", 0))
            ),
            vision_call_count=1 if claim_context.get("ocr_method") == "kimi_ocr" else 0,
            tavily_request_count=int(live_evidence.get("tavily_request_count", 0)),
        )
        _merge_live_details(
            live_status,
            "models",
            {
                "claim_extraction_model": claim_extraction_model,
                "query_generation_model": query_generation_model,
                "reasoning_provider": reasoning_provider,
                "reasoning_model_used": reasoning_model,
                "reasoning_candidates": settings.active_gemini_reasoning_models
                or ([get_model_for_task(NVIDIAModelTask.CLAIM_REASONING)] if get_model_for_task(NVIDIAModelTask.CLAIM_REASONING) else []),
                "reasoning_fallback_used": bool(
                    settings.active_gemini_reasoning_models
                    and reasoning_model
                    and reasoning_model != settings.active_gemini_reasoning_models[0]
                ),
                "rerank_model": rerank_model,
                "vision_model": vision_model,
                "embedding_model": settings.embedding_model,
            },
        )
        await _publish_job_status(job, live_status)

        claim_context.update({
            "extracted_claim": verdict.extracted_claim,
            "detected_language": verdict.detected_language,
            "category": verdict.category,
            "confidence_label": verdict.confidence_label,
            "user_response": verdict.user_response,
            "factual_summary": verdict.user_response,
            "pipeline_verdict": pipeline_verdict,
            "pipeline_warnings": pipeline_warnings + list(getattr(verdict, "warnings", [])),
            "claim_extraction_model": claim_extraction_model,
            "query_generation_model": query_generation_model,
            "reasoning_provider": reasoning_provider,
            "reasoning_model": reasoning_model,
            "nvidia_reasoning_model": reasoning_model if reasoning_provider == "nvidia" else None,
            "claim_extraction": {
                "query_used_for_retrieval": retrieval_query,
                "retrieval_match_query": retrieval_match_query,
                "retrieval_variants": retrieval_variants,
                "entities": extraction.entities,
                "time_context": extraction.time_context,
                "location_context": extraction.location_context,
                "requires_freshness": extraction.requires_freshness,
                "verification_strategy": extraction.verification_strategy,
                "call_count": int(extraction_metadata.get("call_count", 0)),
            },
            "search_query_generation": {
                "queries": [q.model_dump() for q in query_generation.queries],
                "search_queries": flat_queries,
                "english_query": english_retrieval_query,
                "call_count": int(query_metadata.get("call_count", 0)),
            },
            "search_answer_context": live_evidence.get("search_answer_context"),
            "search_provider": "tavily",
            "retrieval": {
                "candidate_count": len(retrieved_candidates),
                "selected_count": len(selected_evidence),
                "pgvector_top_k": settings.pgvector_top_k,
                "final_evidence_top_k": settings.final_evidence_top_k,
                "min_relevant_similarity": settings.min_relevant_similarity,
            },
            "evidence_chunking": {
                "doc_count": len(evidence_documents),
                "chunk_count": len(evidence_chunks),
                "ranked_chunk_count": len(ranked_chunks),
                "classified_chunk_count": len(classified_chunks),
            },
            "rerank": rerank_metadata,
            "chunk_rerank": chunk_rerank_metadata,
            "credibility_scores": credibility_scores,
            "classified_chunks": [
                {
                    "chunk_id": c.get("chunk_id"),
                    "source_id": c.get("source_id"),
                    "stance": c.get("stance"),
                    "stance_confidence": c.get("stance_confidence"),
                    "rationale": c.get("rationale"),
                    "quoted_evidence": c.get("quoted_evidence"),
                }
                for c in classified_chunks
            ],
            "evidence": selected_evidence,
            "ai_usage": ai_usage,
        })

        # ─────────────────────────────────────────────────────────────────────
        # Stage 12: Persist claim
        # ─────────────────────────────────────────────────────────────────────
        cluster = await get_or_create_cluster(
            session,
            topic_hash=hash_value,
            title=verdict.extracted_claim[:120],
            language=verdict.detected_language,
            summary=verdict.user_response,
        )

        claim = Claim(
            cluster_id=cluster.id,
            input_type=input_type,
            source_url=source_url,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            masked_text=masked_text,
            normalized_hash=hash_value,
            language=language,
            verdict=verdict.verdict,
            confidence=verdict.confidence,
            explanation=verdict.explanation,
            reasoning=_build_reasoning_text(
                verdict,
                selected_evidence,
                rerank_applied=bool(rerank_metadata.get("applied")),
                classified_chunks=classified_chunks,
            ),
            share_summary=verdict.user_response,
            review_status="pending",
            context_payload=claim_context,
        )
        session.add(claim)
        await session.flush()
        _append_live_event(
            live_status,
            key="claim_persisted",
            label="Verification persisted",
            status="completed",
            summary="Stored the verified claim, evidence links, and reasoning trace.",
            metadata={"claim_id": str(claim.id)},
        )
        await _publish_job_status(job, live_status)

        if cluster.representative_claim_id is None:
            cluster.representative_claim_id = claim.id

        source_lookup = {str(item["source_id"]): item for item in selected_evidence}
        ordered_source_ids = [
            str(source_id)
            for source_id in verdict.used_source_ids
            if str(source_id) in source_lookup
        ]
        fallback_ids = [
            source_id for source_id in source_lookup.keys() if source_id not in ordered_source_ids
        ]
        for position, source_id in enumerate(ordered_source_ids + fallback_ids, start=1):
            source_item = source_lookup[source_id]
            session.add(
                ClaimEvidenceLink(
                    claim_id=claim.id,
                    source_id=UUID(source_id),
                    rank=int(source_item.get("final_rank") or position),
                    similarity_score=float(source_item["similarity_score"]),
                )
            )

        session.add(
            AuditLog(
                actor="pipeline",
                action="claim_verified",
                entity_type="claim",
                entity_id=str(claim.id),
                details={
                    "normalized_hash": hash_value,
                    "content_hash": content_hash,
                    "verification_pipeline_version": settings.verification_pipeline_version,
                    "language": language,
                    "input_type": input_type,
                    "pipeline_verdict": pipeline_verdict,
                    "verdict": verdict.verdict,
                    "ai_usage": ai_usage,
                },
            )
        )
        try:
            await session.commit()
        except IntegrityError:
            await session.rollback()
            existing_claim_id = await session.scalar(
                select(Claim.id).where(Claim.normalized_hash == hash_value).limit(1)
            )
            if existing_claim_id:
                existing = await _load_claim(session, existing_claim_id)
                if existing:
                    return await _return_existing_claim(session, job, existing, cached_hit=True)
            raise

        persisted_claim = await _load_claim(session, claim.id)
        if persisted_claim is None:
            raise AppError(status_code=500, code="CLAIM_SAVE_FAILED", message="Claim could not be reloaded.")

        live_status["extracted_claim"] = _preview_text(verdict.extracted_claim)
        live_status["retrieval_query"] = _preview_text(retrieval_query)
        live_status["detected_language"] = verdict.detected_language
        _merge_live_details(
            live_status,
            "verdict",
            {
                "claim_id": str(persisted_claim.id),
                "cached_hit": False,
                "verdict": verdict.verdict,
                "confidence": verdict.confidence,
                "confidence_label": verdict.confidence_label,
                "summary": _preview_text(verdict.user_response, limit=260),
            },
        )
        _set_live_stage(live_status, "completed")
        await _complete_job(
            session,
            job,
            claim_id=persisted_claim.id,
            cached_hit=False,
            live_status=live_status,
        )
        await cache_service.set_duplicate_claim_id(hash_value, str(persisted_claim.id))
        await cache_service.set_claim_result(
            hash_value,
            {"claim_id": str(persisted_claim.id), "job_id": str(job.id), "cached": False},
        )

        total_ms = (time.perf_counter() - pipeline_start) * 1000
        logger.info(
            "pipeline_complete total_ms=%.1f verdict=%s pipeline_verdict=%s "
            "evidence_docs=%d chunks=%d classified=%d warnings=%d",
            total_ms, verdict.verdict, pipeline_verdict,
            len(evidence_documents), len(evidence_chunks),
            len(classified_chunks), len(pipeline_warnings),
        )
        return ClaimSubmissionResponseSchema(
            job=serialize_job(job, live_status=live_status),
            claim=serialize_claim(persisted_claim),
            cached=False,
        )
    except AppError as exc:
        _set_live_stage(live_status, "failed", message=exc.message)
        _append_live_event(
            live_status,
            key="verification_failed",
            label="Verification failed",
            status="failed",
            summary=exc.message,
            metadata={"error_code": exc.code},
        )
        await _fail_job(session, job, code=exc.code, message=exc.message, live_status=live_status)
        raise
    except Exception:
        _set_live_stage(
            live_status,
            "failed",
            message="Claim verification pipeline failed unexpectedly.",
        )
        _append_live_event(
            live_status,
            key="verification_failed",
            label="Verification failed",
            status="failed",
            summary="Claim verification pipeline failed unexpectedly.",
            metadata={"error_code": "PIPELINE_FAILED"},
        )
        await _fail_job(
            session,
            job,
            code="PIPELINE_FAILED",
            message="Claim verification pipeline failed unexpectedly.",
            live_status=live_status,
        )
        raise


# ── Public pipeline entry points ──────────────────────────────────────────────

async def process_text_claim(
    session: AsyncSession,
    text: str,
    external_id: str | None = None,
) -> ClaimSubmissionResponseSchema:
    context = {"external_id": external_id} if external_id else {}
    return await _pipeline_from_text(
        session,
        input_type="text",
        raw_text=text,
        context_payload=context,
    )


async def process_url_claim(
    session: AsyncSession,
    url: str,
    external_id: str | None = None,
    supporting_text: str | None = None,
) -> ClaimSubmissionResponseSchema:
    text, metadata = await _fetch_url_text(url)
    if external_id:
        metadata["external_id"] = external_id
    cleaned_supporting_text = _normalize_supporting_text(supporting_text)
    if cleaned_supporting_text:
        metadata["supporting_text"] = cleaned_supporting_text
    combined_text = _compose_claim_source_input(
        primary_text=text,
        supporting_text=cleaned_supporting_text,
        source_label="Page text",
    )
    return await _pipeline_from_text(
        session,
        input_type="url",
        raw_text=combined_text,
        source_url=url,
        context_payload=metadata,
    )


async def process_image_claim(
    session: AsyncSession,
    image_bytes: bytes,
    filename: str | None = None,
    content_type: str | None = None,
    external_id: str | None = None,
    supporting_text: str | None = None,
) -> ClaimSubmissionResponseSchema:
    max_bytes = settings.max_image_size_mb * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise AppError(
            status_code=422,
            code="IMAGE_TOO_LARGE",
            message=f"Image must be {settings.max_image_size_mb} MB or smaller.",
        )

    ocr_text = await extract_text_from_image(image_bytes, mime_type=content_type)
    text = prepare_ocr_text_for_claim_extraction(ocr_text)
    if len(clean_text(text)) < 5:
        text = clean_text(ocr_text)
    metadata: dict[str, Any] = {"ocr_method": "kimi_ocr"}
    if filename:
        metadata["filename"] = filename
    if external_id:
        metadata["external_id"] = external_id
    cleaned_supporting_text = _normalize_supporting_text(supporting_text)
    if cleaned_supporting_text:
        metadata["supporting_text"] = cleaned_supporting_text
    vision_model = get_model_for_task(NVIDIAModelTask.IMAGE_OCR)
    if vision_model:
        metadata["ocr_model"] = vision_model
    # OCR text may be noisy — add warning in context
    metadata["ocr_warning"] = "Input extracted via OCR — text may contain noise or errors."
    metadata["ocr_line_count"] = len([line for line in ocr_text.splitlines() if clean_text(line)])
    metadata["ocr_text_compacted"] = text != clean_text(ocr_text)
    combined_text = _compose_claim_source_input(
        primary_text=text,
        supporting_text=cleaned_supporting_text,
        source_label="Image text",
    )
    return await _pipeline_from_text(
        session,
        input_type="image",
        raw_text=combined_text,
        context_payload=metadata,
    )


async def enqueue_text_claim(
    session: AsyncSession,
    text: str,
    external_id: str | None = None,
) -> ClaimSubmissionResponseSchema:
    context = {"external_id": external_id} if external_id else {}
    live_status = _build_initial_live_status(
        input_type="text",
        raw_input=text,
        context_payload=context,
    )
    job = await _create_job(session, input_type="text", status="queued", live_status=live_status)
    return ClaimSubmissionResponseSchema(
        job=serialize_job(job, live_status=live_status),
        claim=None,
        cached=False,
    )


async def enqueue_url_claim(
    session: AsyncSession,
    url: str,
    external_id: str | None = None,
    supporting_text: str | None = None,
) -> ClaimSubmissionResponseSchema:
    cleaned_supporting_text = _normalize_supporting_text(supporting_text)
    context: dict[str, Any] = {"source_url": url}
    if external_id:
        context["external_id"] = external_id
    if cleaned_supporting_text:
        context["supporting_text"] = cleaned_supporting_text
    live_status = _build_initial_live_status(
        input_type="url",
        raw_input=cleaned_supporting_text or url,
        source_url=url,
        context_payload=context,
    )
    job = await _create_job(session, input_type="url", status="queued", live_status=live_status)
    return ClaimSubmissionResponseSchema(
        job=serialize_job(job, live_status=live_status),
        claim=None,
        cached=False,
    )


async def enqueue_image_claim(
    session: AsyncSession,
    *,
    filename: str | None = None,
    content_type: str | None = None,
    external_id: str | None = None,
    supporting_text: str | None = None,
) -> ClaimSubmissionResponseSchema:
    cleaned_supporting_text = _normalize_supporting_text(supporting_text)
    metadata: dict[str, Any] = {
        "filename": filename,
        "content_type": content_type,
        "ocr_method": "kimi_ocr",
        "ocr_model": get_model_for_task(NVIDIAModelTask.IMAGE_OCR),
        "ocr_warning": "Input extracted via OCR - text may contain noise or errors.",
    }
    if external_id:
        metadata["external_id"] = external_id
    if cleaned_supporting_text:
        metadata["supporting_text"] = cleaned_supporting_text
    live_status = _build_initial_live_status(
        input_type="image",
        raw_input=cleaned_supporting_text,
        context_payload=metadata,
    )
    job = await _create_job(session, input_type="image", status="queued", live_status=live_status)
    return ClaimSubmissionResponseSchema(
        job=serialize_job(job, live_status=live_status),
        claim=None,
        cached=False,
    )


async def _get_job_for_execution(session: AsyncSession, job_id: UUID | str) -> VerificationJob:
    parsed_id = _parse_uuid(job_id, code="INVALID_JOB_ID", message="Verification job ID must be a valid UUID.")
    job = await session.get(VerificationJob, parsed_id)
    if job is None:
        raise AppError(status_code=404, code="JOB_NOT_FOUND", message="Verification job not found.")
    return job


async def run_text_claim_job(
    job_id: UUID | str,
    *,
    text: str,
    external_id: str | None = None,
) -> None:
    async with SessionLocal() as session:
        job = await _get_job_for_execution(session, job_id)
        context = {"external_id": external_id} if external_id else {}
        try:
            await _pipeline_from_text(
                session,
                input_type="text",
                raw_text=text,
                context_payload=context,
                job=job,
                live_status=await _load_live_status(job.id),
            )
        except AppError:
            logger.info("verification_job_text_failed job_id=%s", job_id)
        except Exception:
            logger.exception("verification_job_text_failed job_id=%s", job_id)


async def run_url_claim_job(
    job_id: UUID | str,
    *,
    url: str,
    external_id: str | None = None,
    supporting_text: str | None = None,
) -> None:
    async with SessionLocal() as session:
        job = await _get_job_for_execution(session, job_id)
        cleaned_supporting_text = _normalize_supporting_text(supporting_text)
        live_status = await _load_live_status(job.id) or _build_initial_live_status(
            input_type="url",
            raw_input=cleaned_supporting_text or url,
            source_url=url,
            context_payload={
                "source_url": url,
                **({"supporting_text": cleaned_supporting_text} if cleaned_supporting_text else {}),
            },
        )
        try:
            await _ensure_job_processing(session, job, live_status=live_status)
            _set_live_stage(live_status, "queued", message="Fetching text from the submitted URL...")
            _append_live_event(
                live_status,
                key="url_fetch_started",
                label="URL fetch started",
                status="running",
                summary="Downloading and extracting article text from the submitted URL.",
                metadata={"url": url},
            )
            await _publish_job_status(job, live_status)
            text, metadata = await _fetch_url_text(url)
            if external_id:
                metadata["external_id"] = external_id
            if cleaned_supporting_text:
                metadata["supporting_text"] = cleaned_supporting_text
            combined_text = _compose_claim_source_input(
                primary_text=text,
                supporting_text=cleaned_supporting_text,
                source_label="Page text",
            )
            _append_live_event(
                live_status,
                key="url_fetch_complete",
                label="URL fetch complete",
                status="completed",
                summary="Fetched page content and prepared it for claim extraction.",
                metadata={"fetched_url": metadata.get("fetched_url"), "status_code": metadata.get("status_code")},
            )
            _merge_live_details(
                live_status,
                "input",
                {
                    "source_url": url,
                    "fetched_url": metadata.get("fetched_url"),
                    "status_code": metadata.get("status_code"),
                },
            )
            live_status["raw_input"] = _preview_text(combined_text)
        except AppError as exc:
            _set_live_stage(live_status, "failed", message=exc.message)
            _append_live_event(
                live_status,
                key="url_fetch_failed",
                label="URL fetch failed",
                status="failed",
                summary=exc.message,
                metadata={"error_code": exc.code},
            )
            await _fail_job(session, job, code=exc.code, message=exc.message, live_status=live_status)
            return
        except Exception:
            logger.exception("verification_job_url_failed job_id=%s", job_id)
            _set_live_stage(live_status, "failed", message="Claim verification pipeline failed unexpectedly.")
            _append_live_event(
                live_status,
                key="url_fetch_failed",
                label="URL verification failed",
                status="failed",
                summary="Claim verification pipeline failed unexpectedly.",
                metadata={"error_code": "PIPELINE_FAILED"},
            )
            await _fail_job(
                session,
                job,
                code="PIPELINE_FAILED",
                message="Claim verification pipeline failed unexpectedly.",
                live_status=live_status,
            )
            return

        try:
            await _pipeline_from_text(
                session,
                input_type="url",
                raw_text=combined_text,
                source_url=url,
                context_payload=metadata,
                job=job,
                live_status=live_status,
            )
        except AppError:
            logger.info("verification_job_url_failed job_id=%s", job_id)
        except Exception:
            logger.exception("verification_job_url_failed job_id=%s", job_id)


async def run_image_claim_job(
    job_id: UUID | str,
    *,
    image_bytes: bytes,
    filename: str | None = None,
    content_type: str | None = None,
    external_id: str | None = None,
    supporting_text: str | None = None,
) -> None:
    async with SessionLocal() as session:
        job = await _get_job_for_execution(session, job_id)
        cleaned_supporting_text = _normalize_supporting_text(supporting_text)
        live_status = await _load_live_status(job.id) or _build_initial_live_status(
            input_type="image",
            raw_input=cleaned_supporting_text,
            context_payload={
                "filename": filename,
                "content_type": content_type,
                "ocr_method": "kimi_ocr",
                "ocr_model": get_model_for_task(NVIDIAModelTask.IMAGE_OCR),
                **({"supporting_text": cleaned_supporting_text} if cleaned_supporting_text else {}),
            },
        )
        try:
            await _ensure_job_processing(session, job, live_status=live_status)
            _set_live_stage(live_status, "queued", message="Extracting text from the uploaded image...")
            _append_live_event(
                live_status,
                key="ocr_started",
                label="OCR started",
                status="running",
                summary="Running Kimi OCR to isolate claim-like text from the uploaded image.",
                metadata={"filename": filename, "content_type": content_type},
            )
            await _publish_job_status(job, live_status)
            ocr_text = await extract_text_from_image(image_bytes, mime_type=content_type)
            text = prepare_ocr_text_for_claim_extraction(ocr_text)
            if len(clean_text(text)) < 5:
                text = clean_text(ocr_text)
            metadata: dict[str, Any] = {
                "ocr_method": "kimi_ocr",
                "ocr_model": get_model_for_task(NVIDIAModelTask.IMAGE_OCR),
                "ocr_warning": "Input extracted via OCR - text may contain noise or errors.",
                "ocr_line_count": len([line for line in ocr_text.splitlines() if clean_text(line)]),
                "ocr_text_compacted": text != clean_text(ocr_text),
                "filename": filename,
                "content_type": content_type,
            }
            if external_id:
                metadata["external_id"] = external_id
            if cleaned_supporting_text:
                metadata["supporting_text"] = cleaned_supporting_text
            combined_text = _compose_claim_source_input(
                primary_text=text,
                supporting_text=cleaned_supporting_text,
                source_label="Image text",
            )
            live_status["raw_input"] = _preview_text(combined_text)
            _merge_live_details(
                live_status,
                "input",
                {
                    "filename": filename,
                    "content_type": content_type,
                    "ocr_line_count": metadata["ocr_line_count"],
                    "ocr_text_compacted": metadata["ocr_text_compacted"],
                },
            )
            _set_live_warning_messages(live_status, [str(metadata["ocr_warning"])])
            _append_live_event(
                live_status,
                key="ocr_complete",
                label="OCR complete",
                status="completed",
                summary="Extracted the main image text and prepared it for claim verification.",
                metadata={
                    "ocr_line_count": metadata["ocr_line_count"],
                    "ocr_text_compacted": metadata["ocr_text_compacted"],
                },
            )
        except AppError as exc:
            _set_live_stage(live_status, "failed", message=exc.message)
            _append_live_event(
                live_status,
                key="ocr_failed",
                label="OCR failed",
                status="failed",
                summary=exc.message,
                metadata={"error_code": exc.code},
            )
            await _fail_job(session, job, code=exc.code, message=exc.message, live_status=live_status)
            return
        except Exception:
            logger.exception("verification_job_image_failed job_id=%s", job_id)
            _set_live_stage(live_status, "failed", message="Claim verification pipeline failed unexpectedly.")
            _append_live_event(
                live_status,
                key="ocr_failed",
                label="Image verification failed",
                status="failed",
                summary="Claim verification pipeline failed unexpectedly.",
                metadata={"error_code": "PIPELINE_FAILED"},
            )
            await _fail_job(
                session,
                job,
                code="PIPELINE_FAILED",
                message="Claim verification pipeline failed unexpectedly.",
                live_status=live_status,
            )
            return

        try:
            await _pipeline_from_text(
                session,
                input_type="image",
                raw_text=combined_text,
                context_payload=metadata,
                job=job,
                live_status=live_status,
            )
        except AppError:
            logger.info("verification_job_image_failed job_id=%s", job_id)
        except Exception:
            logger.exception("verification_job_image_failed job_id=%s", job_id)


# ── Read-only pipeline operations ─────────────────────────────────────────────

async def get_claim_by_id(session: AsyncSession, claim_id: UUID | str) -> ClaimResponseSchema | None:
    claim = await _load_claim(session, claim_id)
    return serialize_claim(claim) if claim else None


async def list_claims(
    session: AsyncSession,
    *,
    verdict: str | None = None,
    language: str | None = None,
    review_status: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[ClaimResponseSchema], int]:
    filters = []
    if verdict:
        filters.append(Claim.verdict == verdict)
    if language:
        filters.append(Claim.language == language)
    if review_status:
        filters.append(Claim.review_status == review_status)

    total_statement = select(func.count()).select_from(Claim)
    if filters:
        total_statement = total_statement.where(*filters)
    total = await session.scalar(total_statement) or 0

    statement = (
        select(Claim)
        .options(selectinload(Claim.evidence_links).selectinload(ClaimEvidenceLink.source))
        .order_by(Claim.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    if filters:
        statement = statement.where(*filters)
    claims = (await session.execute(statement)).scalars().all()
    return [serialize_claim(claim) for claim in claims], int(total)


async def update_review_status(
    session: AsyncSession,
    claim_id: UUID | str,
    review_status: str,
    reviewer: str,
) -> ClaimResponseSchema | None:
    parsed_id = _parse_uuid(claim_id, code="INVALID_CLAIM_ID", message="Claim ID must be a valid UUID.")
    claim = await session.get(Claim, parsed_id)
    if claim is None:
        return None
    claim.review_status = review_status
    session.add(
        AuditLog(
            actor=reviewer,
            action="review_status_updated",
            entity_type="claim",
            entity_id=str(claim.id),
            details={"review_status": review_status},
        )
    )
    await session.commit()
    reloaded = await _load_claim(session, claim.id)
    return serialize_claim(reloaded) if reloaded else None


async def get_verification_job(
    session: AsyncSession,
    job_id: UUID | str,
) -> VerificationJobSchema | None:
    parsed_id = _parse_uuid(job_id, code="INVALID_JOB_ID", message="Job ID must be a valid UUID.")
    cached = await cache_service.get_job_status(str(job_id))
    job = await session.get(VerificationJob, parsed_id)
    if job is not None:
        base_payload = serialize_job(job).model_dump(mode="json")
        if isinstance(cached, dict):
            cached_updated_at = str(cached.get("updated_at") or "")
            base_updated_at = str(base_payload.get("updated_at") or "")
            if cached_updated_at and cached_updated_at >= base_updated_at:
                for key in (
                    "claim_id",
                    "status",
                    "normalized_hash",
                    "cached_hit",
                    "error_code",
                    "error_message",
                    "completed_at",
                ):
                    if key in cached:
                        base_payload[key] = cached[key]
            if "live_status" in cached:
                base_payload["live_status"] = cached["live_status"]
        live_status_payload = base_payload.get("live_status")
        if isinstance(live_status_payload, dict):
            if base_payload.get("status") == "completed" and live_status_payload.get("stage_key") != "completed":
                live_status_payload.update(_live_stage_payload("completed"))
                live_status_payload["updated_at"] = str(base_payload.get("completed_at") or base_payload.get("updated_at"))
            elif base_payload.get("status") == "failed" and live_status_payload.get("stage_key") != "failed":
                live_status_payload.update(_live_stage_payload("failed"))
                live_status_payload["updated_at"] = str(base_payload.get("completed_at") or base_payload.get("updated_at"))
        return VerificationJobSchema.model_validate(base_payload, strict=False)
    if isinstance(cached, dict):
        live_status_payload = cached.get("live_status")
        if isinstance(live_status_payload, dict):
            if cached.get("status") == "completed" and live_status_payload.get("stage_key") != "completed":
                live_status_payload.update(_live_stage_payload("completed"))
                live_status_payload["updated_at"] = str(cached.get("completed_at") or cached.get("updated_at"))
            elif cached.get("status") == "failed" and live_status_payload.get("stage_key") != "failed":
                live_status_payload.update(_live_stage_payload("failed"))
                live_status_payload["updated_at"] = str(cached.get("completed_at") or cached.get("updated_at"))
        return VerificationJobSchema.model_validate(cached, strict=False)
    return None
