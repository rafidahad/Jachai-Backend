from __future__ import annotations

from typing import Any
from uuid import UUID

import httpx
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.models.audit_log import AuditLog
from app.models.claim import Claim, ClaimEvidenceLink
from app.models.verification_job import VerificationJob
from app.schemas.claim_schema import ClaimResponseSchema, ClaimSubmissionResponseSchema, VerificationJobSchema
from app.schemas.verdict_schema import EvidenceSnippetSchema
from app.services.ai_model_router import NVIDIAModelTask, get_model_for_task
from app.services.cache_service import cache_service
from app.services.cluster_service import get_or_create_cluster
from app.services.embedding_service import embed_text
from app.services.language_service import detect_language
from app.services.nvidia_llm_service import generate_verdict
from app.services.ocr_service import extract_text_from_image_with_fallback
from app.services.pii_service import mask_pii
from app.services.retrieval_service import retrieve_evidence
from app.services.text_cleaning_service import clean_text, text_from_html
from app.utils.errors import AppError
from app.utils.hashing import normalized_hash
from app.utils.time import utc_now


def _parse_uuid(value: UUID | str, *, code: str, message: str) -> UUID:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except ValueError as exc:
        raise AppError(status_code=422, code=code, message=message) from exc


def serialize_job(job: VerificationJob) -> VerificationJobSchema:
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
    )


def serialize_claim(claim: Claim) -> ClaimResponseSchema:
    evidence = [
        EvidenceSnippetSchema(
            source_id=link.source.id,
            title=link.source.title,
            url=link.source.url,
            publisher=link.source.publisher,
            language=link.source.language,
            source_type=link.source.source_type,
            snippet=link.source.snippet,
            similarity_score=link.similarity_score,
        )
        for link in sorted(claim.evidence_links, key=lambda item: item.rank)
        if link.source is not None
    ]
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
        review_status=claim.review_status,
        verdict=claim.verdict,
        confidence=claim.confidence,
        explanation=claim.explanation,
        reasoning=claim.reasoning,
        share_summary=claim.share_summary,
        created_at=claim.created_at,
        updated_at=claim.updated_at,
        evidence=evidence,
        context_payload=claim.context_payload,
    )


async def _load_claim(session: AsyncSession, claim_id: UUID | str) -> Claim | None:
    parsed_id = _parse_uuid(claim_id, code="INVALID_CLAIM_ID", message="Claim ID must be a valid UUID.")
    statement = (
        select(Claim)
        .options(selectinload(Claim.evidence_links).selectinload(ClaimEvidenceLink.source))
        .where(Claim.id == parsed_id)
    )
    return await session.scalar(statement)


async def _create_job(session: AsyncSession, input_type: str) -> VerificationJob:
    job = VerificationJob(input_type=input_type, status="processing", cached_hit=False)
    session.add(job)
    await session.commit()
    await session.refresh(job)
    await cache_service.set_job_status(str(job.id), serialize_job(job).model_dump(mode="json"))
    return job


async def _complete_job(
    session: AsyncSession,
    job: VerificationJob,
    *,
    claim_id: UUID | None,
    cached_hit: bool,
    status: str = "completed",
) -> None:
    job.claim_id = claim_id
    job.cached_hit = cached_hit
    job.status = status
    job.completed_at = utc_now()
    await session.commit()
    await session.refresh(job)
    await cache_service.set_job_status(str(job.id), serialize_job(job).model_dump(mode="json"))


async def _fail_job(
    session: AsyncSession,
    job: VerificationJob,
    *,
    code: str,
    message: str,
) -> None:
    job.status = "failed"
    job.error_code = code
    job.error_message = message
    job.completed_at = utc_now()
    await session.commit()
    await session.refresh(job)
    await cache_service.set_job_status(str(job.id), serialize_job(job).model_dump(mode="json"))


async def _fetch_url_text(url: str) -> tuple[str, dict[str, Any]]:
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=settings.request_timeout_seconds,
            headers={"User-Agent": "JachAI/0.1"},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise AppError(status_code=422, code="URL_FETCH_FAILED", message=f"Could not fetch URL: {exc}") from exc

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
) -> ClaimSubmissionResponseSchema:
    await _complete_job(session, job, claim_id=claim.id, cached_hit=cached_hit)
    serialized = serialize_claim(claim)
    await cache_service.set_duplicate_claim_id(claim.normalized_hash, str(claim.id))
    await cache_service.set_claim_result(
        claim.normalized_hash,
        {"claim_id": str(claim.id), "job_id": str(job.id), "cached": cached_hit},
    )
    return ClaimSubmissionResponseSchema(job=serialize_job(job), claim=serialized, cached=cached_hit)


async def _pipeline_from_text(
    session: AsyncSession,
    *,
    input_type: str,
    raw_text: str,
    source_url: str | None = None,
    context_payload: dict[str, Any] | None = None,
) -> ClaimSubmissionResponseSchema:
    job = await _create_job(session, input_type=input_type)
    try:
        claim_context = dict(context_payload or {})
        cleaned_text = clean_text(raw_text)
        if len(cleaned_text) < 5:
            raise AppError(status_code=422, code="INPUT_TOO_SHORT", message="Claim text is too short.")
        masked_text = mask_pii(cleaned_text)
        language = detect_language(masked_text)
        hash_value = normalized_hash(masked_text)

        job.normalized_hash = hash_value
        await session.commit()
        await session.refresh(job)
        await cache_service.set_job_status(str(job.id), serialize_job(job).model_dump(mode="json"))

        duplicate_claim_id = await cache_service.get_duplicate_claim_id(hash_value)
        if duplicate_claim_id:
            existing = await _load_claim(session, duplicate_claim_id)
            if existing:
                return await _return_existing_claim(session, job, existing, cached_hit=True)

        cached_payload = await cache_service.get_claim_result(hash_value)
        if cached_payload and cached_payload.get("claim_id"):
            existing = await _load_claim(session, cached_payload["claim_id"])
            if existing:
                return await _return_existing_claim(session, job, existing, cached_hit=True)

        existing_db_claim = await session.scalar(
            select(Claim.id).where(Claim.normalized_hash == hash_value).limit(1)
        )
        if existing_db_claim:
            existing = await _load_claim(session, existing_db_claim)
            if existing:
                return await _return_existing_claim(session, job, existing, cached_hit=True)

        embedding = await embed_text(masked_text)
        retrieved = await retrieve_evidence(session, embedding, top_k=settings.evidence_top_k)
        evidence_payload = [
            {
                "source_id": str(source.id),
                "title": source.title,
                "url": source.url,
                "publisher": source.publisher,
                "language": source.language,
                "source_type": source.source_type,
                "snippet": source.snippet,
                "similarity_score": score,
            }
            for source, score in retrieved
        ]
        verdict = await generate_verdict(masked_text, language, evidence_payload)
        reasoning_model = get_model_for_task(NVIDIAModelTask.CLAIM_REASONING)
        if reasoning_model:
            claim_context["nvidia_reasoning_model"] = reasoning_model
        cluster = await get_or_create_cluster(
            session,
            topic_hash=hash_value,
            title=cleaned_text[:120],
            language=language,
            summary=verdict.summary,
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
            reasoning=verdict.reasoning,
            share_summary=verdict.summary,
            review_status="pending",
            context_payload=claim_context,
        )
        session.add(claim)
        await session.flush()

        if cluster.representative_claim_id is None:
            cluster.representative_claim_id = claim.id

        source_score_lookup = {str(source.id): score for source, score in retrieved}
        source_lookup = {str(source.id): source for source, score in retrieved}
        ordered_source_ids = [str(source_id) for source_id in verdict.source_ids if str(source_id) in source_lookup]
        fallback_ids = [source_id for source_id in source_lookup.keys() if source_id not in ordered_source_ids]
        for index, source_id in enumerate(ordered_source_ids + fallback_ids, start=1):
            session.add(
                ClaimEvidenceLink(
                    claim_id=claim.id,
                    source_id=UUID(source_id),
                    rank=index,
                    similarity_score=float(source_score_lookup[source_id]),
                )
            )

        session.add(
            AuditLog(
                actor="pipeline",
                action="claim_verified",
                entity_type="claim",
                entity_id=str(claim.id),
                details={"normalized_hash": hash_value, "language": language, "input_type": input_type},
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

        await _complete_job(session, job, claim_id=persisted_claim.id, cached_hit=False)
        await cache_service.set_duplicate_claim_id(hash_value, str(persisted_claim.id))
        await cache_service.set_claim_result(
            hash_value,
            {"claim_id": str(persisted_claim.id), "job_id": str(job.id), "cached": False},
        )
        return ClaimSubmissionResponseSchema(
            job=serialize_job(job),
            claim=serialize_claim(persisted_claim),
            cached=False,
        )
    except AppError as exc:
        await _fail_job(session, job, code=exc.code, message=exc.message)
        raise
    except Exception:
        await _fail_job(
            session,
            job,
            code="PIPELINE_FAILED",
            message="Claim verification pipeline failed unexpectedly.",
        )
        raise


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
) -> ClaimSubmissionResponseSchema:
    text, metadata = await _fetch_url_text(url)
    if external_id:
        metadata["external_id"] = external_id
    return await _pipeline_from_text(
        session,
        input_type="url",
        raw_text=text,
        source_url=url,
        context_payload=metadata,
    )


async def process_image_claim(
    session: AsyncSession,
    image_bytes: bytes,
    filename: str | None = None,
    content_type: str | None = None,
    external_id: str | None = None,
) -> ClaimSubmissionResponseSchema:
    text, ocr_method = await extract_text_from_image_with_fallback(image_bytes, mime_type=content_type)
    metadata: dict[str, Any] = {"ocr_method": ocr_method}
    if filename:
        metadata["filename"] = filename
    if external_id:
        metadata["external_id"] = external_id
    if ocr_method == "nvidia_vision_fallback":
        vision_model = get_model_for_task(NVIDIAModelTask.IMAGE_OCR_FALLBACK)
        if vision_model:
            metadata["nvidia_vision_model"] = vision_model
    return await _pipeline_from_text(
        session,
        input_type="image",
        raw_text=text,
        context_payload=metadata,
    )


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
    job = await session.get(VerificationJob, parsed_id)
    if job is not None:
        return serialize_job(job)
    cached = await cache_service.get_job_status(str(job_id))
    return VerificationJobSchema.model_validate(cached, strict=False) if cached else None
