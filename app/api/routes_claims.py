from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.rate_limit import claim_submission_rate_limit
from app.core.security import verify_internal_api_key
from app.db.session import get_db
from app.schemas.claim_schema import (
    ClaimListResponseSchema,
    ClaimLookupResponseSchema,
    ClaimShareSummarySchema,
    ClaimSubmissionResponseSchema,
    ClaimTextRequest,
    ClaimURLRequest,
    ReviewStatusUpdateRequest,
)
from app.services.claim_pipeline import (
    enqueue_image_claim,
    enqueue_text_claim,
    enqueue_url_claim,
    get_claim_by_id,
    list_claims,
    run_image_claim_job,
    run_text_claim_job,
    run_url_claim_job,
    update_review_status,
)
from app.utils.errors import AppError

router = APIRouter(prefix="/claims", tags=["claims"])
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}


@router.post("/text", response_model=ClaimSubmissionResponseSchema, dependencies=[Depends(claim_submission_rate_limit)])
async def submit_text_claim(
    payload: ClaimTextRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db),
) -> ClaimSubmissionResponseSchema:
    response = await enqueue_text_claim(session, payload.text, payload.external_id)
    background_tasks.add_task(
        run_text_claim_job,
        response.job.id,
        text=payload.text,
        external_id=payload.external_id,
    )
    return response


@router.post("/image", response_model=ClaimSubmissionResponseSchema, dependencies=[Depends(claim_submission_rate_limit)])
async def submit_image_claim(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    external_id: Annotated[str | None, Form()] = None,
    session: AsyncSession = Depends(get_db),
) -> ClaimSubmissionResponseSchema:
    if not image.content_type or image.content_type not in ALLOWED_IMAGE_TYPES:
        raise AppError(
            status_code=422,
            code="INVALID_IMAGE",
            message="Image must be PNG, JPG, JPEG, or WEBP.",
        )
    image_bytes = await image.read()
    max_bytes = settings.max_image_size_mb * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise AppError(
            status_code=422,
            code="IMAGE_TOO_LARGE",
            message=f"Image must be {settings.max_image_size_mb} MB or smaller.",
        )
    response = await enqueue_image_claim(
        session,
        filename=image.filename,
        content_type=image.content_type,
        external_id=external_id,
    )
    background_tasks.add_task(
        run_image_claim_job,
        response.job.id,
        image_bytes=image_bytes,
        filename=image.filename,
        content_type=image.content_type,
        external_id=external_id,
    )
    return response


@router.post("/url", response_model=ClaimSubmissionResponseSchema, dependencies=[Depends(claim_submission_rate_limit)])
async def submit_url_claim(
    payload: ClaimURLRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db),
) -> ClaimSubmissionResponseSchema:
    response = await enqueue_url_claim(session, str(payload.url), payload.external_id)
    background_tasks.add_task(
        run_url_claim_job,
        response.job.id,
        url=str(payload.url),
        external_id=payload.external_id,
    )
    return response


@router.get("/{claim_id}", response_model=ClaimLookupResponseSchema)
async def get_claim_endpoint(
    claim_id: str,
    session: AsyncSession = Depends(get_db),
) -> ClaimLookupResponseSchema:
    claim = await get_claim_by_id(session, claim_id)
    if claim is None:
        raise AppError(status_code=404, code="CLAIM_NOT_FOUND", message="Claim not found.")
    return ClaimLookupResponseSchema(claim=claim)


@router.get("/{claim_id}/share-summary", response_model=ClaimShareSummarySchema)
async def claim_share_summary(
    claim_id: str,
    session: AsyncSession = Depends(get_db),
) -> ClaimShareSummarySchema:
    claim = await get_claim_by_id(session, claim_id)
    if claim is None:
        raise AppError(status_code=404, code="CLAIM_NOT_FOUND", message="Claim not found.")
    return ClaimShareSummarySchema(
        claim_id=claim.id,
        summary=claim.share_summary,
        verdict=claim.verdict,
        language=claim.language,
    )


@router.patch(
    "/{claim_id}/review-status",
    response_model=ClaimLookupResponseSchema,
    dependencies=[Depends(verify_internal_api_key)],
)
async def patch_review_status(
    claim_id: str,
    payload: ReviewStatusUpdateRequest,
    session: AsyncSession = Depends(get_db),
) -> ClaimLookupResponseSchema:
    claim = await update_review_status(session, claim_id, payload.review_status, payload.reviewer)
    if claim is None:
        raise AppError(status_code=404, code="CLAIM_NOT_FOUND", message="Claim not found.")
    return ClaimLookupResponseSchema(claim=claim)


@router.get("", response_model=ClaimListResponseSchema)
async def list_claims_endpoint(
    verdict: str | None = None,
    language: str | None = None,
    review_status: str | None = None,
    limit: int = 20,
    offset: int = 0,
    session: AsyncSession = Depends(get_db),
) -> ClaimListResponseSchema:
    items, total = await list_claims(
        session,
        verdict=verdict,
        language=language,
        review_status=review_status,
        limit=limit,
        offset=offset,
    )
    return ClaimListResponseSchema(items=items, total=total, limit=limit, offset=offset)
