from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.rate_limit import claim_submission_rate_limit
from app.core.security import verify_internal_api_key
from app.db.session import get_db
from app.schemas.claim_schema import (
    ClaimListResponseSchema,
    ClaimLookupResponseSchema,
    ClaimShareSummarySchema,
    ReviewStatusUpdateRequest,
    VerifyRequest,
    VerifyResponse,
)
from app.services.claim_pipeline import (
    ClaimPipeline,
    get_claim_by_id,
    list_claims,
    update_review_status,
)
from app.utils.errors import AppError

router = APIRouter(prefix="/claims", tags=["claims"])
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}


@router.post("/verify", response_model=VerifyResponse)
async def verify_claim(
    payload: VerifyRequest,
    session: AsyncSession = Depends(get_db),
) -> VerifyResponse:
    return await ClaimPipeline.verify_claim(session, payload)


@router.post("/verify/image", response_model=VerifyResponse, dependencies=[Depends(claim_submission_rate_limit)])
async def verify_image_claim(
    image: UploadFile = File(...),
    session: AsyncSession = Depends(get_db),
) -> VerifyResponse:
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
    from app.services.ocr_service import extract_text_from_image_with_fallback
    text, ocr_method = await extract_text_from_image_with_fallback(image_bytes, mime_type=image.content_type)
    payload = VerifyRequest(
        input_type="image_ocr",
        content=text,
        language="auto",
    )
    return await ClaimPipeline.verify_claim(session, payload)


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
