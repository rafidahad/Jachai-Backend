from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.rate_limit import internal_rate_limit
from app.core.security import verify_internal_api_key
from app.db.session import get_db
from app.schemas.claim_schema import (
    ClaimSubmissionResponseSchema,
    VerifyRequest,
    VerificationJobSchema,
)
from app.schemas.common import StrictBaseModel
from app.services.claim_pipeline import ClaimPipeline, get_claim_by_id
from app.utils.errors import AppError

router = APIRouter(
    prefix="/webhooks",
    tags=["webhooks"],
    dependencies=[Depends(verify_internal_api_key), Depends(internal_rate_limit)],
)


class InboundMessageSchema(StrictBaseModel):
    message_type: Literal["text", "url"]
    message_text: str | None = None
    message_url: str | None = None
    external_id: str | None = None


@router.post("/inbound-message", response_model=ClaimSubmissionResponseSchema)
async def inbound_message(
    payload: InboundMessageSchema,
    session: AsyncSession = Depends(get_db),
) -> ClaimSubmissionResponseSchema:
    if payload.message_type == "text" and payload.message_text:
        content = payload.message_text
        input_type = "text"
    elif payload.message_type == "url" and payload.message_url:
        content = payload.message_url
        input_type = "url"
    else:
        raise AppError(
            status_code=422,
            code="INVALID_WEBHOOK_PAYLOAD",
            message="Webhook payload must include message_text for text or message_url for url.",
        )

    verify_request = VerifyRequest(
        input_type=input_type,
        content=content,
    )

    response = await ClaimPipeline.verify_claim(session, verify_request)

    claim_schema = None
    if response.claim_id:
        claim_schema = await get_claim_by_id(session, response.claim_id)

    job_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    cached_hit = False
    if response.debug and isinstance(response.debug, dict):
        cached_hit = bool(response.debug.get("cached", False))

    job_schema = VerificationJobSchema(
        id=job_id,
        claim_id=uuid.UUID(response.claim_id) if response.claim_id else None,
        input_type=input_type,
        status="completed",
        normalized_hash=claim_schema.normalized_hash if claim_schema else None,
        cached_hit=cached_hit,
        created_at=now,
        updated_at=now,
        completed_at=now,
    )

    return ClaimSubmissionResponseSchema(
        job=job_schema,
        claim=claim_schema,
        cached=cached_hit,
    )

