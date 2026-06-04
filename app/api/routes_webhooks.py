from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.rate_limit import internal_rate_limit
from app.core.security import verify_internal_api_key
from app.db.session import get_db
from app.schemas.claim_schema import ClaimSubmissionResponseSchema
from app.schemas.common import StrictBaseModel
from app.services.claim_pipeline import process_text_claim, process_url_claim
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
        return await process_text_claim(session, payload.message_text, payload.external_id)
    if payload.message_type == "url" and payload.message_url:
        return await process_url_claim(session, payload.message_url, payload.external_id)
    raise AppError(
        status_code=422,
        code="INVALID_WEBHOOK_PAYLOAD",
        message="Webhook payload must include message_text for text or message_url for url.",
    )
