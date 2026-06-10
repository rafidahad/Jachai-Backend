from __future__ import annotations

import re
from typing import Annotated, Literal

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import Field, HttpUrl, TypeAdapter, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.rate_limit import internal_rate_limit
from app.core.security import verify_internal_api_key
from app.db.session import get_db
from app.schemas.claim_schema import ClaimSubmissionResponseSchema
from app.schemas.common import StrictBaseModel
from app.services.claim_pipeline import (
    enqueue_text_claim,
    enqueue_url_claim,
    run_text_claim_job,
    run_url_claim_job,
)
from app.services.text_cleaning_service import clean_text
from app.utils.errors import AppError

router = APIRouter(
    prefix="/webhooks",
    tags=["webhooks"],
    dependencies=[Depends(verify_internal_api_key), Depends(internal_rate_limit)],
)

URL_PATTERN = re.compile(r"(?P<url>https?://[^\s<>\"]+|www\.[^\s<>\"]+)", re.IGNORECASE)
HTTP_URL_ADAPTER = TypeAdapter(HttpUrl)


class InboundMessageSchema(StrictBaseModel):
    message_type: Literal["text", "url"] | None = None
    message_text: str | None = None
    message_url: str | None = None
    external_id: str | None = None
    supporting_text: str | None = None
    platform: str | None = None
    sender_id: str | None = None
    message_id: str | None = None
    text: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


def _normalize_optional_text(value: str | None) -> str | None:
    cleaned = clean_text(value or "")
    return cleaned or None


def _build_external_id(payload: InboundMessageSchema) -> str | None:
    if payload.external_id:
        return payload.external_id
    parts = [payload.platform, payload.sender_id, payload.message_id]
    normalized = [clean_text(part) for part in parts if part and clean_text(part)]
    return ":".join(normalized) if normalized else None


def _validate_message_url(value: str) -> str:
    candidate = clean_text(value)
    if candidate.lower().startswith("www."):
        candidate = f"https://{candidate}"
    try:
        return str(HTTP_URL_ADAPTER.validate_python(candidate))
    except ValidationError as exc:
        raise AppError(
            status_code=422,
            code="INVALID_WEBHOOK_PAYLOAD",
            message="Webhook payload must include a valid absolute URL.",
        ) from exc


def _extract_url_and_supporting_text(text: str) -> tuple[str | None, str | None]:
    cleaned = clean_text(text)
    if not cleaned:
        return None, None

    match = URL_PATTERN.search(cleaned)
    if not match:
        return None, cleaned

    detected_url = _validate_message_url(match.group("url").rstrip(".,!?:;)"))
    supporting_text = clean_text(f"{cleaned[:match.start()]} {cleaned[match.end():]}")
    return detected_url, (supporting_text or None)


def _normalize_inbound_message(payload: InboundMessageSchema) -> dict[str, str | None]:
    external_id = _build_external_id(payload)

    if payload.message_type == "text":
        message_text = _normalize_optional_text(payload.message_text)
        if message_text:
            return {
                "message_type": "text",
                "message_text": message_text,
                "message_url": None,
                "supporting_text": None,
                "external_id": external_id,
            }
        raise AppError(
            status_code=422,
            code="INVALID_WEBHOOK_PAYLOAD",
            message="Webhook payload must include message_text for text inputs.",
        )

    if payload.message_type == "url":
        message_url = payload.message_url
        supporting_text = _normalize_optional_text(payload.supporting_text)
        if not message_url and payload.message_text:
            message_url, extracted_supporting_text = _extract_url_and_supporting_text(
                payload.message_text
            )
            supporting_text = supporting_text or extracted_supporting_text
        elif not supporting_text and payload.message_text:
            supporting_text = _normalize_optional_text(payload.message_text)
        if message_url:
            return {
                "message_type": "url",
                "message_text": None,
                "message_url": _validate_message_url(message_url),
                "supporting_text": supporting_text,
                "external_id": external_id,
            }
        raise AppError(
            status_code=422,
            code="INVALID_WEBHOOK_PAYLOAD",
            message="Webhook payload must include message_url for URL inputs.",
        )

    raw_text = _normalize_optional_text(payload.text or payload.message_text)
    if raw_text:
        detected_url, supporting_text = _extract_url_and_supporting_text(raw_text)
        if detected_url:
            return {
                "message_type": "url",
                "message_text": None,
                "message_url": detected_url,
                "supporting_text": supporting_text,
                "external_id": external_id,
            }
        return {
            "message_type": "text",
            "message_text": raw_text,
            "message_url": None,
            "supporting_text": None,
            "external_id": external_id,
        }

    raise AppError(
        status_code=422,
        code="INVALID_WEBHOOK_PAYLOAD",
        message="Webhook payload must include text, message_text, or message_url.",
    )


@router.post("/inbound-message", response_model=ClaimSubmissionResponseSchema)
async def inbound_message(
    payload: InboundMessageSchema,
    background_tasks: BackgroundTasks,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> ClaimSubmissionResponseSchema:
    normalized = _normalize_inbound_message(payload)

    if normalized["message_type"] == "text" and normalized["message_text"]:
        response = await enqueue_text_claim(
            session,
            normalized["message_text"],
            normalized["external_id"],
        )
        background_tasks.add_task(
            run_text_claim_job,
            response.job.id,
            text=normalized["message_text"],
            external_id=normalized["external_id"],
        )
        return response

    if normalized["message_type"] == "url" and normalized["message_url"]:
        response = await enqueue_url_claim(
            session,
            normalized["message_url"],
            normalized["external_id"],
            supporting_text=normalized["supporting_text"],
        )
        background_tasks.add_task(
            run_url_claim_job,
            response.job.id,
            url=normalized["message_url"],
            external_id=normalized["external_id"],
            supporting_text=normalized["supporting_text"],
        )
        return response

    raise AppError(
        status_code=422,
        code="INVALID_WEBHOOK_PAYLOAD",
        message="Webhook payload must include message_text for text or message_url for URL inputs.",
    )
