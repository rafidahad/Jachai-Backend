from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field, HttpUrl, StrictBool, StrictInt, StrictStr

from app.schemas.common import StrictBaseModel
from app.schemas.verdict_schema import EvidenceSnippetSchema, LanguageLabel, VerdictLabel

InputType = Literal["text", "image", "url"]
ReviewStatus = Literal["pending", "reviewed", "flagged"]


class ClaimTextRequest(StrictBaseModel):
    text: StrictStr = Field(min_length=5, max_length=10000)
    external_id: StrictStr | None = Field(default=None, max_length=128)


class ClaimURLRequest(StrictBaseModel):
    url: HttpUrl
    external_id: StrictStr | None = Field(default=None, max_length=128)


class ReviewStatusUpdateRequest(StrictBaseModel):
    review_status: ReviewStatus
    reviewer: StrictStr = Field(min_length=1, max_length=64)


class ClaimResponseSchema(StrictBaseModel):
    id: UUID
    cluster_id: UUID | None = None
    input_type: InputType
    source_url: StrictStr | None = None
    raw_text: StrictStr
    cleaned_text: StrictStr
    masked_text: StrictStr
    normalized_hash: StrictStr
    language: LanguageLabel
    review_status: ReviewStatus
    verdict: VerdictLabel
    confidence: float
    explanation: StrictStr
    reasoning: StrictStr
    share_summary: StrictStr
    created_at: datetime
    updated_at: datetime
    evidence: list[EvidenceSnippetSchema] = Field(default_factory=list)
    context_payload: dict[str, object] = Field(default_factory=dict)


class VerificationJobSchema(StrictBaseModel):
    id: UUID
    claim_id: UUID | None = None
    input_type: InputType
    status: Literal["queued", "processing", "completed", "failed"]
    normalized_hash: StrictStr | None = None
    cached_hit: StrictBool
    error_code: StrictStr | None = None
    error_message: StrictStr | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None


class ClaimSubmissionResponseSchema(StrictBaseModel):
    job: VerificationJobSchema
    claim: ClaimResponseSchema | None = None
    cached: StrictBool = False


class ClaimListResponseSchema(StrictBaseModel):
    items: list[ClaimResponseSchema]
    total: StrictInt
    limit: StrictInt
    offset: StrictInt


class ClaimShareSummarySchema(StrictBaseModel):
    claim_id: UUID
    summary: StrictStr
    verdict: VerdictLabel
    language: LanguageLabel


class ClaimLookupResponseSchema(StrictBaseModel):
    claim: ClaimResponseSchema


class ClaimsPageFiltersSchema(StrictBaseModel):
    verdict: VerdictLabel | None = None
    language: LanguageLabel | None = None
    review_status: ReviewStatus | None = None
    limit: StrictInt = Field(default=20, ge=1, le=100)
    offset: StrictInt = Field(default=0, ge=0)
