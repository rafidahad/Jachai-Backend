from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field, HttpUrl, StrictBool, StrictInt, StrictStr, field_validator

from app.schemas.ai_schema import AIUsageSchema
from app.schemas.common import StrictBaseModel
from app.schemas.verdict_schema import PIPELINE_TO_LEGACY_VERDICT, ConfidenceLabel, EvidenceSnippetSchema, LanguageLabel, VerdictLabel

_LEGACY_VERDICTS: frozenset[str] = frozenset(VerdictLabel.__args__)  # type: ignore[attr-defined]


def _coerce_verdict(v: object) -> object:
    """Normalize old pipeline verdict labels to legacy VerdictLabel values."""
    if isinstance(v, str):
        if v in _LEGACY_VERDICTS:
            return v
        if v in PIPELINE_TO_LEGACY_VERDICT:
            return PIPELINE_TO_LEGACY_VERDICT[v]
        return "Not Enough Evidence"
    return v

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
    extracted_claim: StrictStr
    detected_language: StrictStr
    category: StrictStr
    review_status: ReviewStatus
    verdict: VerdictLabel
    confidence: float
    confidence_label: ConfidenceLabel
    explanation: StrictStr
    user_response: StrictStr
    reasoning: StrictStr
    share_summary: StrictStr
    created_at: datetime
    updated_at: datetime
    evidence: list[EvidenceSnippetSchema] = Field(default_factory=list)
    ai_usage: AIUsageSchema | None = None
    context_payload: dict[str, object] = Field(default_factory=dict)

    @field_validator("verdict", mode="before")
    @classmethod
    def normalize_verdict(cls, v: object) -> object:
        return _coerce_verdict(v)


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

    @field_validator("verdict", mode="before")
    @classmethod
    def normalize_verdict(cls, v: object) -> object:
        return _coerce_verdict(v)


class ClaimLookupResponseSchema(StrictBaseModel):
    claim: ClaimResponseSchema


class ClaimsPageFiltersSchema(StrictBaseModel):
    verdict: VerdictLabel | None = None
    language: LanguageLabel | None = None
    review_status: ReviewStatus | None = None
    limit: StrictInt = Field(default=20, ge=1, le=100)
    offset: StrictInt = Field(default=0, ge=0)
