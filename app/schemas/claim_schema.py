from __future__ import annotations
from datetime import datetime
from typing import Literal, Any
from uuid import UUID
from pydantic import Field, StrictBool, StrictInt, StrictStr, StrictFloat
from app.schemas.common import StrictBaseModel
from app.schemas.verdict_schema import EvidenceSnippetSchema, LanguageLabel, ConfidenceLabel
from app.schemas.ai_schema import AIUsageSchema

class VerifyRequestOptions(StrictBaseModel):
    max_search_results: StrictInt = 8
    max_sources_to_fetch: StrictInt = 5
    max_evidence_chunks: StrictInt = 12
    require_citations: StrictBool = True

class VerifyRequest(StrictBaseModel):
    input_type: Literal["text", "url", "image_ocr"]
    content: StrictStr = Field(..., min_length=1)
    language: Literal["auto", "en", "bn"] = "auto"
    options: VerifyRequestOptions = Field(default_factory=VerifyRequestOptions)

class ClaimDetailsSchema(StrictBaseModel):
    original_input: StrictStr
    normalized_claim: StrictStr
    claim_type: StrictStr
    entities: list[StrictStr] = Field(default_factory=list)
    time_context: StrictStr | None = None
    location_context: StrictStr | None = None
    requires_freshness: StrictBool

class EvidenceDetailsSchema(StrictBaseModel):
    evidence_id: StrictStr
    title: StrictStr
    url: StrictStr
    domain: StrictStr
    published_date: StrictStr | None = None
    passage: StrictStr
    stance: Literal["supports", "refutes", "neutral", "background"]
    relevance_score: StrictFloat
    credibility_score: StrictFloat
    snippet_only: StrictBool

class SourceDetailsSchema(StrictBaseModel):
    source_id: StrictStr
    title: StrictStr
    url: StrictStr
    domain: StrictStr
    source_type: Literal["official", "government", "academic", "reputable_news", "fact_check", "primary_source", "social_media", "blog", "unknown", "low_quality"]
    credibility_score: StrictFloat
    credibility_reason: StrictStr

class SearchQueryDetailsSchema(StrictBaseModel):
    query: StrictStr
    purpose: Literal["general", "official", "refutation", "recent", "background"]

class VerifyResponse(StrictBaseModel):
    claim_id: StrictStr | None = None
    verdict: Literal["supported", "refuted", "misleading", "partially_true", "outdated", "insufficient_evidence", "unverifiable"]
    confidence: StrictFloat
    claim: ClaimDetailsSchema
    explanation: StrictStr
    evidence: list[EvidenceDetailsSchema] = Field(default_factory=list)
    sources: list[SourceDetailsSchema] = Field(default_factory=list)
    search_queries: list[SearchQueryDetailsSchema] = Field(default_factory=list)
    warnings: list[StrictStr] = Field(default_factory=list)
    debug: dict[StrictStr, Any] | None = None

class ReviewStatusUpdateRequest(StrictBaseModel):
    review_status: Literal["pending", "reviewed", "flagged"]
    reviewer: StrictStr = Field(min_length=1, max_length=64)

class ClaimResponseSchema(StrictBaseModel):
    id: UUID
    cluster_id: UUID | None = None
    input_type: Literal["text", "image", "url", "image_ocr"]
    source_url: StrictStr | None = None
    raw_text: StrictStr
    cleaned_text: StrictStr
    masked_text: StrictStr
    normalized_hash: StrictStr
    language: LanguageLabel
    extracted_claim: StrictStr
    detected_language: StrictStr
    category: StrictStr
    review_status: Literal["pending", "reviewed", "flagged"]
    verdict: StrictStr
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

class ClaimListResponseSchema(StrictBaseModel):
    items: list[ClaimResponseSchema]
    total: StrictInt
    limit: StrictInt
    offset: StrictInt

class ClaimShareSummarySchema(StrictBaseModel):
    claim_id: UUID
    summary: StrictStr
    verdict: StrictStr
    language: LanguageLabel

class ClaimLookupResponseSchema(StrictBaseModel):
    claim: ClaimResponseSchema


class VerificationJobSchema(StrictBaseModel):
    id: UUID
    claim_id: UUID | None = None
    input_type: Literal["text", "image", "url", "image_ocr"]
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

