from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field, StrictFloat, StrictStr

from app.schemas.common import StrictBaseModel

# ── Legacy verdict labels (kept for DB + frontend backward compat) ─────────────
VerdictLabel = Literal["Likely True", "Likely False", "Misleading", "Not Enough Evidence"]
LanguageLabel = Literal["Bangla", "English", "Hindi", "Banglish", "Hinglish", "Mixed", "Unknown"]
ConfidenceLabel = Literal["Low", "Medium", "High"]

# ── New pipeline verdict labels ────────────────────────────────────────────────
PipelineVerdictLabel = Literal[
    "supported",
    "refuted",
    "misleading",
    "partially_true",
    "outdated",
    "insufficient_evidence",
    "unverifiable",
]

# Maps new pipeline labels → legacy labels (for backward compat)
PIPELINE_TO_LEGACY_VERDICT: dict[str, VerdictLabel] = {
    "supported": "Likely True",
    "refuted": "Likely False",
    "misleading": "Misleading",
    "partially_true": "Misleading",
    "outdated": "Misleading",
    "insufficient_evidence": "Not Enough Evidence",
    "unverifiable": "Not Enough Evidence",
}

ALLOWED_PIPELINE_VERDICTS: set[str] = set(PipelineVerdictLabel.__args__)  # type: ignore[attr-defined]


class LLMVerdictSchema(StrictBaseModel):
    extracted_claim: StrictStr = Field(min_length=5)
    detected_language: StrictStr = Field(min_length=2)
    category: StrictStr = Field(min_length=2)
    verdict: VerdictLabel
    confidence: StrictFloat = Field(ge=0.0, le=1.0)
    confidence_label: ConfidenceLabel
    explanation: StrictStr = Field(min_length=1)
    user_response: StrictStr = Field(min_length=1)
    used_source_ids: list[UUID] = Field(default_factory=list)
    # New field — carries the richer pipeline label alongside the legacy label
    pipeline_verdict: PipelineVerdictLabel | None = None
    warnings: list[str] = Field(default_factory=list)


class ClaimExtractionSchema(StrictBaseModel):
    extracted_claim: StrictStr = Field(min_length=5)
    detected_language: StrictStr = Field(min_length=2)
    category: StrictStr = Field(min_length=2)
    # New richer fields (optional so fallback works without them)
    entities: list[str] = Field(default_factory=list)
    time_context: str | None = None
    location_context: str | None = None
    requires_freshness: bool = False
    verification_strategy: str = "general_web"
    detected_claims: list[dict] = Field(default_factory=list)


class SearchQuerySchema(StrictBaseModel):
    """Single typed search query with purpose and priority."""
    query: StrictStr = Field(min_length=3)
    purpose: Literal["general", "official", "refutation", "recent", "background"] = "general"
    priority: int = Field(default=1, ge=1)


class SearchQueryGenerationSchema(StrictBaseModel):
    queries: list[SearchQuerySchema] = Field(default_factory=list)
    # Legacy flat list — populated from queries for backward compat
    search_queries: list[StrictStr] = Field(default_factory=list)


class EvidenceChunkSchema(StrictBaseModel):
    """A passage-level evidence chunk extracted from a fetched source."""
    chunk_id: StrictStr
    source_id: StrictStr
    title: StrictStr
    url: StrictStr
    domain: StrictStr
    published_date: str | None = None
    text: StrictStr
    snippet_only: bool = False


class RankedChunkSchema(StrictBaseModel):
    """Evidence chunk after ranking/reranking."""
    chunk_id: StrictStr
    source_id: StrictStr
    title: StrictStr
    url: StrictStr
    domain: StrictStr
    published_date: str | None = None
    text: StrictStr
    snippet_only: bool = False
    relevance_score: float = Field(ge=0.0, le=1.0, default=0.0)
    ranking_reason: str = ""


class ClassifiedChunkSchema(StrictBaseModel):
    """Evidence chunk after LLM stance classification."""
    chunk_id: StrictStr
    source_id: StrictStr
    title: StrictStr
    url: StrictStr
    domain: StrictStr
    published_date: str | None = None
    text: StrictStr
    snippet_only: bool = False
    relevance_score: float = Field(ge=0.0, le=1.0, default=0.0)
    ranking_reason: str = ""
    stance: Literal["supports", "refutes", "neutral", "background"] = "neutral"
    stance_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    rationale: str = ""
    quoted_evidence: str = ""


class SourceCredibilitySchema(StrictBaseModel):
    """Per-source credibility assessment."""
    source_id: StrictStr
    domain: StrictStr
    source_type: Literal[
        "official", "government", "academic", "reputable_news", "fact_check",
        "primary_source", "social_media", "blog", "unknown", "low_quality"
    ] = "unknown"
    credibility_score: float = Field(ge=0.0, le=1.0, default=0.5)
    credibility_reason: str = ""
    snippet_only: bool = False


class ReasoningVerdictSchema(StrictBaseModel):
    verdict: VerdictLabel
    confidence: StrictFloat = Field(ge=0.0, le=1.0)
    confidence_label: ConfidenceLabel
    explanation: StrictStr = Field(min_length=1)
    user_response: StrictStr = Field(min_length=1)
    used_source_ids: list[UUID] = Field(default_factory=list)
    pipeline_verdict: PipelineVerdictLabel | None = None
    key_evidence_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class EvidenceSnippetSchema(StrictBaseModel):
    source_id: UUID
    title: StrictStr
    url: StrictStr
    publisher: StrictStr | None = None
    language: StrictStr
    source_type: StrictStr
    snippet: StrictStr
    similarity_score: StrictFloat | None = None
    match_score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    rerank_score: StrictFloat | None = None
    search_score: StrictFloat | None = None
    trust_score: StrictFloat | None = None
    provider: StrictStr | None = None
    initial_rank: int | None = None
    final_rank: int | None = None
    # New fields
    stance: str | None = None
    credibility_score: float | None = None
    snippet_only: bool = False


class VerdictResponseSchema(StrictBaseModel):
    verdict: VerdictLabel
    confidence: StrictFloat = Field(ge=0.0, le=1.0)
    confidence_label: ConfidenceLabel
    explanation: StrictStr
    user_response: StrictStr
    evidence: list[EvidenceSnippetSchema] = Field(default_factory=list)


class VisionOCRSchema(StrictBaseModel):
    extracted_text: StrictStr = Field(min_length=1)
