from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field, StrictFloat, StrictStr

from app.schemas.common import StrictBaseModel

VerdictLabel = Literal["Likely True", "Likely False", "Misleading", "Not Enough Evidence"]
LanguageLabel = Literal["Bangla", "English", "Hindi", "Banglish", "Hinglish", "Mixed", "Unknown"]
ConfidenceLabel = Literal["Low", "Medium", "High"]


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


class ClaimExtractionSchema(StrictBaseModel):
    extracted_claim: StrictStr = Field(min_length=5)
    detected_language: StrictStr = Field(min_length=2)
    category: StrictStr = Field(min_length=2)


class SearchQueryGenerationSchema(StrictBaseModel):
    search_queries: list[StrictStr] = Field(min_length=1, max_length=6)


class ReasoningVerdictSchema(StrictBaseModel):
    verdict: VerdictLabel
    confidence: StrictFloat = Field(ge=0.0, le=1.0)
    confidence_label: ConfidenceLabel
    explanation: StrictStr = Field(min_length=1)
    user_response: StrictStr = Field(min_length=1)
    used_source_ids: list[UUID] = Field(default_factory=list)


class EvidenceSnippetSchema(StrictBaseModel):
    source_id: UUID
    title: StrictStr
    url: StrictStr
    publisher: StrictStr | None = None
    language: StrictStr
    source_type: StrictStr
    snippet: StrictStr
    similarity_score: StrictFloat | None = None
    rerank_score: StrictFloat | None = None
    initial_rank: int | None = None
    final_rank: int | None = None


class VerdictResponseSchema(StrictBaseModel):
    verdict: VerdictLabel
    confidence: StrictFloat = Field(ge=0.0, le=1.0)
    confidence_label: ConfidenceLabel
    explanation: StrictStr
    user_response: StrictStr
    evidence: list[EvidenceSnippetSchema] = Field(default_factory=list)


class VisionOCRSchema(StrictBaseModel):
    extracted_text: StrictStr = Field(min_length=1)
