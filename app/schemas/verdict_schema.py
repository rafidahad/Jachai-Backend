from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field, StrictFloat, StrictStr

from app.schemas.common import StrictBaseModel

VerdictLabel = Literal["Likely True", "Likely False", "Misleading", "Not Enough Evidence"]
LanguageLabel = Literal["Bangla", "English", "Hindi", "Banglish", "Hinglish", "Mixed"]


class LLMVerdictSchema(StrictBaseModel):
    verdict: VerdictLabel
    confidence: StrictFloat = Field(ge=0.0, le=1.0)
    explanation: StrictStr = Field(min_length=1)
    reasoning: StrictStr = Field(min_length=1)
    summary: StrictStr = Field(min_length=1)
    source_ids: list[UUID] = Field(default_factory=list)


class EvidenceSnippetSchema(StrictBaseModel):
    source_id: UUID
    title: StrictStr
    url: StrictStr
    publisher: StrictStr | None = None
    language: StrictStr
    source_type: StrictStr
    snippet: StrictStr
    similarity_score: StrictFloat | None = None


class VerdictResponseSchema(StrictBaseModel):
    verdict: VerdictLabel
    confidence: StrictFloat = Field(ge=0.0, le=1.0)
    explanation: StrictStr
    reasoning: StrictStr
    summary: StrictStr
    evidence: list[EvidenceSnippetSchema] = Field(default_factory=list)
