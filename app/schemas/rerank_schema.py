from __future__ import annotations

from uuid import UUID

from pydantic import Field, StrictFloat, StrictInt, StrictStr

from app.schemas.common import StrictBaseModel


class EvidenceCandidateSchema(StrictBaseModel):
    source_id: UUID
    title: StrictStr
    url: StrictStr
    publisher: StrictStr | None = None
    language: StrictStr
    source_type: StrictStr
    snippet: StrictStr = Field(min_length=1)
    similarity_score: StrictFloat = Field(ge=0.0, le=1.0)
    rerank_score: StrictFloat | None = None
    initial_rank: StrictInt | None = None
    final_rank: StrictInt | None = None


class RerankResultSchema(StrictBaseModel):
    evidence_id: UUID
    rerank_score: StrictFloat
    final_rank: StrictInt = Field(ge=1)
