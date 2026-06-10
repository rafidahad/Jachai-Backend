from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field, HttpUrl, StrictBool, StrictInt, StrictStr

from app.schemas.common import StrictBaseModel
from app.schemas.verdict_schema import LanguageLabel

SourceType = Literal["article", "fact_check", "government", "research", "social_post", "tavily_search", "other"]


class SourceIngestItemSchema(StrictBaseModel):
    title: StrictStr = Field(min_length=3, max_length=255)
    url: HttpUrl
    publisher: StrictStr | None = Field(default=None, max_length=255)
    language: LanguageLabel | None = None
    source_type: SourceType = "article"
    snippet: StrictStr = Field(min_length=10)
    text_content: StrictStr = Field(min_length=20)
    metadata: dict[str, object] = Field(default_factory=dict)


class SourceIngestRequestSchema(StrictBaseModel):
    items: list[SourceIngestItemSchema] = Field(min_length=1, max_length=100)


class SourceResponseSchema(StrictBaseModel):
    id: UUID
    title: StrictStr
    url: StrictStr
    publisher: StrictStr | None = None
    language: LanguageLabel | StrictStr
    source_type: StrictStr
    snippet: StrictStr
    text_content: StrictStr
    metadata: dict[str, object]
    embedding_available: StrictBool
    created_at: datetime
    updated_at: datetime


class SourceIngestResponseSchema(StrictBaseModel):
    items: list[SourceResponseSchema]
    created_count: StrictInt
    updated_count: StrictInt


class SourceListResponseSchema(StrictBaseModel):
    items: list[SourceResponseSchema]
    total: StrictInt
