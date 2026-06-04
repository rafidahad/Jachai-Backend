from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import StrictInt, StrictStr

from app.schemas.common import StrictBaseModel
from app.schemas.dashboard_schema import RecentClaimItemSchema
from app.schemas.verdict_schema import LanguageLabel


class RumorClusterSchema(StrictBaseModel):
    id: UUID
    topic_hash: StrictStr
    title: StrictStr
    language: LanguageLabel | StrictStr
    summary: StrictStr
    claim_count: StrictInt
    representative_claim_id: UUID | None = None
    created_at: datetime
    updated_at: datetime


class RumorClusterListResponseSchema(StrictBaseModel):
    items: list[RumorClusterSchema]
    total: StrictInt


class RumorClusterDetailResponseSchema(StrictBaseModel):
    cluster: RumorClusterSchema
    recent_claims: list[RecentClaimItemSchema]
