from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import StrictInt, StrictStr

from app.schemas.common import StrictBaseModel
from app.schemas.verdict_schema import LanguageLabel, VerdictLabel


class SummaryMetricSchema(StrictBaseModel):
    total_claims: StrictInt
    claims_today: StrictInt
    reviewed_claims: StrictInt
    total_sources: StrictInt
    total_clusters: StrictInt


class DistributionItemSchema(StrictBaseModel):
    label: StrictStr
    count: StrictInt


class ClaimsOverTimeItemSchema(StrictBaseModel):
    date: date
    count: StrictInt


class RecentClaimItemSchema(StrictBaseModel):
    id: StrictStr
    verdict: VerdictLabel
    language: LanguageLabel | StrictStr
    review_status: Literal["pending", "reviewed", "flagged"]
    created_at: datetime
    share_summary: StrictStr


class DashboardSummaryResponseSchema(StrictBaseModel):
    metrics: SummaryMetricSchema


class DashboardDistributionResponseSchema(StrictBaseModel):
    items: list[DistributionItemSchema]


class DashboardClaimsOverTimeResponseSchema(StrictBaseModel):
    items: list[ClaimsOverTimeItemSchema]


class DashboardRecentClaimsResponseSchema(StrictBaseModel):
    items: list[RecentClaimItemSchema]
